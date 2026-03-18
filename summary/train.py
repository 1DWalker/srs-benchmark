from collections import defaultdict
import hashlib
from pathlib import Path
from time import time
from typing import Dict, NamedTuple
import lmdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, root_mean_squared_error
import torch
import wandb
from rwkv.utils import get_number_of_trainable_parameters
import random

from summary import decoder_ops, fsrs_encoder_model
from summary.model import Model
from utils import compact_lmdb, load_tensor, parse_toml

WEIGHT_DECAY = 1e-2
ADAMW_BETAS = (0.90, 0.999)
ADAMW_EPS = 1e-8
CLIP = 2

def log_model(log, model, device):
    with torch.no_grad():
        for name, param in model.named_parameters():
            log[f"model/{name}.data.mean"] = param.mean().item()
            if param.numel() > 1:
                log[f"model/{name}.data.std"] = param.std().item()
            log[f"model/{name}.data.min"] = param.min().item()
            log[f"model/{name}.data.max"] = param.max().item()
            log[f"model/{name}.data.25th"] = torch.quantile(param, 0.25).item()
            log[f"model/{name}.data.50th"] = torch.quantile(param, 0.50).item()
            log[f"model/{name}.data.75th"] = torch.quantile(param, 0.75).item()
            if param.grad is not None:
                log[f"model/{name}.grad.mean"] = param.grad.mean().item()
                if param.numel() > 1:
                    log[f"model/{name}.grad.std"] = param.grad.std().item()
                log[f"model/{name}.grad.min"] = param.grad.min().item()
                log[f"model/{name}.grad.max"] = param.grad.max().item()
                log[f"model/{name}.grad.25th"] = torch.quantile(param.grad, 0.25).item()
                log[f"model/{name}.grad.50th"] = torch.quantile(param.grad, 0.50).item()
                log[f"model/{name}.grad.75th"] = torch.quantile(param.grad, 0.75).item()

def get_optimizer(config, model):
    exclude_names = model.get_excluded_params()
    found_exclude = set()
    params_by_width = defaultdict(list)
    param_width = {}
    param_groups = []
    for name, param in model.named_parameters():
        width = param.shape[-1]
        param_width[name] = width

    for name, param in model.named_parameters():
        width = param.shape[-1]
        if "bias" in name[-4:]:
            search_name = name[:-4] + "weight"
            params_by_width[param_width[search_name]].append((name, param))
        else:
            params_by_width[width].append((name, param))

    for width, named_parameters in params_by_width.items():
        decay_params = []
        combine_params = []
        non_weight_params = []
        exclude_params = []
        print("WIDTH:", width)
        for name, param in named_parameters:
            in_exclude_names = np.array([x in name for x in exclude_names]).any()
            if in_exclude_names and "weight" not in name:
                exclude_params.append(param)
                found_exclude.add(name)
                print("Exclude:", name, param.shape)
            elif "weight" not in name:
                print("Skip decay:", name, param.shape)
                non_weight_params.append(param)
            elif "combine" in name:
                print("Combine:", name)
                combine_params.append(param)
            else:
                print("Decay:", name, param.shape)
                decay_params.append(param)

        lr = 16 * config.PEAK_LR / width
        # if width < 16:
        #     lr /= 10 / 1.5
        #     print("Reducing LR for width:", width)
        print("Width:", width, "lr:", lr, len(decay_params), len(non_weight_params))
        print()
        param_groups.append(
            {
                "params": decay_params,
                "weight_decay": WEIGHT_DECAY,
                "lr": lr,
            }
        )
        param_groups.append(
            {
                "params": combine_params,
                "weight_decay": WEIGHT_DECAY,
                "lr": lr / 2,
            }
        )
        param_groups.append(
            {
                "params": non_weight_params,
                "weight_decay": WEIGHT_DECAY / 10,
                "lr": lr,
            }
        )
        # param_groups.append(
        #     {
        #         "params": exclude_params,
        #         "weight_decay": WEIGHT_DECAY / 100,
        #         "lr": lr,
        #     }
        # )

    assert len(found_exclude) == len(exclude_names)
    return torch.optim.AdamW(
        param_groups,
        eps=ADAMW_EPS,
        betas=ADAMW_BETAS,
        fused=True,
    ), param_groups[0]['lr']

class ValidateResult(NamedTuple):
    loss_weighted_review: float
    loss_weighted_user: float
    cond_loss: float
    first_bce: float
    first_ce: float

def validate(model, summary_txn, label_filter_txn, validate_users, config, model_mode="eval", log=None):
    device = config.DEVICE
    torch.cuda.empty_cache()
    try:
        tot_loss = 0
        tot_loss_n = 0
        losses = []
        cond_losses = []
        first_bce_scores = []
        first_ce_scores = []
        for user in validate_users:
            reserved = torch.cuda.memory_reserved()
            if reserved >= config.THRESHOLD_RESERVED_GB * 1024 ** 3:
                print(f"Reserved: {reserved / (1024 ** 3):.3f} GB. Emptying cache.")
                torch.cuda.empty_cache()

            if model_mode == "eval":
                model.eval()
            else:
                model.train()
            batches = decoder_ops.get_data(summary_txn, user, config.DEVICE)
            T = decoder_ops.extract_num_reviews(batches)
            splits = load_tensor(label_filter_txn, f"{user}_split", device=device).tolist()
            equalize_review_ths = load_tensor(label_filter_txn, f"{user}_review_ths", device).tolist()
            rmse_bins = load_tensor(label_filter_txn, f"{user}_rmse_bins", device).tolist()
            assert len(splits) == 6
            with torch.inference_mode():
                max_review_th = []
                for split_i in range(len(splits) - 1):
                    max_review_th.append(splits[split_i] - 1)

                max_review_th_s = torch.tensor(max_review_th, device=device)
                min_review_th_s = torch.ones_like(max_review_th_s)
                encoding_s = decoder_ops.encode(batches, model.encoder_model, min_review_th_s=min_review_th_s, max_review_th_s=max_review_th_s)
                loss, loss_n, cond_loss, cond_n, rmse_raw, rmse_bins, auc = decoder_ops.decode_full(batches, model.card_model, encoding_s, splits, equalize_review_ths, rmse_bins, device=device, equalize_test_reviews=True)
                tot_loss += loss * loss_n
                tot_loss_n += loss_n
                losses.append(loss.item())
                cond_losses.append(cond_loss.item())

                first_stats_accum = None
                for i in range(encoding_s.size(0)):
                    encoding = encoding_s[i]
                    first_stats = decoder_ops.first_decode(batches, model.first_review_model, encoding, splits[i], min(T, splits[i + 1] - 1))
                    if first_stats_accum is None:
                        first_stats_accum = first_stats
                    else:
                        first_stats_accum = decoder_ops.combine_decode_results(first_stats_accum, first_stats)

                if log is not None:
                    log[f"validate_user_loss/user_{user}"] = loss

                first_bce_loss = first_stats_accum.bce_loss_sum.item() / (1e-7 + first_stats_accum.loss_n.item())
                first_ce_loss = first_stats_accum.ce_loss_sum.item() / (1e-7 + first_stats_accum.loss_n.item())
                first_bce_scores.append(first_bce_loss)
                first_ce_scores.append(first_ce_loss)

                print()
                print(f"User: {user}, RMSE: {rmse_raw:.6f}, LogLoss: {loss:.6f}, RMSE (bins): {rmse_bins:.6f}, AUC: {(-1 if auc is None else auc):.6f}, size: {loss_n}")
                print(f"Cond CE: {cond_loss.item():.6f}, n: {int(cond_n)}")
                print(f"First learn: LogLoss: {first_bce_loss:.3f}, CE: {first_ce_loss:.3f}")
                for split_i, parameters in enumerate(encoding_s):
                    first_rating_dist = [round(x, 2) for x in decoder_ops.extract_first_review_dist(model.first_review_model, encoding_s[split_i]).cpu().tolist()]
                    # fsrs_params = model.card_model.encoding_to_fsrs(parameters).cpu().numpy()
                    # print(f"Split: {split_i}, first rating: {first_rating_dist}, params: {list(map(lambda x: round(float(x), 4), fsrs_params.tolist()))}")
                    print(f"Split: {split_i}, first rating: {first_rating_dist}")
    except Exception as e:
        print(e)
        raise e

    user_avg_loss = np.array(losses).mean()
    print(f"Mean validation loss: by review: {tot_loss / tot_loss_n:.4f}, by user: {user_avg_loss:.4f}")
    cond_loss = np.mean(cond_losses)
    print(f"Conditional loss: {cond_loss:.4f}")
    first_bce = np.mean(first_bce_scores)
    first_ce = np.mean(first_ce_scores)
    print(f"First BCE mean: {first_bce:.4f}, CE mean: {first_ce:.4f}")
    return ValidateResult(
        loss_weighted_review=tot_loss / tot_loss_n,
        loss_weighted_user=user_avg_loss,
        cond_loss=cond_loss,
        first_bce=first_bce,
        first_ce=first_ce,
    )

def get_split(n):
    assert n >= 12
    # k = min([random.randint(max(100, int(0.2 * n)), n) for _ in range(2)])
    k = random.randint(max(100, int(0.2 * n)), n)
    # r = l + k - 1 <= n-1 implies l <= n - k
    # l = random.randint(0, n - k)

    # setting l to a random value has a huge negative effect, it struggles to overfit on 2 users
    l = 0
    return l, l + k - 1

def generate_subsplits(l, r, T):
    g = (r - l + 1) // 6
    linspace = np.linspace((l + r) // 2, r - g)
    floored_linspace = np.floor(linspace).astype(int)
    return np.random.choice(floored_linspace)

def get_superiority(lagging_dict, now_dict):
    n = 0
    w = 0
    for user in lagging_dict.keys():
        n += 1
        if lagging_dict[user] > now_dict[user]:
            w += 1
    return w / max(1, n)

def cosine_down(step, total_steps, base=1e-3):
    return base + (1 - base) * (1 + np.cos(0.5 * np.pi * (1 + step / total_steps)))

def make_iter_seed(base_seed: int, step: int) -> int:
    combined = f"{base_seed}_{step}".encode()
    digest = hashlib.sha256(combined).digest()
    return int.from_bytes(digest[:4], "big")

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def main(config):
    seed = config.SEED + (len(config.TRAIN_MODE) if config.TRAIN_MODE != "WSD" else 0)
    set_seed(seed)
    model = Model().to(config.DEVICE)
    optimizer, peak_lr_first_param = get_optimizer(config, model)
    encoder_model_params = get_number_of_trainable_parameters(model.encoder_model)
    # encoder_model_global_params = get_number_of_trainable_parameters(model.encoder_model.intermediate_global_encoders) + get_number_of_trainable_parameters(model.encoder_model.last_global_encoder)
    encoder_model_global_params = 0
    for name, param in model.encoder_model.named_parameters():
        if param.requires_grad and "global" in name and "weight_linear" not in name and "value_linear" not in name:
            encoder_model_global_params += param.numel()
    encoder_model_card_parallel_params = encoder_model_params - encoder_model_global_params
    decoder_model_global_params = 0
    for name, param in model.card_model.named_parameters():
        if param.requires_grad and "global" in name and "weight_linear" not in name and "value_linear" not in name:
            decoder_model_global_params += param.numel()
    decoder_model_params = get_number_of_trainable_parameters(model.card_model)
    decoder_model_card_parallel_params = decoder_model_params - decoder_model_global_params
    forgetting_curve_params = get_number_of_trainable_parameters(model.card_model.forgetting_curve_nn)
    for name, param in model.card_model.forgetting_curve_nn.named_parameters():
        if param.requires_grad and "global" in name and "weight_linear" not in name and "value_linear" not in name:
            forgetting_curve_params -= param.numel()
    # card_model_params = get_number_of_trainable_parameters(model.card_model)
    # curve_params = get_number_of_trainable_parameters(model.card_model.forgetting_curve_nn)
    # print(f"Number of trainable parameters: {encoder_model_params + card_model_params}, Encoder: parallel: {encoder_model_card_parallel_params}, global: {encoder_model_global_params}, total: {encoder_model_params}, Card: {card_model_params}, Forgetting curve: {curve_params}")

    if config.TRAIN_MODE == "WSD":
        start_factor = max(1e-8, config.WARMUP_START_LR / config.PEAK_LR)
        start_lr = config.PEAK_LR  # Scheduler handles it
        warmup_steps = config.WARMUP_STEPS
        decay_steps = int(0.1 * config.TOTAL_STEPS)
        constant_steps = config.TOTAL_STEPS - warmup_steps - decay_steps
        print("Warmup steps:", warmup_steps)
        print("Constant steps:", constant_steps)
        print("Decay steps:", decay_steps)
        print(start_factor, start_lr)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps
        )
        main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        decay_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda t: cosine_down(t, decay_steps)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler, decay_scheduler],
            milestones=[warmup_steps, warmup_steps + constant_steps],
        )
    elif config.TRAIN_MODE == "D":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda t: cosine_down(t, config.TOTAL_STEPS)
        )
    else:
        raise ValueError(f"Invalid train mode: {config.TRAIN_MODE}")

    if config.LOAD_MODEL:
        model_path = f"{config.LOAD_MODEL_FOLDER}/{config.LOAD_MODEL_NAME}.pth"
        optim_path = f"{config.LOAD_MODEL_FOLDER}/{config.LOAD_MODEL_NAME}_optim.pth"
        print("Loading model:", model_path)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        optimizer_state = torch.load(optim_path, weights_only=True)
        if config.TRAIN_MODE == "WSD":
            for init_group, load_group in zip(optimizer.state_dict()["param_groups"], optimizer_state["param_groups"]):
                load_group["lr"] = init_group["lr"]
        optimizer.load_state_dict(optimizer_state)
        # for init_group, load_group in zip(optimizer.state_dict()["param_groups"], optimizer_state["param_groups"]):
        #     print(init_group["lr"])
        # exit()
        print("Loaded model:", model_path)
    else:
        print("No model loaded.")

    train_users = list(range(config.TRAIN_USER_START, config.TRAIN_USER_END + 1))
    validate_overfit_users = list(range(config.VALIDATE_OVERFIT_USER_START, config.VALIDATE_OVERFIT_USER_END + 1))
    for user in validate_overfit_users:
        assert user in train_users
    validate_users = list(range(config.VALIDATE_USER_START, config.VALIDATE_USER_END + 1))
    for user in validate_users:
        assert user not in train_users

    compressed_db_env = lmdb.open(config.SUMMARY_DB_PATH, readonly=True, lock=False)
    label_filter_env = lmdb.open(config.LABEL_FILTER_DB_PATH, readonly=True, lock=False)

    print(f"encoder params total: {encoder_model_params}, enc parallel: {encoder_model_card_parallel_params}, enc global: {encoder_model_global_params}")
    print(f"decoder params total: {decoder_model_params}, dec parallel: {decoder_model_card_parallel_params}, dec global: {decoder_model_global_params}, forgetting curve: {forgetting_curve_params}")
    if config.USE_WANDB:
        wandb_config = {
            "peak_lr": config.PEAK_LR,
            "adamw_betas": ADAMW_BETAS,
            "adamw_eps": ADAMW_EPS,
            "weight_decay": WEIGHT_DECAY,
            "clip": CLIP,
            "user_start": config.TRAIN_USER_START,
            "user_end": config.TRAIN_USER_END,
            "seed": config.SEED,
            "parameters": get_number_of_trainable_parameters(model),
            "params_encoder_total": encoder_model_params,
            "params_encoder_parallel": encoder_model_card_parallel_params,
            "params_encoder_global": encoder_model_global_params,
            "params_decoder_total": decoder_model_params,
            "params_decoder_parallel": decoder_model_card_parallel_params,
            "params_decoder_global": decoder_model_global_params,
            "params_forgetting_curve": forgetting_curve_params,
        }
        wandb_kwargs = dict(
            project=config.WANDB_PROJECT_NAME,
            config=wandb_config,
            name=config.WANDB_RUN_NAME,
            id=config.WANDB_RUN_NAME,
        )
        if config.WANDB_RESUME_MODE != "never":
            wandb_kwargs["resume"] = config.WANDB_RESUME_MODE
        wandb.init(**wandb_kwargs)

    train_review_loss_dict = {}
    train_review_loss_bce_dict = {}
    lagging_train_review_loss_dict = {}
    lagging_train_review_loss_bce_dict = {}
    train_first_loss_dict = {}
    train_first_loss_bce_dict = {}
    step = 0
    time_start = time()
    with compressed_db_env.begin(write=False) as summary_txn:
        with label_filter_env.begin(write=False) as label_filter_txn:
            for epoch in range(int(1e9)):
                for user_i in range(len(train_users)):
                    log = {}
                    log["epoch"] = epoch
                    step += 1

                    iter_seed = make_iter_seed(base_seed=config.SEED, step=step)
                    set_seed(iter_seed)

                    if user_i == 0:
                        random.shuffle(train_users)

                    if step < config.START_STEP:
                        if config.START_SCHEDULER_AT_START_STEP:
                            scheduler.step()
                        continue

                    user = train_users[user_i]

                    validate_iter = (step + 1) % config.VALIDATE_STEPS == 0
                    current_lr = scheduler.get_last_lr()[0] / peak_lr_first_param * config.PEAK_LR
                    log["lr"] = current_lr

                    model.train()
                    encode_batches = decoder_ops.get_data(summary_txn, user, config.DEVICE)
                    if config.USE_MAX_SEQ_LEN:
                        if random.random() < 0.9:
                            decode_max_L =  64
                        else:
                            decode_max_L = int(1e9)
                    else:
                        decode_max_L = int(1e9)
                    decode_batches = decoder_ops.get_data(summary_txn, user, config.DEVICE, max_L=decode_max_L, merge=config.MERGE_DECODE_BATCHES)
                    T = decoder_ops.extract_num_reviews(encode_batches)
                    if T > config.SKIP_LENGTH:
                        print()
                        print(f"Skipping: {user}, size: {T}")
                    else:
                        try:
                            train_l, test_r = get_split(T)
                            test_l = generate_subsplits(train_l, test_r, T)
                            train_r = test_l - 1
                            encoding = decoder_ops.encode_single(encode_batches, model.encoder_model, train_l + 1, train_r + 1, log=log)
                            # fsrs_params = model.card_model.encoding_to_fsrs(encoding)
                            fsrs_params = None

                            review_stats: decoder_ops.DecodeResult = decoder_ops.decode(decode_batches, model.card_model, encoding, min_review_th=test_l + 1, max_review_th=test_r + 1)
                            first_review_logits = decoder_ops.extract_first_review_dist_logits(model.first_review_model, encoding)
                            first_stats: decoder_ops.DecodeResult = decoder_ops.first_logits(decode_batches, first_review_logits, min_review_th=test_l + 1, max_review_th=test_r + 1)
                            review_loss_ce_avg = review_stats.ce_loss_sum / (1e-7 + review_stats.loss_n)
                            review_loss_bce_avg = review_stats.bce_loss_sum / (1e-7 + review_stats.loss_n)
                            review_n = review_stats.loss_n
                            first_loss_ce_avg = first_stats.ce_loss_sum / (1e-7 + first_stats.loss_n)
                            first_loss_bce_avg = first_stats.bce_loss_sum / (1e-7 + first_stats.loss_n)
                            first_n = first_stats.loss_n

                            ce_avg = (review_stats.ce_loss_sum + first_stats.ce_loss_sum) / (1e-7 + review_n + first_n)
                            bce_avg = (review_stats.bce_loss_sum + first_stats.bce_loss_sum) / (1e-7 + review_n + first_n)
                            tot_loss = bce_avg + 1e-1 * ce_avg

                            if user in train_review_loss_dict:
                                lagging_train_review_loss_dict[user] = train_review_loss_dict[user]
                            train_review_loss_dict[user] = review_loss_ce_avg.item()
                            if user in train_review_loss_bce_dict:
                                lagging_train_review_loss_bce_dict[user] = train_review_loss_bce_dict[user]
                            train_review_loss_bce_dict[user] = review_loss_bce_avg.item()
                            train_first_loss_dict[user] = first_loss_ce_avg.item()
                            train_first_loss_bce_dict[user] = first_loss_bce_avg.item()

                            if review_n < 100:
                                print("Skipping: label sizes are too small:", review_n.item())
                            elif review_loss_ce_avg.requires_grad:
                                if step % 10 == 0 and step - config.START_STEP >= len(train_users):
                                    log["train_loss_bce_avg"] = np.mean(np.array(list(train_review_loss_bce_dict.values())))
                                    log["train_loss_avg"] = np.mean(np.array(list(train_review_loss_dict.values())))
                                    log["train_first_loss_bce_avg"] = np.mean(np.array(list(train_first_loss_bce_dict.values())))
                                    log["train_first_loss_avg"] = np.mean(np.array(list(train_first_loss_dict.values())))
                                    if len(lagging_train_review_loss_bce_dict) == len(train_review_loss_bce_dict):
                                        log["superiority/bce_epoch_superiority"] = get_superiority(lagging_train_review_loss_bce_dict, train_review_loss_bce_dict)
                                        log["superiority/epoch_superiority"] = get_superiority(lagging_train_review_loss_dict, train_review_loss_dict)
                                log["encoding/encoding_std"] = encoding.std()
                                if fsrs_params is not None:
                                    for i, x in enumerate(fsrs_params.tolist()):
                                        i_str = "0" * (2 - len(str(i))) + str(i)
                                        log[f"encoding_value/{i_str}"] = x
                                
                                tot_loss.backward()
                                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                                log["grad_norm"] = grad_norm

                                if step % 10 == 0:
                                    print()
                                    print(f"Indices:", train_l, test_l, test_r, train_r - train_l + 1, T, "User:", user)
                                    print(f"Review -- loss: ce: {review_loss_ce_avg:.3f}, bce: {review_loss_bce_avg:.3f}, n: {review_n}")
                                    print(f"First -- loss: ce: {first_loss_ce_avg:.3f}, bce: {first_loss_bce_avg:.3f}, n: {first_n}")
                                    if fsrs_params is not None:
                                        print(f"encoding: {list(map(lambda x: round(float(x), 4), fsrs_params.tolist()))}")
                                    print(f"First rating dist: {[round(x, 2) for x in torch.nn.functional.softmax(first_review_logits, dim=-1).cpu().tolist()]}")
                                    print(f"{step} {epoch}, user: {user}, loss_ce: {review_loss_ce_avg.item():.3f}, loss_bce: {review_loss_bce_avg.item():.3f}, n: {review_n}, grad norm: {grad_norm:.3f}, lr: {current_lr:.3e}")
                                    print(f"Elapsed: {time() - time_start:.3f}")
                                    time_start = time()
                                optimizer.step()
                                optimizer.zero_grad()

                                reserved = torch.cuda.memory_reserved()
                                if reserved >= config.THRESHOLD_RESERVED_GB * 1024 ** 3:
                                    print(f"Reserved: {reserved / (1024 ** 3):.3f} GB. T: {T}. Emptying cache.")
                                    torch.cuda.empty_cache()
                            else:
                                print("No grad required.")
                            log["nan/train_nan"] = 0
                        except Exception as e:
                            print(e)
                            log["nan/train_nan"] = 1
                            raise e
                
                    scheduler.step()
                    if step % 50 == 0:
                        log_model(log, model, device=config.DEVICE)

                    if validate_iter or (step + 1) % 5000 == 0:
                        save_model_path = (
                            f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_{step}.pth"
                        )
                        save_optim_path = f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_{step}_optim.pth"
                        Path(config.SAVE_MODEL_FOLDER).mkdir(parents=True, exist_ok=True)               
                        torch.save(model.state_dict(), save_model_path)
                        torch.save(optimizer.state_dict(), save_optim_path)
                        print("MODEL SAVED.")

                    if validate_iter:
                        validate_result: ValidateResult = validate(model, summary_txn, label_filter_txn, validate_users, config, model_mode="eval", log=log)
                        log["validation/validation_loss"] = validate_result.loss_weighted_review
                        log["validation/validation_loss_user"] = validate_result.loss_weighted_user
                        log["validation/cond_loss"] = validate_result.cond_loss
                        log["validation/first_bce"] = validate_result.first_bce
                        log["validation/first_ce"] = validate_result.first_ce
                        validate_result_train_mode: ValidateResult = validate(model, summary_txn, label_filter_txn, validate_users, config, model_mode="train", log=log)
                        log["validation_train_mode/validation_loss"] = validate_result_train_mode.loss_weighted_review
                        log["validation_train_mode/validation_loss_user"] = validate_result_train_mode.loss_weighted_user
                        validate_overfit_result: ValidateResult = validate(model, summary_txn, label_filter_txn, validate_overfit_users, config, model_mode="eval", log=log)
                        log["validation_overfit/validation_overfit_loss"] = validate_overfit_result.loss_weighted_review
                        log["validation_overfit/validation_overfit_loss_user"] = validate_overfit_result.loss_weighted_user
                        log["validation_overfit/cond_loss"] = validate_overfit_result.cond_loss
                        log["validation_overfit/first_bce"] = validate_overfit_result.first_bce
                        log["validation_overfit/first_ce"] = validate_overfit_result.first_ce

                    if config.USE_WANDB:
                        wandb.log(log, step=step, commit=True)
                    if step == config.TOTAL_STEPS - 1:
                        break
                if step == config.TOTAL_STEPS - 1:
                    break


if __name__ == '__main__':
    config = parse_toml()
    main(config)