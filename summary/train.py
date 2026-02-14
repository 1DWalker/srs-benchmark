from pathlib import Path
from typing import Dict, NamedTuple
import lmdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, root_mean_squared_error
import torch
import wandb
from rwkv.utils import get_number_of_trainable_parameters
import random

from summary import decoder_ops
from summary.model import Model
from utils import compact_lmdb, load_tensor, parse_toml

WEIGHT_DECAY = 1e-3
ADAMW_BETAS = (0.90, 0.999)
ADAMW_EPS = 1e-8
CLIP = 3

def log_model(log, model, device):
    with torch.no_grad():
        sizes = [300, 1000, 3000, 10000, 30000, 100000]
        model.eval()
        recency_const_s, recency_degree_s = model.encoder_model.train_size_to_recency_poly(torch.tensor(sizes, device=device))

        for size, recency_const, recency_degree in zip(sizes, recency_const_s.cpu().tolist(), recency_degree_s.cpu().tolist()):
            log[f"recency_const/recency_const_{size}"] = recency_const
            log[f"recency_degree/recency_degree_{size}"] = recency_degree

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
    decay_params = []
    other_params = []
    exclude_names = ["recency_nn_last_linear", "forgetting_curve_last_linear", "first_review_last_linear"] 
    found_exclude = set()
    for name, param in model.named_parameters():
        if "weight" not in name:
            print("Skip decay:", name)
            other_params.append(param)
        elif np.array([x in name for x in exclude_names]).any():
            other_params.append(param)
            found_exclude.add(name)
            print("Skip decay:", name)
        else:
            decay_params.append(param)

    assert len(decay_params) > 0
    assert len(other_params) > 0
    assert len(found_exclude) == len(exclude_names)
    return torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": WEIGHT_DECAY,
                "lr": config.PEAK_LR,
            },
            {"params": other_params, "weight_decay": 0.0, "lr": config.PEAK_LR},
        ],
        eps=ADAMW_EPS,
        betas=ADAMW_BETAS,
    )

class ValidateResult(NamedTuple):
    loss_weighted_review: float
    loss_weighted_user: float
    first_bce: float
    first_ce: float

def validate(model, summary_txn, label_filter_txn, validate_users, config, log=None):
    device = config.DEVICE
    torch.cuda.empty_cache()
    try:
        tot_loss = 0
        tot_loss_n = 0
        losses = []

        first_bce_scores = []
        first_ce_scores = []
        for user in validate_users:
            reserved = torch.cuda.memory_reserved()
            if reserved >= config.THRESHOLD_RESERVED_GB * 1024 ** 3:
                print(f"Reserved: {reserved / (1024 ** 3):.3f} GB. Emptying cache.")
                torch.cuda.empty_cache()

            model.eval()
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
                min_review_th_s = torch.zeros_like(max_review_th_s)
                encoding_s, sum_encoding_s = decoder_ops.encode(batches, model.encoder_model, min_review_th_s=min_review_th_s, max_review_th_s=max_review_th_s)
                loss, loss_n, rmse_raw, rmse_bins, auc = decoder_ops.decode_full(batches, model.card_model, encoding_s, splits, equalize_review_ths, rmse_bins, device=device, equalize_test_reviews=True)
                tot_loss += loss * loss_n
                tot_loss_n += loss_n
                losses.append(loss.item())

                first_stats_accum = None
                for i in range(encoding_s.size(0)):
                    encoding = encoding_s[i]
                    first_stats = decoder_ops.first_decode(batches, model.first_review_model, encoding, splits[i], splits[i + 1] - 1)
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
                print(f"First learn: LogLoss: {first_bce_loss:.3f}, CE: {first_ce_loss:.3f}")
                for split_i, parameters in enumerate(encoding_s.cpu().numpy()):
                    first_rating_dist = [round(x, 2) for x in decoder_ops.extract_first_review_dist(model.first_review_model, encoding_s[split_i]).cpu().tolist()]
                    print(f"Split: {split_i}, first rating: {first_rating_dist}, params: {list(map(lambda x: round(float(x), 4), parameters.tolist()))}")
    except Exception as e:
        print(e)
        raise e

    user_avg_loss = np.array(losses).mean()
    print(f"Mean validation loss: by review: {tot_loss / tot_loss_n:.4f}, by user: {user_avg_loss:.4f}")
    first_bce = np.mean(first_bce_scores)
    first_ce = np.mean(first_ce_scores)
    print(f"First BCE mean: {first_bce:.4f}, CE mean: {first_ce:.4f}")
    return ValidateResult(
        loss_weighted_review=tot_loss / tot_loss_n,
        loss_weighted_user=user_avg_loss,
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

def cosine_down(step, total_steps):
    return 1 + np.cos(0.5 * np.pi * (1 + step / total_steps))


def main(config):
    seed = config.SEED + config.START_STEP
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(config.SEED)
    model = Model().to(config.DEVICE)
    optimizer = get_optimizer(config, model)
    encoder_model_params = get_number_of_trainable_parameters(model.encoder_model)
    card_model_params = get_number_of_trainable_parameters(model.card_model)
    curve_params = get_number_of_trainable_parameters(model.card_model.forgetting_curve_nn)
    print(f"Number of trainable parameters: {encoder_model_params + card_model_params}, Encoder: {encoder_model_params}, Card: {card_model_params}, Forgetting curve: {curve_params}")

    if config.TRAIN_MODE == "WSD":
        start_factor = max(1e-4, config.WARMUP_START_LR / config.PEAK_LR)
        start_lr = start_factor * config.PEAK_LR
        warmup_steps = config.WARMUP_STEPS
        decay_steps = int(0.1 * config.TOTAL_STEPS)
        constant_steps = config.TOTAL_STEPS - warmup_steps - decay_steps
        print("Warmup steps:", warmup_steps)
        print("Constant steps:", constant_steps)
        print("Decay steps:", decay_steps)
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
            for group in optimizer_state["param_groups"]:
                group["lr"] = start_lr
        optimizer.load_state_dict(optimizer_state)
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

    if config.USE_WANDB:
        wandb_config = {
            "peak_lr": config.PEAK_LR,
            "adamw_betas": ADAMW_BETAS,
            "adamw_eps": ADAMW_EPS,
            "weight_decay": WEIGHT_DECAY,
            "clip": CLIP,
            "user_start": config.TRAIN_USER_START,
            "user_end": config.TRAIN_USER_END,
            "parameters": get_number_of_trainable_parameters(model),
            "seed": config.SEED,
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
    with compressed_db_env.begin(write=False) as summary_txn:
        with label_filter_env.begin(write=False) as label_filter_txn:
            for epoch in range(int(1e9)):
                random.shuffle(train_users)
                for user_i, user in enumerate(train_users):
                    log = {}
                    log["epoch"] = epoch
                    step += 1
                    if step < config.START_STEP:
                        if config.START_SCHEDULER_AT_START_STEP:
                            scheduler.step()
                        continue
                    validate_iter = (step + 1) % config.VALIDATE_STEPS == 0
                    current_lr = optimizer.param_groups[0]["lr"]
                    log["lr"] = current_lr

                    model.train()
                    batches = decoder_ops.get_data(summary_txn, user, config.DEVICE)
                    T = decoder_ops.extract_num_reviews(batches)
                    if T > config.SKIP_LENGTH:
                        print()
                        print(f"Skipping: {user}, size: {T}")
                    else:
                        try:
                            train_l, test_r = get_split(T)
                            test_l = generate_subsplits(train_l, test_r, T)
                            train_r = test_l - 1
                            encoding, sum_encoding = decoder_ops.encode_single(batches, model.encoder_model, train_l, train_r)
                            loss_sum_encoding_reg = torch.linalg.vector_norm(sum_encoding)

                            review_stats: decoder_ops.DecodeResult = decoder_ops.decode(batches, model.card_model, encoding, min_review_th=test_l, max_review_th=test_r)
                            first_review_logits = decoder_ops.extract_first_review_dist_logits(model.first_review_model, encoding)
                            first_stats: decoder_ops.DecodeResult = decoder_ops.first_logits(batches, first_review_logits, min_review_th=test_l, max_review_th=test_r)
                            review_loss_ce_avg = review_stats.ce_loss_sum / (1e-7 + review_stats.loss_n)
                            review_loss_bce_avg = review_stats.bce_loss_sum / (1e-7 + review_stats.loss_n)
                            review_n = review_stats.loss_n
                            first_loss_ce_avg = first_stats.ce_loss_sum / (1e-7 + first_stats.loss_n)
                            first_loss_bce_avg = first_stats.bce_loss_sum / (1e-7 + first_stats.loss_n)
                            first_n = first_stats.loss_n

                            ce_avg = (review_stats.ce_loss_sum + first_stats.ce_loss_sum) / (1e-7 + review_n + first_n)
                            bce_avg = (review_stats.bce_loss_sum + first_stats.bce_loss_sum) / (1e-7 + review_n + first_n)
                            tot_loss = bce_avg + 1e-1 * ce_avg + 1e-3 * loss_sum_encoding_reg

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
                                log["sum_encoding_loss"] = loss_sum_encoding_reg
                                log["encoding/sum_encoding_std"] = sum_encoding.std()
                                log["encoding/encoding_std"] = encoding.std()
                                tot_loss.backward()
                                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                                log["grad_norm"] = grad_norm

                                if step % 10 == 0:
                                    print()
                                    print(f"Indices:", train_l, test_l, test_r, train_r - train_l + 1, T, "User:", user)
                                    print(f"Review -- loss: ce: {review_loss_ce_avg:.3f}, bce: {review_loss_bce_avg:.3f}, n: {review_n}")
                                    print(f"First -- loss: ce: {first_loss_ce_avg:.3f}, bce: {first_loss_bce_avg:.3f}, n: {first_n}")
                                    print(f"sum encoding: {list(map(lambda x: round(float(x), 4), sum_encoding.tolist()))}")
                                    print(f"encoding: {list(map(lambda x: round(float(x), 4), encoding.tolist()))}")
                                    print(f"First rating dist: {[round(x, 2) for x in torch.nn.functional.softmax(first_review_logits, dim=-1).cpu().tolist()]}")
                                    print(f"{step} {epoch}, user: {user}, loss_ce: {review_loss_ce_avg.item():.3f}, loss_bce: {review_loss_bce_avg.item():.3f}, n: {review_n}, grad norm: {grad_norm:.3f}, lr: {current_lr:.3e}")
                                optimizer.step()
                                optimizer.zero_grad()

                                reserved = torch.cuda.memory_reserved()
                                if reserved >= config.THRESHOLD_RESERVED_GB * 1024 ** 3:
                                    print(f"Reserved: {reserved / (1024 ** 3):.3f} GB. Emptying cache.")
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

                    if validate_iter:
                        save_model_path = (
                            f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_{step}.pth"
                        )
                        save_optim_path = f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_{step}_optim.pth"
                        Path(config.SAVE_MODEL_FOLDER).mkdir(parents=True, exist_ok=True)               
                        torch.save(model.state_dict(), save_model_path)
                        torch.save(optimizer.state_dict(), save_optim_path)
                        print("MODEL SAVED.")

                        validate_result: ValidateResult = validate(model, summary_txn, label_filter_txn, validate_users, config, log)
                        log["validation/validation_loss"] = validate_result.loss_weighted_review
                        log["validation/validation_loss_user"] = validate_result.loss_weighted_user
                        log["validation/first_bce"] = validate_result.first_bce
                        log["validation/first_ce"] = validate_result.first_ce
                        validate_overfit_result: ValidateResult = validate(model, summary_txn, label_filter_txn, validate_overfit_users, config, log)
                        log["validation_overfit/validation_overfit_loss"] = validate_overfit_result.loss_weighted_review
                        log["validation_overfit/validation_overfit_loss_user"] = validate_overfit_result.loss_weighted_user
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