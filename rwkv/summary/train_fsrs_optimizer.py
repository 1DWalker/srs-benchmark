from pathlib import Path
import lmdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, root_mean_squared_error
import torch
import wandb
from fsrs.fsrs import FSRS6
from rwkv.summary.summary_model import FSRSSummaryModel
from rwkv.utils import get_number_of_trainable_parameters, transfer_child_grad_to_master
import random

from utils import compact_lmdb, load_tensor, parse_toml
from fsrs.evaluate_fsrs import evaluate_batched_parameters, evaluate_full

WEIGHT_DECAY = 1e-2
ADAMW_EPS = 1e-18
ADAMW_BETAS = (0.90, 0.999)
CLIP = 0.5

def log_model(log, model):
    for name, param in model.named_parameters():
        log[f"{name}.data.mean"] = param.mean().item()
        if param.numel() > 1:
            log[f"{name}.data.std"] = param.std().item()
        log[f"{name}.data.min"] = param.min().item()
        log[f"{name}.data.max"] = param.max().item()
        log[f"{name}.data.25th"] = torch.quantile(param, 0.25).item()
        log[f"{name}.data.50th"] = torch.quantile(param, 0.50).item()
        log[f"{name}.data.75th"] = torch.quantile(param, 0.75).item()
        if param.grad is not None:
            log[f"{name}.grad.mean"] = param.grad.mean().item()
            if param.numel() > 1:
                log[f"{name}.grad.std"] = param.grad.std().item()
            log[f"{name}.grad.min"] = param.grad.min().item()
            log[f"{name}.grad.max"] = param.grad.max().item()
            log[f"{name}.grad.25th"] = torch.quantile(param.grad, 0.25).item()
            log[f"{name}.grad.50th"] = torch.quantile(param.grad, 0.50).item()
            log[f"{name}.grad.75th"] = torch.quantile(param.grad, 0.75).item()

def get_optimizer(config, model):
    encode_params = []
    decay_params = []
    channel_mixer_params = []
    decay_head_params = []
    other_params = []
    head_targets = [
        "fsrs_linear",
    ]
    for name, param in model.named_parameters():
        # Param constraint is to exclude layer/group norm weights
        if (
            "weight" in name
            and "lora" not in name
            and "scale" not in name
            and len(param.squeeze().shape) >= 2
        ):
            is_head_param = False
            for head_target in head_targets:
                if head_target in name:
                    is_head_param = True
            if is_head_param:
                decay_head_params.append(param)
            elif "features2card" in name:
                encode_params.append(param)
            elif "channel_mixer" in name:
                channel_mixer_params.append(param)
            else:
                decay_params.append(param)
        else:
            other_params.append(param)

    return torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": WEIGHT_DECAY,
                "lr": config.PEAK_LR,
            },
            {
                "params": channel_mixer_params,
                "weight_decay": WEIGHT_DECAY,
                "lr": config.PEAK_LR,
            },
            {
                "params": decay_head_params,
                "weight_decay": 0.001,
                "lr": config.PEAK_LR,
            },
            {"params": encode_params, "weight_decay": 1e-2, "lr": config.PEAK_LR},
            {"params": other_params, "weight_decay": 0.0, "lr": config.PEAK_LR},
        ],
        eps=ADAMW_EPS,
        betas=ADAMW_BETAS,
    )

def get_data(txn, user_id, device, dtype=None):
    scaled_seconds_T1 = load_tensor(txn, f"{user_id}_scaled_seconds_T1", device)
    rating_onehot_T4 = load_tensor(txn, f"{user_id}_rating_onehot_T4", device).to(scaled_seconds_T1.dtype)
    scaled_elapsed_days_T1 = load_tensor(txn, f"{user_id}_scaled_elapsed_days_T1", device)
    scaled_state_T1 = load_tensor(txn, f"{user_id}_scaled_state_T1", device)
    scaled_duration_T1 = load_tensor(txn, f"{user_id}_scaled_duration_T1", device)
    scaled_day_offset_diff_T1 = load_tensor(txn, f"{user_id}_scaled_day_offset_diff_T1", device)
    perm_T_tensor = load_tensor(txn, f"{user_id}_perm_T_tensor", device).long()
    perm_inv_T_tensor = load_tensor(txn, f"{user_id}_perm_inv_T_tensor", device).long()
    indices_I = load_tensor(txn, f"{user_id}_indices_I", device).long()
    features_TC = torch.cat((rating_onehot_T4, scaled_seconds_T1, scaled_elapsed_days_T1, scaled_duration_T1, scaled_state_T1, scaled_day_offset_diff_T1), dim=-1)
    label_elapsed_seconds_T = load_tensor(txn, f"{user_id}_label_elapsed_seconds_T", device)
    if dtype is not None:
        features_TC = features_TC.to(dtype)
        label_elapsed_seconds_T = label_elapsed_seconds_T.to(dtype)
    return features_TC, indices_I, perm_T_tensor, perm_inv_T_tensor, label_elapsed_seconds_T

def decorate_training_sample(T, device):
    l, r = get_split(T)
    global_dropout = random.uniform(0.0, 0.5)
    skip_T_np = np.random.rand(T) < global_dropout  # Skip randomly
    skip_T_np[:l] = 1  # Skip all < l
    timeshift_select_list = []
    last = 0
    found = False
    for i, skip in enumerate(skip_T_np):
        if not found:
            timeshift_select_list.append(i)
        else:
            timeshift_select_list.append(last)
        if not skip:
            last = i
            found = True

    skip_T = torch.tensor(skip_T_np, dtype=torch.bool, device=device)
    timeshift_select_T = torch.tensor(timeshift_select_list, dtype=torch.long, device=device)

    assert timeshift_select_T.size(0) == T
    assert skip_T.size(0) == T
    return l, r, timeshift_select_T, skip_T

def validate(summarizer_model, fsrs_model, summary_txn, fsrs_txn, label_filter_txn, validate_users, device):
    torch.cuda.empty_cache()
    try:
        tot_loss = 0
        tot_loss_n = 0
        for user in validate_users:
            summarizer_model.eval()
            summarizer_in = get_data(summary_txn, user, device)
            T = summarizer_in[0].size(0)
            timeshift_select_T = torch.cat((torch.zeros(1, dtype=torch.long, device=device), torch.arange(start=0, end=T - 1, dtype=torch.long, device=device)))
            skip_T = torch.full((T,), fill_value=0, dtype=torch.bool, device=device)
            splits = load_tensor(label_filter_txn, f"{user}_split", device=device).tolist()
            assert len(splits) == 6
            with torch.no_grad():
                summarizer_out_TP = summarizer_model(*summarizer_in, timeshift_select_T, skip_T)
                parameter_list = []
                for split_i in range(len(splits) - 1):
                    test_min_review_th = splits[split_i]
                    fsrs_param_index = test_min_review_th - 2
                    assert fsrs_param_index >= 0
                    fsrs_params_P = summarizer_out_TP[fsrs_param_index]
                    parameter_list.append(fsrs_params_P)
                loss, loss_n, rmse_raw, rmse_bins, auc = evaluate_full(fsrs_txn, fsrs_model, parameter_list, splits, user, device=device, equalize_test_reviews=True)
                print()
                print(f"FSRS - User: {user}, RMSE: {rmse_raw:.6f}, LogLoss: {loss:.6f}, RMSE (bins): {rmse_bins:.6f}, AUC: {auc:.6f}, size: {loss_n}")
                for split_i, parameters in enumerate(parameter_list):
                    print(f"Split: {split_i}, params: {list(map(lambda x: round(float(x), 4), parameters.tolist()))}")

                tot_loss += loss * loss_n
                tot_loss_n += loss_n
    except Exception as e:
        print("Exception in validate. RWKV-7 nan?")
        print(e)
        return None
    print(f"Mean validation loss: {tot_loss / tot_loss_n:.4f}")
    return tot_loss / tot_loss_n

def get_split(n):
    assert n >= 12
    k = min([random.randint(max(100, int(0.2 * n)), n) for _ in range(2)])
    # r = l + k - 1 <= n-1 implies l <= n - k
    l = random.randint(0, n - k)
    return l, l + k - 1

def generate_subsplits(l, r, T):
    M = 200000
    p = 0.30  # Estimate of the proportion of memory that is used for RWKV evaluation at T = M and 32 subsplits
    v = int(M * 32 * 1 / p)

    subsplit_values = min(r - l + 1, max(32, int((v - (1 - p) * v / M * T) / T)))
    g = (r - l + 1) // 6
    linspace = np.linspace((l + r) // 2, r - g, num=subsplit_values)
    floored_linspace = np.floor(linspace).astype(int)
    return np.unique(floored_linspace).tolist()

def main(config):
    random.seed(config.SEED)
    fsrs_model = FSRS6().to(config.DEVICE)

    master_model = FSRSSummaryModel().to(config.DEVICE)
    model = FSRSSummaryModel().selective_cast(config.DTYPE).to(config.DEVICE)
    optimizer = get_optimizer(config, master_model)
    print("Number of trainable parameters:", get_number_of_trainable_parameters(model))

    if config.TRAIN_MODE == "WS":
        start_factor = max(1e-4, config.WARMUP_START_LR / config.PEAK_LR)
        start_lr = start_factor * config.PEAK_LR
        warmup_steps = config.WARMUP_STEPS
        print("Warmup steps:", warmup_steps)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps
        )
        main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    elif config.TRAIN_MODE == "D":

        def cosine_down(step, total_steps):
            return 1 + np.cos(0.5 * np.pi * (1 + step / total_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda t: cosine_down(t, config.TOTAL_STEPS)
        )
    else:
        raise ValueError(f"Invalid train mode: {config.TRAIN_MODE}")

    if config.LOAD_MODEL:
        model_path = f"{config.LOAD_MODEL_FOLDER}/{config.LOAD_MODEL_NAME}.pth"
        optim_path = f"{config.LOAD_MODEL_FOLDER}/{config.LOAD_MODEL_NAME}_optim.pth"
        print("Loading model:", model_path)
        master_model.load_state_dict(torch.load(model_path, weights_only=True))
        optimizer_state = torch.load(optim_path, weights_only=True)
        if config.TRAIN_MODE == "WS":
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
    if 4371 in train_users:
        train_users.remove(4371)
        print("Removed user 4371 from train_users.")
    if 4371 in validate_users:
        validate_users.remove(4371)
        print("Removed user 4371 from validate_users.")

    summary_env = lmdb.open(config.SUMMARY_DB_PATH, readonly=True, lock=False)
    fsrs_evaluate_env = lmdb.open(config.FSRS_EVALUATE_DB_PATH, readonly=True, lock=False)
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
        if config.WANDB_RESUME:
            wandb.init(
                project=config.WANDB_PROJECT_NAME,
                id=config.WANDB_RESUME_ID,
                resume="must",
                config=wandb_config,
            )
        else:
            wandb.init(project=config.WANDB_PROJECT_NAME, config=wandb_config)

    with summary_env.begin(write=False) as summary_txn:
        with fsrs_evaluate_env.begin(write=False) as fsrs_txn:
            with label_filter_env.begin(write=False) as label_filter_txn:
                step = config.START_STEP - 1
                for epoch in range(int(1e9)):
                    random.shuffle(train_users)
                    for user_i, user in enumerate(train_users):
                        log = {}
                        log["epoch"] = epoch
                        step += 1
                        validate_iter = (step + 1) % config.VALIDATE_STEPS == 0
                        # validate_iter = True
                        log["step"] = step
                        current_lr = optimizer.param_groups[0]["lr"]
                        log["lr"] = current_lr

                        model.copy_downcast_(master_model, dtype=config.DTYPE)
                        model.train()
                        training_sample = get_data(summary_txn, user, config.DEVICE)
                        T = training_sample[0].size(0)
                        if T > config.SKIP_LENGTH:
                            print(f"Skipping: {user}")
                        else:
                            try:
                                train_l, train_r, timeshift_select_T, skip_T = decorate_training_sample(training_sample[0].size(0), device=config.DEVICE)
                                subsplits = generate_subsplits(train_l, train_r, T)
                                print()
                                print(f"Indices:", train_l, train_r, train_r - train_l + 1, T, "User:", user)
                                print(f"Number of subsplits:", len(subsplits))
                                summarizer_out_TP = model(*training_sample, timeshift_select_T, skip_T)

                                params_list = []
                                min_review_ths_list = []
                                for subsplit in subsplits:
                                    fsrs_params_P = summarizer_out_TP[subsplit - 1]
                                    params_list.append(fsrs_params_P)
                                    min_review_ths_list.append(subsplit + 1)

                                params_hp = torch.stack(params_list, dim=0)
                                min_review_ths_h = torch.tensor(min_review_ths_list, device=config.DEVICE)
                                max_review_ths_h = torch.full_like(min_review_ths_h, fill_value=train_r + 1)
                                loss_avg_h, loss_tot_h, loss_n_h = evaluate_batched_parameters(fsrs_txn, fsrs_model, params_hp, user, min_review_th_h=min_review_ths_h, max_review_th_h=max_review_ths_h, device=config.DEVICE, equalize_test_reviews=True, skip_same_day_reviews=True)
                                loss_n = loss_n_h.sum()
                                loss_fsrs = loss_tot_h.sum() / (1e-7 + loss_n)

                                for h in np.floor(np.linspace(0, len(subsplits) - 1, num=3)).astype(int):
                                    subsplit = subsplits[h]
                                    print(f"Subsplit: {subsplit}, loss: {loss_avg_h[h].item():.3f} ({loss_n_h[h]}), params: {list(map(lambda x: round(float(x), 4), params_hp[h].tolist()))}")

                                for param_i, param in enumerate(fsrs_params_P.tolist()):
                                    log[f"fsrs_param_{param_i}"] = param

                                loss = loss_fsrs
                                log["unstable_gradient"] = 0
                                if loss_n < 100:
                                    print("Skipping: label sizes are too small.", loss_n)
                                elif loss.requires_grad:
                                    log["train_loss"] = loss
                                    log["train_loss_fsrs"] = loss_fsrs
                                    loss.backward()
                                    transfer_child_grad_to_master(master=master_model, child=model)
                                    grad_norm = torch.nn.utils.clip_grad_norm_(master_model.parameters(), CLIP)
                                    log["grad_norm"] = grad_norm
                                    if grad_norm > 100:
                                        log["unstable_gradient"] = 1
                                    print(f"{step} {epoch}, user: {user}, loss: {loss.item():.3f}, loss_fsrs: {loss_fsrs.item():.3f} ({loss_n}), grad norm: {grad_norm:.3f}, lr: {current_lr:.3e}")
                                    optimizer.step()
                                    optimizer.zero_grad()

                                    reserved = torch.cuda.memory_reserved()
                                    if reserved >= config.THRESHOLD_RESERVED_GB * 1024 ** 3:
                                        print(f"Reserved: {reserved / (1024 ** 3):.3f} GB. Emptying cache.")
                                        torch.cuda.empty_cache()
                                else:
                                    print("No grad required.")
                                log["train_nan"] = 0
                            except Exception as e:
                                print("Exception caught. Nan from RWKV-7? Skipping batch.")
                                print(e)
                                log["train_nan"] = 1
                                # raise e
                        
                        scheduler.step()
                        if step % 50 == 0:
                            log_model(log, master_model)

                        if validate_iter:
                            save_model_path = (
                                f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_{step}.pth"
                            )
                            save_optim_path = f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_{step}_optim.pth"
                            Path(config.SAVE_MODEL_FOLDER).mkdir(parents=True, exist_ok=True)               
                            torch.save(master_model.state_dict(), save_model_path)
                            torch.save(optimizer.state_dict(), save_optim_path)
                            print("MODEL SAVED.")

                            model.copy_downcast_(master_model, dtype=config.DTYPE)
                            validation_overfit_out = validate(model, fsrs_model, summary_txn, fsrs_txn, label_filter_txn, validate_overfit_users, config.DEVICE)
                            if validation_overfit_out is not None:
                                validation_overfit_fsrs = validation_overfit_out
                                log["validation_overfit_loss"] = validation_overfit_fsrs
                            validation_out = validate(model, fsrs_model, summary_txn, fsrs_txn, label_filter_txn, validate_users, config.DEVICE)
                            if validation_out is not None:
                                validation_fsrs = validation_out
                                log["validation_loss"] = validation_fsrs
                            if validation_overfit_out is None or validation_out is None:
                                log["validation_nan"] = 1
                            else:
                                log["validation_nan"] = 0

                        if config.USE_WANDB:
                            wandb.log(log, step=step)
                        if step == config.TOTAL_STEPS - 1:
                            break
                    if step == config.TOTAL_STEPS - 1:
                        break


if __name__ == '__main__':
    # decorate_training_sample(20, torch.device("cpu"))
    # exit()
    config = parse_toml()
    main(config)