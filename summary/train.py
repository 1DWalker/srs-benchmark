from pathlib import Path
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

WEIGHT_DECAY = 1e-2
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
    return torch.optim.AdamW(model.parameters(), lr=config.PEAK_LR, weight_decay=WEIGHT_DECAY, betas=ADAMW_BETAS)

def validate(model, summary_txn, label_filter_txn, validate_users, device):
    torch.cuda.empty_cache()
    try:
        tot_loss = 0
        tot_loss_n = 0
        for user in validate_users:
            model.eval()
            batches = decoder_ops.get_data(summary_txn, user, config.DEVICE)
            T = decoder_ops.extract_num_reviews(batches)
            splits = load_tensor(label_filter_txn, f"{user}_split", device=device).tolist()
            equalize_review_ths = load_tensor(label_filter_txn, f"{user}_review_ths", device).tolist()
            rmse_bins = load_tensor(label_filter_txn, f"{user}_rmse_bins", device).tolist()
            assert len(splits) == 6
            with torch.inference_mode():
                encoding_list = []
                for split_i in range(len(splits) - 1):
                    test_min_review_th = splits[split_i]
                    encoding, sum_encoding = decoder_ops.encode(batches, model.encoder_model, min_review_th=0, max_review_th=test_min_review_th - 1)
                    encoding_list.append(encoding)
                loss, loss_n, rmse_raw, rmse_bins, auc = decoder_ops.decode_full(batches, model.card_model, encoding_list, splits, equalize_review_ths, rmse_bins, device=device, equalize_test_reviews=True)
                print()
                print(f"User: {user}, RMSE: {rmse_raw:.6f}, LogLoss: {loss:.6f}, RMSE (bins): {rmse_bins:.6f}, AUC: {(-1 if auc is None else auc):.6f}, size: {loss_n}")
                for split_i, parameters in enumerate(encoding_list):
                    print(f"Split: {split_i}, params: {list(map(lambda x: round(float(x), 4), parameters.tolist()))}")

                tot_loss += loss * loss_n
                tot_loss_n += loss_n
    except Exception as e:
        print(e)
        raise e
    print(f"Mean validation loss: {tot_loss / tot_loss_n:.4f}")
    return tot_loss / tot_loss_n

def get_split(n):
    assert n >= 12
    k = min([random.randint(max(100, int(0.2 * n)), n) for _ in range(2)])
    # r = l + k - 1 <= n-1 implies l <= n - k
    l = random.randint(0, n - k)
    return l, l + k - 1

def generate_subsplits(l, r, T):
    g = (r - l + 1) // 6
    linspace = np.linspace((l + r) // 2, r - g)
    floored_linspace = np.floor(linspace).astype(int)
    return np.random.choice(floored_linspace)

def main(config):
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    # fsrs_model = FSRS6().to(config.DEVICE)
    model = Model().to(config.DEVICE)
    optimizer = get_optimizer(config, model)
    encoder_model_params = get_number_of_trainable_parameters(model.encoder_model)
    card_model_params = get_number_of_trainable_parameters(model.card_model)
    curve_params = get_number_of_trainable_parameters(model.card_model.forgetting_curve_nn)
    print(f"Number of trainable parameters: {encoder_model_params + card_model_params}, Encoder: {encoder_model_params}, Card: {card_model_params}, Forgetting curve: {curve_params}")

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
        model.load_state_dict(torch.load(model_path, weights_only=True))
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
        if config.WANDB_RESUME:
            wandb.init(
                project=config.WANDB_PROJECT_NAME,
                id=config.WANDB_RESUME_ID,
                resume="must",
                config=wandb_config,
            )
        else:
            wandb.init(project=config.WANDB_PROJECT_NAME, config=wandb_config)

    with compressed_db_env.begin(write=False) as summary_txn:
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

                    model.train()
                    batches = decoder_ops.get_data(summary_txn, user, config.DEVICE)
                    T = decoder_ops.extract_num_reviews(batches)
                    try:
                        train_l, test_r = get_split(T)
                        test_l = generate_subsplits(train_l, test_r, T)
                        train_r = test_l - 1
                        print()
                        print(f"Indices:", train_l, test_l, test_r, train_r - train_l + 1, T, "User:", user)
                        # summarizer_out_TP = summary_model(*training_sample)
                        encoding, sum_encoding = decoder_ops.encode(batches, model.encoder_model, train_l, train_r)
                        print("encoding", encoding)

                        loss_avg, loss_tot, loss_n = decoder_ops.decode(batches, model.card_model, encoding, min_review_th=test_l, max_review_th=test_r)
                        loss = loss_tot.sum() / (1e-7 + loss_n)

                        print(f"loss: {loss_avg} ({loss_n}), sum encoding: {list(map(lambda x: round(float(x), 4), sum_encoding.tolist()))}")

                        log["unstable_gradient"] = 0
                        if loss_n < 100:
                            print("Skipping: label sizes are too small.", loss_n)
                        elif loss.requires_grad:
                            log["train_loss"] = loss
                            log["train_loss_fsrs"] = loss
                            loss.backward()
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                            log["grad_norm"] = grad_norm
                            if grad_norm > 100:
                                log["unstable_gradient"] = 1
                            print(f"{step} {epoch}, user: {user}, loss: {loss.item():.3f}, loss_fsrs: {loss.item():.3f} ({loss_n}), grad norm: {grad_norm:.3f}, lr: {current_lr:.3e}")
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
                        print(e)
                        log["train_nan"] = 1
                        raise e
                
                    scheduler.step()
                    if step % 50 == 0:
                        log_model(log, model)

                    if validate_iter:
                        save_model_path = (
                            f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_{step}.pth"
                        )
                        save_optim_path = f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_{step}_optim.pth"
                        Path(config.SAVE_MODEL_FOLDER).mkdir(parents=True, exist_ok=True)               
                        torch.save(model.state_dict(), save_model_path)
                        torch.save(optimizer.state_dict(), save_optim_path)
                        print("MODEL SAVED.")

                        validation_overfit_out = validate(model, summary_txn, label_filter_txn, validate_overfit_users, config.DEVICE)
                        if validation_overfit_out is not None:
                            validation_overfit_fsrs = validation_overfit_out
                            log["validation_overfit_loss"] = validation_overfit_fsrs
                        validation_out = validate(model, summary_txn, label_filter_txn, validate_users, config.DEVICE)
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