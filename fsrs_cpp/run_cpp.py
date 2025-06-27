import json
from pathlib import Path
import queue
import time
import lmdb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error
import torch
from tqdm import tqdm
from fsrs_cpp import _FSRS_CPP
from script import sort_jsonl
from utils import load_tensor, parse_toml
import multiprocessing as mp

def process(config, user_id, env):
    with env.begin(write=False) as txn:
        device = torch.device("cpu")
        packed_review_th_T = load_tensor(txn, f"{user_id}_packed_review_th_T", device)
        packed_rating_T = load_tensor(txn, f"{user_id}_packed_rating_T", device)
        packed_elapsed_days_real_T = load_tensor(txn, f"{user_id}_packed_elapsed_days_real_T", device)
        packed_elapsed_days_int_T = load_tensor(txn, f"{user_id}_packed_elapsed_days_int_T", device)
        packed_label_elapsed_days_real_T = load_tensor(txn, f"{user_id}_packed_label_elapsed_days_real_T", device)
        packed_label_elapsed_days_int_T = load_tensor(txn, f"{user_id}_packed_label_elapsed_days_int_T", device)
        all_test_set_rmse_bin_ind = load_tensor(txn, f"{user_id}_all_test_set_rmse_bin_ind", device)

        loss_tot = 0
        loss_n_tot = 0
        y_all = []
        y_pred_all = []
        for split_i in range(5):
            pretrain_params = load_tensor(txn, f"{user_id}_split_{split_i}_pretrain_params", device)
            epochs = load_tensor(txn, f"{user_id}_split_{split_i}_epochs", device)
            train_ords = load_tensor(txn, f"{user_id}_split_{split_i}_train_ords", device)   
            locs = load_tensor(txn, f"{user_id}_split_{split_i}_locs", device)   
            locs_lens = load_tensor(txn, f"{user_id}_split_{split_i}_locs_lens", device)   
            keys = load_tensor(txn, f"{user_id}_split_{split_i}_keys", device)   
            keys_lens = load_tensor(txn, f"{user_id}_split_{split_i}_keys_lens", device)   
            train_set_locs = load_tensor(txn, f"{user_id}_split_{split_i}_train_set_locs", device)   
            train_set_keys = load_tensor(txn, f"{user_id}_split_{split_i}_train_set_keys", device)   
            test_set_locs = load_tensor(txn, f"{user_id}_split_{split_i}_test_set_locs", device)   
            test_set_keys = load_tensor(txn, f"{user_id}_split_{split_i}_test_set_keys", device)   

            loss, loss_n, best_params, y, y_pred = torch.ops.fsrs.fsrs_optimizer(
                pretrain_params,
                epochs,
                train_ords,
                locs,
                locs_lens,
                keys,
                keys_lens,
                train_set_locs,
                train_set_keys,
                test_set_locs,
                test_set_keys,
                packed_review_th_T,
                packed_rating_T,
                packed_elapsed_days_real_T,
                packed_elapsed_days_int_T,
                packed_label_elapsed_days_real_T,
                packed_label_elapsed_days_int_T,
            )
            loss_tot += loss
            loss_n_tot += loss_n
            y_all.append(y)
            y_pred_all.append(y_pred)

        test_y = torch.cat(y_all)
        test_y_pred = torch.cat(y_pred_all)
        test_y_np = test_y.numpy()
        test_y_pred_np = test_y_pred.numpy()
        logloss = loss_tot.item() / loss_n_tot.item()
        rmse_raw = root_mean_squared_error(y_true=test_y_np, y_pred=test_y_pred_np)
        rmse_bins = torch.ops.fsrs.compute_rmse_bins(test_y, test_y_pred, all_test_set_rmse_bin_ind)
        try:
            auc = round(roc_auc_score(y_true=test_y_np, y_score=test_y_pred_np), 6)
        except Exception:
            auc = None

        return {
            "metrics": {
                "RMSE": round(rmse_raw, 6),
                "LogLoss": round(logloss, 6),
                "RMSE(bins)": round(rmse_bins.item(), 6),
                "AUC": auc,
            },
            "user": int(user_id),
            "size": loss_n_tot.item(),
            "parameters": {
                "0": list(map(lambda x: round(x, 4), best_params.tolist()))
            },
        }


def worker_job(config, job_queue, writer_queue, progress_queue):
    env = lmdb.open(config.DB_PATH, readonly=True, lock=False)
    while True:
        try:
            user = job_queue.get_nowait()
            if user is None:
                return

            writer_queue.put(process(config, user, env))
            progress_queue.put(1)
        except queue.Empty:
            return

def writer_job(result_file, writer_queue):
    while True:
        message = writer_queue.get()
        if message is None:
            return
        stats = message
        with open(result_file, "a") as f:
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")
    
def progress_tracker(total_items, progress_queue):
    with tqdm(total=total_items, desc="Progress") as pbar:
        for _ in range(total_items):
            progress_queue.get()
            pbar.update(1)

def main(config):
    mp.set_start_method("spawn", force=True)
    unprocessed_users = []
    Path("result").mkdir(parents=True, exist_ok=True)
    result_file = Path(f"result/{config.OUT_NAME}.jsonl")
    if result_file.exists():
        data = sort_jsonl(result_file)
        processed_user = set(map(lambda x: x["user"], data))
    else:
        processed_user = set()

    users = list(range(config.USER_START, config.USER_END + 1))
    if 4371 in users:
        users.remove(4371)
        print("Removed user 4371 from users.")
    for user_id in users:
        if user_id not in processed_user:
            unprocessed_users.append(user_id)

    unprocessed_users.sort()
    print("Unprocessed users length:", len(unprocessed_users))

    job_queue = mp.Queue()
    for user_id in users:
        job_queue.put(user_id)

    with mp.Manager() as manager:
        writer_queue = manager.Queue()
        writer = mp.Process(
            target=writer_job, args=(result_file, writer_queue)
        )
        writer.start()

        progress_queue = manager.Queue()
        progress_process = mp.Process(
            target=progress_tracker, args=(len(unprocessed_users), progress_queue)
        )
        progress_process.start()

        jobs = [mp.Process(
            target=worker_job, args=(config, job_queue, writer_queue, progress_queue)
        ) for _ in range(config.PROCESSES)]
        for job in jobs:
            job.start()

        for job in jobs:
            job.join()

        writer_queue.put(None)
        writer.join()
        progress_process.terminate()

    sort_jsonl(result_file)

if __name__ == '__main__':
    config = parse_toml()

    env = lmdb.open(config.DB_PATH, readonly=True, lock=False)
    # process(config, 2, env)

    main(config)