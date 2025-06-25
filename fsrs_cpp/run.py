import json
from pathlib import Path
import queue
import time
import lmdb
import numpy as np
from sklearn.metrics import log_loss
import torch
from tqdm import tqdm
from fsrs_cpp.fsrs6_reference import FSRS6
from fsrs_cpp.fsrs_ops import FSRSCppFunction, RevlogTensors
from script import sort_jsonl
from utils import load_tensor, parse_toml
import multiprocessing as mp

def fsrs_fun(params, review_th_batch, revlog_tensors):
    return FSRSCppFunction.apply(params, review_th_batch, revlog_tensors)

def train_eval(params, train_set_review_ths, revlog_tensors: RevlogTensors):
    with torch.no_grad():
        pred_y = fsrs_fun(params, train_set_review_ths, revlog_tensors)
        y = (revlog_tensors.packed_rating_T[revlog_tensors.perm_inv_T_tensor[train_set_review_ths - 1]] > 1).float()
        return log_loss(y_true=y, y_pred=pred_y, labels=[0, 1])

def process(config, user_id, env):
    print("Process:", user_id)
    fsrs_reference = FSRS6()

    with env.begin(write=False) as txn:
        device = torch.device("cpu")
        start = time.time()
        packed_review_th_T = load_tensor(txn, f"{user_id}_packed_review_th_T", device)
        packed_rating_T = load_tensor(txn, f"{user_id}_packed_rating_T", device)
        packed_elapsed_days_real_T = load_tensor(txn, f"{user_id}_packed_elapsed_days_real_T", device)
        packed_elapsed_days_int_T = load_tensor(txn, f"{user_id}_packed_elapsed_days_int_T", device)
        packed_label_elapsed_days_real_T = load_tensor(txn, f"{user_id}_packed_label_elapsed_days_real_T", device)
        packed_label_elapsed_days_int_T = load_tensor(txn, f"{user_id}_packed_label_elapsed_days_int_T", device)
        perm_T_tensor = load_tensor(txn, f"{user_id}_perm_T_tensor", device)
        perm_inv_T_tensor = load_tensor(txn, f"{user_id}_perm_inv_T_tensor", device)
        card_locs_T = load_tensor(txn, f"{user_id}_card_locs_T", device)
        y_T = (packed_rating_T[perm_inv_T_tensor] > 1).float()
        revlog_tensors = RevlogTensors(
            packed_review_th_T=packed_review_th_T,
            packed_rating_T=packed_rating_T,
            packed_elapsed_days_real_T=packed_elapsed_days_real_T,
            packed_elapsed_days_int_T=packed_elapsed_days_int_T,
            packed_label_elapsed_days_real_T=packed_label_elapsed_days_real_T,
            packed_label_elapsed_days_int_T=packed_label_elapsed_days_int_T,
            perm_T_tensor=perm_T_tensor,
            perm_inv_T_tensor=perm_inv_T_tensor,
            card_locs_T=card_locs_T,
        )

        test_pred_y_all = []
        test_y_all = []
        for split_i in range(5):
            epochs = load_tensor(txn, f"{user_id}_split_{split_i}_epochs", device)
            epochs_np = epochs.numpy()
            review_ths = load_tensor(txn, f"{user_id}_split_{split_i}_review_ths", device)
            batch_lens = load_tensor(txn, f"{user_id}_split_{split_i}_batch_lens", device)
            review_th_batches = review_ths.split(batch_lens.tolist())
            lrs = load_tensor(txn, f"{user_id}_split_{split_i}_lrs", device)
            train_set_review_ths = load_tensor(txn, f"{user_id}_split_{split_i}_train_set_review_ths", device)
            test_set_review_ths = load_tensor(txn, f"{user_id}_split_{split_i}_test_set_review_ths", device)

            pretrain_params = load_tensor(txn, f"{user_id}_split_{split_i}_pretrain_params", device)
            assert pretrain_params.size(0) == 4
            params = fsrs_reference.initial_params.detach().clone().requires_grad_(True)
            with torch.no_grad():
                for i in range(4):
                    params[i] = pretrain_params[i]

            optim = torch.optim.Adam([params], lr=4e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=lrs.size(0)
            )

            # print(params)
            best_loss = np.inf
            for batch_i, review_th_batch in enumerate(review_th_batches):
                is_new_epoch = batch_i == 0 or epochs_np[batch_i - 1] != epochs_np[batch_i]
                if is_new_epoch:
                    eval_loss = train_eval(params.detach().clone(), train_set_review_ths, revlog_tensors)
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        best_params = params.detach().clone()

                pred_y_batch = fsrs_fun(params, review_th_batch, revlog_tensors)
                y_batch = y_T[review_th_batch - 1]

                loss_fn = torch.nn.BCELoss(reduction='none')
                loss = loss_fn(pred_y_batch, y_batch)
                loss_sum = loss.sum()
                loss_reg = fsrs_reference.get_regularization_loss(params, review_th_batch.size(0)) / train_set_review_ths.size(0)
                loss_all = loss_sum + loss_reg
                loss_all.backward()
                # print("loss:", loss_sum, loss_reg)
                optim.step()
                optim.zero_grad()
                params = fsrs_reference.clip(params)
                scheduler.step()

            eval_loss = train_eval(params.detach().clone(), train_set_review_ths, revlog_tensors)
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_params = params.detach().clone()

            test_pred_y = fsrs_fun(best_params, test_set_review_ths, revlog_tensors)
            test_y = y_T[test_set_review_ths - 1]
            test_pred_y_all.extend(test_pred_y.detach().numpy())
            test_y_all.extend(test_y.numpy())

        logloss = log_loss(y_true=test_y_all, y_pred=test_pred_y_all, labels=[0, 1])
        return {
            "metrics": {
                "LogLoss": round(logloss, 6)
            },
            "user": int(user_id),
            "size": len(test_y_all),
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
    main(config)