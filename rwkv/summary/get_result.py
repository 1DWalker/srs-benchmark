from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import queue
import lmdb
import torch
from tqdm import tqdm
from fsrs.evaluate_fsrs import evaluate_full
from fsrs.fsrs import FSRS6
from rwkv.summary.summary_model import FSRSSummaryModel
from rwkv.summary.train_fsrs_optimizer import evaluate_aux, get_data
from script import sort_jsonl
from utils import load_tensor, parse_toml
import multiprocessing as mp

def to_json(user_id, loss, loss_n, rmse_raw, rmse_bins, auc, params):
    try:
        auc = round(auc, 6)
    except:
        auc = None
    stats = {
        "metrics": {
            "RMSE": round(rmse_raw, 6),
            "LogLoss": round(loss, 6),
            "RMSE(bins)": round(rmse_bins, 6),
            "AUC": auc,
        },
        "user": int(user_id),
        "size": loss_n,
        "parameters": {
            "0": list(map(lambda x: round(x, 4), params.tolist()))
        }
    }
    return stats

@torch.inference_mode()
def worker_job(config, job_queue, writer_queue, progress_queue):
    device = torch.device("cpu")
    fsrs_model = FSRS6().to(device)
    summarizer_model = FSRSSummaryModel().to(device)
    summarizer_model.load_state_dict(torch.load(config.MODEL_PATH, weights_only=True))

    summary_env = lmdb.open(config.SUMMARY_DB_PATH, readonly=True, lock=False)
    fsrs_evaluate_env = lmdb.open(config.FSRS_EVALUATE_DB_PATH, readonly=True, lock=False)
    label_filter_env = lmdb.open(config.LABEL_FILTER_DB_PATH, readonly=True, lock=False)

    with summary_env.begin(write=False) as summary_txn:
        with fsrs_evaluate_env.begin(write=False) as fsrs_txn:
            with label_filter_env.begin(write=False) as label_filter_txn:
                while True:
                    try:
                        user = job_queue.get_nowait()
                        if user is None:
                            return

                        summarizer_in = get_data(summary_txn, user, device, torch.float)
                        T = summarizer_in[0].size(0)
                        timeshift_select_T = torch.cat((torch.zeros(1, dtype=torch.long, device=device), torch.arange(start=0, end=T - 1, dtype=torch.long, device=device)))
                        skip_T = torch.full((T,), fill_value=0, dtype=torch.bool, device=device)
                        splits = load_tensor(label_filter_txn, f"{user}_split", device=device).tolist()

                        summarizer_out_TP, aux = summarizer_model(*summarizer_in, timeshift_select_T, skip_T)
                        parameter_list = []
                        for split_i in range(len(splits) - 1):
                            test_min_review_th = splits[split_i]
                            fsrs_param_index = test_min_review_th - 2
                            assert fsrs_param_index >= 0
                            fsrs_params_P = summarizer_out_TP[fsrs_param_index]
                            parameter_list.append(fsrs_params_P)
                        loss, loss_n, rmse_raw, rmse_bins, auc = evaluate_full(fsrs_txn, fsrs_model, parameter_list, splits, user, device=device, equalize_test_reviews=True)
                        aux_loss, aux_loss_n, aux_rmse_raw, aux_rmse_bins, aux_auc = evaluate_aux(label_filter_txn, summary_txn, user, aux, device, equalize_test_reviews=True, include_other_metrics=True)

                        print()
                        print(f"FSRS - User: {user}, RMSE: {rmse_raw:.6f}, LogLoss: {loss:.6f}, RMSE (bins): {rmse_bins:.6f}, AUC: {auc:.6f}, size: {loss_n}")
                        print(f"AUX  - User: {user}, RMSE: {aux_rmse_raw:.6f}, LogLoss: {aux_loss:.6f}, RMSE (bins): {aux_rmse_bins:.6f}, AUC: {aux_auc:.6f}, size: {aux_loss_n}")
                        for split_i, parameters in enumerate(parameter_list):
                            print(f"Split: {split_i}, params: {list(map(lambda x: round(float(x), 4), parameters.tolist()))}")

                        progress_queue.put(1)
                        writer_queue.put((to_json(user, loss.item(), loss_n, rmse_raw, rmse_bins, auc, fsrs_params_P), to_json(user, aux_loss.item(), aux_loss_n, aux_rmse_raw, aux_rmse_bins, aux_auc, fsrs_params_P)))
                    except queue.Empty:
                        return

def writer_job(result_file, result_aux_file, writer_queue):
    while True:
        message = writer_queue.get()
        if message is None:
            return
        stats, stats_aux = message
        with open(result_file, "a") as f:
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")
        with open(result_aux_file, "a") as f:
            f.write(json.dumps(stats_aux, ensure_ascii=False) + "\n")
    
def progress_tracker(total_items, progress_queue):
    with tqdm(total=total_items, desc="Generating Data") as pbar:
        for _ in range(total_items):
            progress_queue.get()
            pbar.update(1)

def main(config):
    mp.set_start_method("spawn", force=True)
    unprocessed_users = []
    Path("result").mkdir(parents=True, exist_ok=True)
    result_file = Path(f"result/{config.OUT_NAME}.jsonl")
    result_aux_file = Path(f"result/{config.OUT_NAME}_aux.jsonl")
    if result_file.exists():
        data = sort_jsonl(result_file)
        processed_user = set(map(lambda x: x["user"], data))
    else:
        processed_user = set()
    if result_aux_file.exists():
        sort_jsonl(result_aux_file)

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
            target=writer_job, args=(result_file, result_aux_file, writer_queue)
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
    sort_jsonl(result_aux_file)


if __name__ == '__main__':
    config = parse_toml()
    main(config)