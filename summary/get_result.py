from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import queue
import lmdb
from summary import decoder_ops, fsrs_encoder_model
from summary.model import Model
import torch
from tqdm import tqdm
from script import sort_jsonl
from utils import load_tensor, parse_toml
import multiprocessing as mp

def to_json(user_id, loss, loss_n, cond_loss, cond_n, first_bce, first_ce, first_n, rmse_raw, rmse_bins, auc, initial_rating, params):
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
            "CE | passed review": round(cond_loss, 6),
            "First BCE": round(first_bce, 6),
            "First CE": round(first_ce, 6),
        },
        "user": int(user_id),
        "size": loss_n,
        "passed review size": int(cond_n),
        "first rating size": int(first_n),
        "first rating": initial_rating,
        "parameters": {
            "0": list(params.tolist())
        }
    }
    return stats

@torch.inference_mode()
def worker_job(config, job_queue, writer_queue, progress_queue):
    device = config.DEVICE
    model = Model().to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH, weights_only=True))

    summary_env = lmdb.open(config.SUMMARY_DB_PATH, readonly=True, lock=False)
    label_filter_env = lmdb.open(config.LABEL_FILTER_DB_PATH, readonly=True, lock=False)

    with summary_env.begin(write=False) as summary_txn:
        with label_filter_env.begin(write=False) as label_filter_txn:
            while True:
                try:
                    user = job_queue.get_nowait()
                    if user is None:
                        return

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
                        min_review_th_s = torch.ones_like(max_review_th_s)
                        encoding_s = decoder_ops.encode(batches, model.encoder_model, min_review_th_s=min_review_th_s, max_review_th_s=max_review_th_s, compress_encoding=True)
                        loss, loss_n, cond_loss, cond_n, rmse_raw, rmse_bins, auc = decoder_ops.decode_full(batches, model.card_model, encoding_s, splits, equalize_review_ths, rmse_bins, device=device, equalize_test_reviews=True)

                        first_stats_accum = None
                        for i in range(encoding_s.size(0)):
                            encoding = encoding_s[i]
                            first_stats = decoder_ops.first_decode(batches, model.first_review_model, encoding, splits[i], min(T, splits[i + 1] - 1), T)
                            if first_stats_accum is None:
                                first_stats_accum = first_stats
                            else:
                                first_stats_accum = decoder_ops.combine_decode_results(first_stats_accum, first_stats)

                        first_bce_loss = first_stats_accum.bce_loss_sum.item() / (1e-7 + first_stats_accum.loss_n.item())
                        first_ce_loss = first_stats_accum.ce_loss_sum.item() / (1e-7 + first_stats_accum.loss_n.item())
                        first_n = first_stats_accum.loss_n.item()

                        print()
                        print(f"User: {user}, RMSE: {rmse_raw:.6f}, LogLoss: {loss:.6f}, RMSE (bins): {rmse_bins:.6f}, AUC: {(-1 if auc is None else auc):.6f}, size: {loss_n}")
                        print(f"Cond CE: {cond_loss.item():.6f}, n: {int(cond_n)}")
                        print(f"First learn: LogLoss: {first_bce_loss:.3f}, CE: {first_ce_loss:.3f}, n: {first_n}")
                        for split_i, parameters in enumerate(encoding_s.cpu().numpy()):
                            first_rating_dist = [round(x, 3) for x in decoder_ops.extract_first_review_dist(model.first_review_model, encoding_s[split_i]).cpu().tolist()]
                            print(f"Split: {split_i}, first rating: {first_rating_dist}")

                        progress_queue.put(1)
                        writer_queue.put(to_json(user, loss.item(), loss_n, cond_loss.item(), cond_n, first_bce_loss, first_ce_loss, first_n, rmse_raw, rmse_bins, auc, first_rating_dist, encoding_s[-1]))
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
    with tqdm(total=total_items, desc="Progress", smoothing=5e-3) as pbar:
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