
import multiprocessing
import lmdb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import load_tensor, parse_toml, save_tensor


def process(user_id, config):
    df = pd.read_parquet(
        config.DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    df["review_th"] = range(1, df.shape[0] + 1)
    len_before = len(df)
    df.drop(df[~df["rating"].isin([1, 2, 3, 4])].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    assert len_before == len(df), f"{user_id} has invalid ratings, review_th might be incorrect"

    equalize_env = lmdb.open(config.LABEL_FILTER_DB_PATH, readonly=True, lock=False)
    with equalize_env.begin(write=False) as txn:
        equalize_review_ths = load_tensor(txn, f"{user_id}_review_ths", device="cpu")
        # rmse_bins = load_tensor(txn, f"{user_id}_rmse_bins", device="cpu")
    equalize_review_ths_set = set(equalize_review_ths.tolist())

    df["is_equalize_review"] = df["review_th"].isin(equalize_review_ths_set).astype(int)
    df["elapsed_days_real"] = df["elapsed_seconds"].map(lambda x: max(0, x)) / 86400
    df["elapsed_days_int"] = df["elapsed_days"].map(lambda x: max(0, x))
    df["label_elapsed_days_real"] = df.groupby("card_id")["elapsed_days_real"].shift(-1).fillna(0)
    df["label_elapsed_days_int"] = df.groupby("card_id")["elapsed_days_int"].shift(-1).fillna(0)

    card_locs = df.groupby("card_id")["review_th"].apply(list).to_dict()
    ordered_card_ids = sorted(card_locs, key=lambda k: len(card_locs[k]), reverse=True)

    perm = []
    card_locs_dict = {}
    for card_id in ordered_card_ids:
        card_locs_dict[card_id] = len(perm)
        for review_th in card_locs[card_id]:
            perm.append(review_th - 1)

    perm_inv = [-1 for _ in range(len(perm))]
    for i in range(len(perm)):
        perm_inv[perm[i]] = i

    with torch.no_grad():
        perm_T_tensor = torch.tensor(perm, dtype=torch.int)
        perm_inv_T_tensor = torch.tensor(perm_inv, dtype=torch.int)
        card_locs_T = torch.tensor(df["card_id"].map(card_locs_dict), dtype=torch.int)
        packed_review_th_T = torch.tensor(df["review_th"], dtype=torch.int)[perm]
        packed_rating_T = torch.tensor(df["rating"], dtype=torch.int)[perm]
        packed_elapsed_days_real_T = torch.tensor(df["elapsed_days_real"], dtype=config.DTYPE)[perm]
        packed_elapsed_days_int_T = torch.tensor(df["elapsed_days_int"], dtype=config.DTYPE)[perm]
        packed_label_elapsed_days_real_T = torch.tensor(df["label_elapsed_days_real"], dtype=config.DTYPE)[perm]
        packed_label_elapsed_days_int_T = torch.tensor(df["label_elapsed_days_int"], dtype=config.DTYPE)[perm]

    return packed_review_th_T, packed_rating_T, packed_elapsed_days_real_T, packed_elapsed_days_int_T, packed_label_elapsed_days_real_T, packed_label_elapsed_days_int_T, perm_T_tensor, perm_inv_T_tensor, card_locs_T

def job(user_id, config, writer_queue, progress_queue):
    writer_queue.put((user_id, process(user_id, config)))
    progress_queue.put(1)

def save_job(lmdb_path, lmdb_size, writer_queue):
    print(f"lmdb size: {lmdb_size}")
    env = lmdb.open(lmdb_path, lmdb_size)
    while True:
        sample = writer_queue.get()
        if sample is None:
            break
        user_id, tensors = sample

        with env.begin(write=True) as txn:
            (
                packed_review_th_T,
                packed_rating_T,
                packed_elapsed_days_real_T,
                packed_elapsed_days_int_T,
                packed_label_elapsed_days_real_T,
                packed_label_elapsed_days_int_T,
                perm_T_tensor,
                perm_inv_T_tensor,
                card_locs_T,
            ) = tensors
            save_tensor(txn, f"{user_id}_packed_review_th_T", packed_review_th_T)
            save_tensor(txn, f"{user_id}_packed_rating_T", packed_rating_T)
            save_tensor(txn, f"{user_id}_packed_elapsed_days_real_T", packed_elapsed_days_real_T)
            save_tensor(txn, f"{user_id}_packed_elapsed_days_int_T", packed_elapsed_days_int_T)
            save_tensor(txn, f"{user_id}_packed_label_elapsed_days_real_T", packed_label_elapsed_days_real_T)
            save_tensor(txn, f"{user_id}_packed_label_elapsed_days_int_T", packed_label_elapsed_days_int_T)
            save_tensor(txn, f"{user_id}_perm_T_tensor", perm_T_tensor)
            save_tensor(txn, f"{user_id}_perm_inv_T_tensor", perm_inv_T_tensor)
            save_tensor(txn, f"{user_id}_card_locs_T", card_locs_T)
            txn.put(f"{user_id}_done".encode(), "true".encode())
            print("Done", user_id, packed_rating_T.shape)

def progress_tracker(total_items, progress_queue):
    with tqdm(total=total_items, desc="Generating Data") as pbar:
        for _ in range(total_items):
            progress_queue.get()
            pbar.update(1)

def main(config):
    USER_IDS = list(range(config.USER_START, config.USER_END + 1))
    if 4371 in USER_IDS:
        USER_IDS.remove(4371)
        print("Removed user 4371.")

    done_set = set()
    unprocessed_users = []
    env = lmdb.open(config.DB_PATH)
    with env.begin(write=False) as txn:
        for user_id in USER_IDS:
            if txn.get(f"{user_id}_done".encode()) is not None:
                done_set.add(user_id)
            else:
                unprocessed_users.append(user_id)
    env.close()
    print("unprocessed:", unprocessed_users)

    with multiprocessing.Manager() as manager:
        writer_queue = manager.Queue()
        writer = multiprocessing.Process(
            target=save_job, args=(config.DB_PATH, config.DB_SIZE, writer_queue)
        )
        writer.start()

        progress_queue = manager.Queue()
        progress_process = multiprocessing.Process(
            target=progress_tracker, args=(len(unprocessed_users), progress_queue)
        )
        progress_process.start()

        with multiprocessing.Pool(processes=config.PROCESSES) as pool:
            pool.starmap(
                job,
                [
                    (user_id, config, writer_queue, progress_queue)
                    for user_id in unprocessed_users
                ],
            )

        writer_queue.put(None)
        writer.join()
        progress_process.terminate()

if __name__ == '__main__':
    config = parse_toml()
    main(config)