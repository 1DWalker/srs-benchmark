from io import BytesIO
import json
import math
import multiprocessing
import lmdb
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from itertools import accumulate
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

import random

from utils import load_tensor, parse_toml, save_tensor

random.seed(1234)

def process_df(df, dtype, device):
    df = df.sort_values(["card_id", "review_th"]).copy()
    df = df[df["rating"].isin([1, 2, 3, 4])]
    df["elapsed_seconds"] = df["elapsed_seconds"].clip(lower=0)
    df["elapsed_days_int"] = df["elapsed_days"].clip(lower=0)
    df["is_same_day"] = (df["elapsed_days"] == 0)
    df["elapsed_days_real"] = df["elapsed_seconds"] / 86400.0
    df.drop(columns="elapsed_days", inplace=True)
    g = df.groupby("card_id", sort=False)
    df["i"] = g.cumcount() + 1
    df["has_label"] = g.cumcount(ascending=False) != 0
    df["label_review_th"] = g["review_th"].shift(-1).fillna(-1)
    df["label_elapsed_days_int"] = g["elapsed_days_int"].shift(-1).fillna(0)
    df["label_elapsed_days_real"] = g["elapsed_days_real"].shift(-1).fillna(0)
    df["label_rating"] = g["rating"].shift(-1).fillna(0)
    df["label_is_same_day"] = g["is_same_day"].shift(-1).fillna(False)

    out = {}
    for cid, group in g:
        arr = group
        out[cid] = (
            torch.as_tensor(arr["review_th"].to_numpy(copy=True), dtype=torch.int32, device=device),
            torch.as_tensor(arr["elapsed_days_int"].to_numpy(copy=True), dtype=torch.int32, device=device),
            torch.as_tensor(arr["elapsed_days_real"].to_numpy(copy=True), dtype=dtype, device=device),
            torch.as_tensor(arr["rating"].to_numpy(copy=True), dtype=dtype, device=device),

            torch.as_tensor(arr["label_elapsed_days_int"].to_numpy(copy=True), dtype=dtype, device=device),
            torch.as_tensor(arr["label_elapsed_days_real"].to_numpy(copy=True), dtype=dtype, device=device),
            torch.as_tensor(arr["label_rating"].to_numpy(copy=True), dtype=dtype, device=device),
            torch.as_tensor(arr["label_review_th"].to_numpy(copy=True), dtype=torch.int32, device=device),
            torch.as_tensor(
                arr["label_is_same_day"].to_numpy(dtype=np.bool_, copy=True),
                dtype=torch.bool,
                device=device,
            ),
            torch.as_tensor(arr["has_label"].to_numpy(copy=True), device=device),
        )

    return out

def greedy_splits(freqs, factor, allowed_excess_in_one_step=20000):
    # 3000 -> 4.1
    # 10000 -> 2.7 (2.4 lowest)
    # 20000 -> 2.4? (2.15 lowest)
    # 30000 -> 2.7 (2.2 lowest)
    # 100k -> 3
    lens = list(reversed(sorted(freqs.keys())))
    splits = []
    l = 0
    while l < len(lens):
        r = l
        used = lens[l] * freqs[lens[l]]
        waste = 0
        while r + 1 < len(lens):
            next_used = used + lens[r + 1] * freqs[lens[r + 1]]
            extra_waste = (lens[l] - lens[r + 1]) * freqs[lens[r + 1]]
            next_waste = waste + extra_waste
            if (
                (factor - 1) * next_used >= next_waste
                and next_waste <= allowed_excess_in_one_step
            ):
                used = next_used
                waste = next_waste
                r += 1
            else:
                break

        splits.append(lens[l])
        l = r + 1

    splits.reverse()
    return splits

def process(user_id, config):
    df_revlogs = pd.read_parquet(
        config.DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    df_revlogs["review_th"] = range(1, df_revlogs.shape[0] + 1)
    df_revlogs.drop(columns=["user_id"], inplace=True)
    card_id_to_tensors = process_df(df_revlogs, config.DTYPE, torch.device("cpu"))
    
    sizes_freq_dict = {}
    for card_id, tensors in card_id_to_tensors.items():
        features = tensors[0]
        if features.size(0) not in sizes_freq_dict:
            sizes_freq_dict[features.size(0)] = 0
        sizes_freq_dict[features.size(0)] += 1


    factor = min(config.MAX_FACTOR, config.MAX_TOTAL_SIZE / len(df_revlogs))
    splits = greedy_splits(sizes_freq_dict, factor=factor)
    def get_group_i(size):
        i = 0
        while size > splits[i]:
            i += 1
        return i
    
    groups = [[] for _ in range(len(splits))]
    for card_id, tensors in card_id_to_tensors.items():
        features = tensors[0]
        groups[get_group_i(features.size(0))].append(tensors)

    total_size = 0
    result = []
    for group_i, group_lists in enumerate(groups):
        if len(group_lists) == 0:
            continue

        feature_review_th, feature_elapsed_days_int, feature_elapsed_days_real, feature_rating, label_elapsed_days_int, label_elapsed_days_real, label_rating, label_review_th, label_is_same_day, has_label = list(zip(*group_lists))
        feature_review_th = pad_sequence(feature_review_th, batch_first=True, padding_value=0)
        feature_elapsed_days_int = pad_sequence(feature_elapsed_days_int, batch_first=True, padding_value=0)
        feature_elapsed_days_real = pad_sequence(feature_elapsed_days_real, batch_first=True, padding_value=0)
        feature_rating = pad_sequence(feature_rating, batch_first=True, padding_value=0)
        label_elapsed_days_int = pad_sequence(label_elapsed_days_int, batch_first=True, padding_value=0)
        label_elapsed_days_real = pad_sequence(label_elapsed_days_real, batch_first=True, padding_value=0)
        label_rating = pad_sequence(label_rating, batch_first=True, padding_value=0)
        label_review_th = pad_sequence(
            label_review_th, batch_first=True, padding_value=-1
        )
        label_is_same_day = pad_sequence(label_is_same_day, batch_first=True, padding_value=0)
        has_label = pad_sequence(has_label, batch_first=True, padding_value=0)
        total_size += feature_rating.size(0) * feature_rating.size(1)
        result.append((feature_review_th, feature_elapsed_days_int, feature_elapsed_days_real, feature_rating, label_elapsed_days_int, label_elapsed_days_real, label_rating, label_review_th, label_is_same_day, has_label))
    
    print("Total size:", total_size, len(df_revlogs), total_size / len(df_revlogs))
    assert total_size / len(df_revlogs) <= 1.01 * config.MAX_FACTOR
    assert total_size <= 1.01 * config.MAX_TOTAL_SIZE

    return result


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
            for i, (feature_review_th, feature_elapsed_days_int, feature_elapsed_days_real, feature_rating, label_elapsed_days_int, label_elapsed_days_real, label_rating, label_review_th, label_is_same_day, has_label) in enumerate(
                tensors
            ):
                save_tensor(txn, f"{user_id}_feature_review_th_{i}", feature_review_th)
                save_tensor(txn, f"{user_id}_feature_elapsed_days_int_{i}", feature_elapsed_days_int)
                save_tensor(txn, f"{user_id}_feature_elapsed_days_real_{i}", feature_elapsed_days_real)
                save_tensor(txn, f"{user_id}_feature_rating_{i}", feature_rating)
                save_tensor(txn, f"{user_id}_label_elapsed_days_int_{i}", label_elapsed_days_int)
                save_tensor(txn, f"{user_id}_label_elapsed_days_real_{i}", label_elapsed_days_real)
                save_tensor(txn, f"{user_id}_label_rating_{i}", label_rating)
                save_tensor(txn, f"{user_id}_label_review_th_{i}", label_review_th)
                save_tensor(txn, f"{user_id}_label_is_same_day_{i}", label_is_same_day)
                save_tensor(txn, f"{user_id}_has_label_{i}", has_label)

            save_tensor(txn, f"{user_id}_batches", torch.tensor(len(tensors)))
            txn.put(f"{user_id}_done".encode(), "true".encode())
            print("Done", user_id)


def progress_tracker(total_items, progress_queue):
    with tqdm(total=total_items, desc="Generating Data", smoothing=1e-2) as pbar:
        for _ in range(total_items):
            progress_queue.get()
            pbar.update(1)


def main(config):
    USER_IDS = list(range(config.USER_START, config.USER_END + 1))

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
    # unprocessed_users = list(range(1, 101))


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


if __name__ == "__main__":
    config = parse_toml()
    # process(1, config)
    main(config)
    # exit()

    # # for user_id in range(2, 3):
    # df_revlogs = pd.read_parquet(
    #     config.DATA_PATH / "revlogs", filters=[("user_id", "=", 1)]
    # )
    # df_revlogs["review_th"] = range(1, df_revlogs.shape[0] + 1)
    # df_revlogs = df_revlogs[df_revlogs["review_th"] <= 6]
    # print(df_revlogs)
    # df_revlogs.drop(columns=["user_id"], inplace=True)
    # df_revlogs.drop(df_revlogs[~df_revlogs["rating"].isin([1, 2, 3, 4])].index, inplace=True)
    # print(df_revlogs)
    #     # df = process_df(df_revlogs, config.DTYPE, torch.device("cpu"))
    # # print(df)
