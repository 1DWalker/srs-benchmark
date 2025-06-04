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

from rwkv.utils import load_tensor
from utils import parse_toml

random.seed(1234)

def process_df(df, dtype, device):
    df.sort_values(by=["card_id", "review_th"], inplace=True)
    df.drop(df[~df["rating"].isin([1, 2, 3, 4])].index, inplace=True)
    df["elapsed_seconds"] = df["elapsed_seconds"].map(lambda x: max(0, x))
    df["is_same_day"] = (df["elapsed_days"] == 0).astype(int)
    df["elapsed_days_int"] = df["elapsed_days"].map(lambda x: max(0, x))
    df.drop(columns=["elapsed_days"], inplace=True)
    df["elapsed_days_real"] = df["elapsed_seconds"] / 86400
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    df["i"] = df.groupby("card_id").cumcount() + 1
    df["has_label"] = (df.groupby("card_id").cumcount(ascending=False) != 0).astype(int)

    def process_group(group):
        feature_elapsed_days_int = torch.tensor(group["elapsed_days_int"].to_numpy(), dtype=torch.int, device=device)
        feature_elapsed_days_real = torch.tensor(group["elapsed_days_real"].to_numpy(), dtype=dtype, device=device)
        feature_rating = torch.tensor(group["rating"].to_numpy(), dtype=dtype, device=device)
        label_review_th = torch.tensor(
            group["review_th"].shift(-1).fillna(-1).to_numpy(),
            dtype=torch.int32,
            device=device,
        )
        label_elapsed_days_int = torch.tensor(
            group["elapsed_days_int"].shift(-1).fillna(0).to_numpy(),
            dtype=dtype,
            device=device,
        )
        label_elapsed_days_real = torch.tensor(
            group["elapsed_days_real"].shift(-1).fillna(0).to_numpy(),
            dtype=dtype,
            device=device,
        )
        label_y = torch.tensor(
            group["y"].shift(-1).fillna(0).to_numpy(), dtype=dtype, device=device
        )
        label_is_same_day = torch.tensor(group["is_same_day"].shift(-1).fillna(0).to_numpy(), dtype=torch.bool, device=device)
        label_is_equalize = torch.tensor(group["is_equalize_review"].shift(-1).fillna(0).to_numpy(), dtype=torch.bool, device=device)
        has_label = torch.tensor(group["has_label"].to_numpy(), device=device)
        return feature_elapsed_days_int, feature_elapsed_days_real, feature_rating, label_elapsed_days_int, label_elapsed_days_real, label_y, label_review_th, label_is_same_day, label_is_equalize, has_label

    card_id_to_group = {
        card_id: group.reset_index(drop=True)
        for card_id, group in df.groupby("card_id")
    }
    card_id_to_tensors = {
        card_id: process_group(group) for card_id, group in card_id_to_group.items()
    }
    return card_id_to_tensors

def greedy_splits(freqs, factor, allowed_excess_in_one_step=100000):
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
                and extra_waste <= allowed_excess_in_one_step
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
    equalize_env = lmdb.open(config.LABEL_FILTER_DB_PATH, readonly=True, lock=False)
    with equalize_env.begin(write=False) as txn:
        equalize_review_ths_list = load_tensor(txn, f"{user_id}_review_ths", device="cpu").tolist()
    equalize_review_ths_set = set(equalize_review_ths_list)
    df_revlogs["is_equalize_review"] = df_revlogs["review_th"].isin(equalize_review_ths_set).astype(int)
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

        feature_elapsed_days_int, feature_elapsed_days_real, feature_rating, label_elapsed_days_int, label_elapsed_days_real, label_y, label_review_th, label_is_same_day, label_is_equalize, has_label = list(zip(*group_lists))
        feature_elapsed_days_int = pad_sequence(feature_elapsed_days_int, batch_first=True, padding_value=0)
        feature_elapsed_days_real = pad_sequence(feature_elapsed_days_real, batch_first=True, padding_value=0)
        feature_rating = pad_sequence(feature_rating, batch_first=True, padding_value=0)
        label_elapsed_days_int = pad_sequence(label_elapsed_days_int, batch_first=True, padding_value=0)
        label_elapsed_days_real = pad_sequence(label_elapsed_days_real, batch_first=True, padding_value=0)
        label_y = pad_sequence(label_y, batch_first=True, padding_value=0)
        label_review_th = pad_sequence(
            label_review_th, batch_first=True, padding_value=-1
        )
        label_is_same_day = pad_sequence(label_is_same_day, batch_first=True, padding_value=0)
        label_is_equalize = pad_sequence(label_is_equalize, batch_first=True, padding_value=0)
        has_label = pad_sequence(has_label, batch_first=True, padding_value=0)
        total_size += feature_rating.size(0) * feature_rating.size(1)
        result.append((feature_elapsed_days_int, feature_elapsed_days_real, feature_rating, label_elapsed_days_int, label_elapsed_days_real, label_y, label_review_th, label_is_same_day, label_is_equalize, has_label))
    
    print("Total size:", total_size, len(df_revlogs), total_size / len(df_revlogs))
    assert total_size / len(df_revlogs) <= 1.01 * config.MAX_FACTOR
    assert total_size <= 1.01 * config.MAX_TOTAL_SIZE

    return result


def job(user_id, config, writer_queue, progress_queue):
    writer_queue.put((user_id, process(user_id, config)))
    progress_queue.put(1)


def save_tensor(txn, key, tensor):
    tensor = tensor.clone().contiguous()
    buffer = BytesIO()
    torch.save(tensor, buffer)
    txn.put(key.encode(), buffer.getvalue())


def save_job(lmdb_path, lmdb_size, writer_queue):
    print(f"lmdb size: {lmdb_size}")
    env = lmdb.open(lmdb_path, lmdb_size)
    while True:
        sample = writer_queue.get()
        if sample is None:
            break
        user_id, tensors = sample

        with env.begin(write=True) as txn:
            for i, (feature_elapsed_days_int, feature_elapsed_days_real, feature_rating, label_elapsed_days_int, label_elapsed_days_real, label_y, label_review_th, label_is_same_day, label_is_equalize, has_label) in enumerate(
                tensors
            ):
                save_tensor(txn, f"{user_id}_feature_elapsed_days_int_{i}", feature_elapsed_days_int)
                save_tensor(txn, f"{user_id}_feature_elapsed_days_real_{i}", feature_elapsed_days_real)
                save_tensor(txn, f"{user_id}_feature_rating_{i}", feature_rating)
                save_tensor(txn, f"{user_id}_label_elapsed_days_int_{i}", label_elapsed_days_int)
                save_tensor(txn, f"{user_id}_label_elapsed_days_real_{i}", label_elapsed_days_real)
                save_tensor(txn, f"{user_id}_label_y_{i}", label_y)
                save_tensor(txn, f"{user_id}_label_review_th_{i}", label_review_th)
                save_tensor(txn, f"{user_id}_label_is_same_day_{i}", label_is_same_day)
                save_tensor(txn, f"{user_id}_label_is_equalize_{i}", label_is_equalize)
                save_tensor(txn, f"{user_id}_has_label_{i}", has_label)

            save_tensor(txn, f"{user_id}_batches", torch.tensor(len(tensors)))
            txn.put(f"{user_id}_done".encode(), "true".encode())
            print("Done", user_id)


def progress_tracker(total_items, progress_queue):
    with tqdm(total=total_items, desc="Generating Data") as pbar:
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
    unprocessed_users = list(range(1, 11))


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
    # process(8902, config)
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
