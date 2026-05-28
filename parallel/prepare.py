from __future__ import annotations

import multiprocessing as mp
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from io import BytesIO
from pathlib import Path
from typing import NamedTuple


if __name__ == "__mp_main__":
    signal.signal(signal.SIGINT, signal.SIG_IGN)

import lmdb
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from config import Config, create_parser
from features import create_features
from parallel.config import (
    LMDB_PATH,
    LMDB_SIZE,
    N_SPLITS,
    USER_MAX_TRAIN_SPLIT_LENGTHS_KEY,
)
from models.model_factory import create_model
from parallel.tensor_lmdb import (
    get_tensor,
    put_tensor,
    tensor_field_keys,
    user_done_key,
    user_tensor_prefix,
)
from parallel.tensors import UserTensorBlob
from utils import rmse_matrix_bin_key


SECONDS_PER_DAY = 86_400
BATCH_LOADER_SEED = 2023

lmdb_env: lmdb.Environment | None = None
worker_config: Config | None = None
USER_TENSOR_FIELDS = tuple(field.name for field in fields(UserTensorBlob))


class BenchmarkTensors(NamedTuple):
    test_index: torch.Tensor
    rmse_bins: torch.Tensor
    split: torch.Tensor
    train_index: torch.Tensor
    split_review_ord: torch.Tensor
    train_split_lengths: torch.Tensor


class RawTensorLayout(NamedTuple):
    tensors: dict[str, torch.Tensor]
    raw_to_grouped_index: np.ndarray


def save_user_tensors(
    txn: lmdb.Transaction,
    user_id: int,
    blob: UserTensorBlob,
) -> None:
    for field_name in USER_TENSOR_FIELDS:
        put_tensor(
            txn,
            user_tensor_prefix(user_id, field_name),
            getattr(blob, field_name),
        )


def load_user_blob_fields(txn: lmdb.Transaction, user_id: int) -> UserTensorBlob:
    return UserTensorBlob.from_dict(
        {
            field_name: get_tensor(txn, user_tensor_prefix(user_id, field_name))
            for field_name in USER_TENSOR_FIELDS
        }
    )


def save_metadata_tensor(
    txn: lmdb.Transaction,
    key: str,
    tensor: torch.Tensor,
) -> None:
    buffer = BytesIO()
    torch.save(tensor, buffer)
    txn.put(key.encode(), buffer.getvalue())


def get_max_train_split_length(blob: UserTensorBlob) -> int:
    if blob.train_split_lengths.numel() == 0:
        return 0
    return int(blob.train_split_lengths.max().item())


def is_current_user_blob(blob: UserTensorBlob, config: Config) -> bool:
    if blob.card_sorted_index.numel() != blob.rating.numel():
        return False
    if blob.seq_len.numel() != blob.rating.numel():
        return False
    if blob.card_last_index.numel() != int((blob.seq_len == 1).sum().item()):
        return False
    if blob.split_review_ord.numel() != blob.train_index.numel():
        return False
    return blob.split.numel() == config.n_splits


def save_global_metadata(user_max_train_split_lengths: list[int]) -> None:
    env = _open_lmdb_env(LMDB_PATH, LMDB_SIZE)
    with env.begin(write=True) as txn:
        save_metadata_tensor(
            txn,
            USER_MAX_TRAIN_SPLIT_LENGTHS_KEY,
            torch.tensor(user_max_train_split_lengths, dtype=torch.int32),
        )
    env.close()


def stop_executor_now(executor: ProcessPoolExecutor, futures: list) -> None:
    for future in futures:
        future.cancel()

    kill_workers = getattr(executor, "kill_workers", None)
    if kill_workers is not None:
        kill_workers()
        return

    terminate_workers = getattr(executor, "terminate_workers", None)
    if terminate_workers is not None:
        terminate_workers()
        return

    processes = getattr(executor, "_processes", None)
    executor.shutdown(wait=False, cancel_futures=True)
    if processes is None:
        return
    for process in processes.values():
        if process.is_alive():
            process.terminate()


def _open_lmdb_env(
    lmdb_path: Path,
    lmdb_size: int,
) -> lmdb.Environment:
    return lmdb.open(
        str(lmdb_path),
        map_size=lmdb_size,
    )


def init_worker(
    lmdb_path: Path,
    lmdb_size: int,
    config: Config,
) -> None:
    global lmdb_env, worker_config
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    lmdb_env = _open_lmdb_env(lmdb_path, lmdb_size)
    worker_config = config


def load_user_parquet(data_path: Path, user_id: int) -> pd.DataFrame:
    return pd.read_parquet(data_path / "revlogs" / f"{user_id=}")


def build_card_grouping(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    raw_index = np.arange(len(df), dtype=np.int32)
    order_df = pd.DataFrame(
        {
            "card_id": df["card_id"].to_numpy(),
            "raw_index": raw_index,
        }
    )
    card_sorted_index = order_df.sort_values(
        by=["card_id", "raw_index"],
        kind="stable",
    )["raw_index"].to_numpy(dtype=np.int32)

    raw_to_grouped_index = np.empty(len(card_sorted_index), dtype=np.int32)
    raw_to_grouped_index[card_sorted_index] = np.arange(
        len(card_sorted_index),
        dtype=np.int32,
    )
    return df.iloc[card_sorted_index], card_sorted_index, raw_to_grouped_index


def build_raw_tensors(df: pd.DataFrame) -> RawTensorLayout:
    grouped_df, card_sorted_index, raw_to_grouped_index = build_card_grouping(df)
    seq_len = grouped_df.groupby("card_id").cumcount() + 1
    card_sizes = grouped_df.groupby("card_id", sort=False, dropna=False).size()
    card_last_index = np.cumsum(card_sizes.to_numpy(dtype=np.int32), dtype=np.int32) - 1
    return RawTensorLayout(
        tensors={
            "ratings": torch.tensor(grouped_df["rating"].to_numpy(), dtype=torch.int8),
            "elapsed_days_int": torch.tensor(
                grouped_df["elapsed_days"].to_numpy(),
                dtype=torch.int32,
            ).clamp_min(0),
            "elapsed_days_real": torch.tensor(
                grouped_df["elapsed_seconds"].to_numpy() / SECONDS_PER_DAY,
                dtype=torch.float32,
            ).clamp_min(0),
            "card_sorted_index": torch.tensor(card_sorted_index, dtype=torch.int32),
            "seq_len": torch.tensor(seq_len.to_numpy(), dtype=torch.int32),
            "card_last_index": torch.tensor(card_last_index, dtype=torch.int32),
        },
        raw_to_grouped_index=raw_to_grouped_index,
    )


def review_th_to_grouped_index(
    review_th: pd.Series,
    raw_to_grouped_index: np.ndarray,
) -> np.ndarray:
    raw_index = review_th.to_numpy(dtype=np.int32) - 1
    return raw_to_grouped_index[raw_index].astype(np.int32)


def pack_user_tensors(
    raw_tensors: dict[str, torch.Tensor],
    benchmark_tensors: BenchmarkTensors,
) -> UserTensorBlob:
    return UserTensorBlob(
        rating=raw_tensors["ratings"],
        elapsed_days_int=raw_tensors["elapsed_days_int"],
        elapsed_days_real=raw_tensors["elapsed_days_real"],
        card_sorted_index=raw_tensors["card_sorted_index"],
        seq_len=raw_tensors["seq_len"],
        card_last_index=raw_tensors["card_last_index"],
        test_index=benchmark_tensors.test_index,
        rmse_bins=benchmark_tensors.rmse_bins,
        split=benchmark_tensors.split,
        train_index=benchmark_tensors.train_index,
        split_review_ord=benchmark_tensors.split_review_ord,
        train_split_lengths=benchmark_tensors.train_split_lengths,
    )


def empty_benchmark_tensors() -> BenchmarkTensors:
    empty_int32 = torch.tensor([], dtype=torch.int32)
    return BenchmarkTensors(
        test_index=empty_int32,
        rmse_bins=torch.tensor([], dtype=torch.int32),
        split=empty_int32,
        train_index=empty_int32,
        split_review_ord=empty_int32,
        train_split_lengths=empty_int32,
    )


def build_batch_order(batch_count: int) -> np.ndarray:
    if batch_count == 0:
        return np.array([], dtype=np.int32)

    generator = torch.Generator()
    generator.manual_seed(BATCH_LOADER_SEED)
    return np.concatenate(
        [
            torch.randperm(batch_count, generator=generator).numpy()
            for _ in range(BATCH_ORDER_EPOCHS)
        ]
    ).astype(np.int32)


def get_training_layout(
    train_set: pd.DataFrame,
    batch_size: int,
    max_seq_len: int,
    raw_to_grouped_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = max(1, int(batch_size))
    if train_set.empty:
        empty = np.array([], dtype=np.int32)
        return empty, empty

    train_set = train_set.copy()
    train_set["_seq_len"] = train_set["tensor"].map(len)
    train_set = train_set[train_set["_seq_len"] <= max_seq_len]
    if train_set.empty:
        empty = np.array([], dtype=np.int32)
        return empty, empty

    train_set["review_i"] = range(len(train_set))
    train_set = train_set.sort_values(by=["_seq_len"], kind="stable")
    train_index = review_th_to_grouped_index(
        train_set["review_th"],
        raw_to_grouped_index,
    )
    return train_index, train_set["review_i"].to_numpy()


def concat_int32(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=np.int32)
    return np.concatenate(arrays).astype(np.int32)


def build_benchmark_tensors(
    df: pd.DataFrame,
    config: Config,
    raw_to_grouped_index: np.ndarray,
) -> BenchmarkTensors:
    feature_df = create_features(df.copy(), config=config)
    if len(feature_df) == 0:
        return empty_benchmark_tensors()

    model = create_model(config)
    batch_size = getattr(model, "batch_size", config.batch_size)
    max_seq_len = config.max_seq_len

    bins = feature_df.apply(rmse_matrix_bin_key, axis=1)
    bin_codes = bins.astype("category").cat.codes.to_numpy()
    test_index_values = review_th_to_grouped_index(
        feature_df["review_th"],
        raw_to_grouped_index,
    )

    test_indices = []
    rmse_bins = []
    split_test_lengths = []
    train_indices = []
    split_review_ords = []
    train_split_lengths = []
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    for train_index, test_index in tscv.split(feature_df):
        test_indices.append(test_index_values[test_index])
        rmse_bins.append(bin_codes[test_index])
        split_test_lengths.append(len(test_index))

        train_set = feature_df.iloc[train_index]
        train_set = model.filter_training_data(train_set)
        split_train_index, split_review_ord = get_training_layout(
            train_set,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            raw_to_grouped_index=raw_to_grouped_index,
        )
        train_indices.append(split_train_index)
        split_review_ords.append(split_review_ord)
        train_split_lengths.append(len(split_train_index))

    test_indices = np.concatenate(test_indices)
    rmse_bins = np.concatenate(rmse_bins)
    train_indices_array = concat_int32(train_indices)
    split_review_ord_array = concat_int32(split_review_ords)
    return BenchmarkTensors(
        test_index=torch.tensor(test_indices, dtype=torch.int32),
        rmse_bins=torch.tensor(rmse_bins, dtype=torch.int32),
        split=torch.tensor(split_test_lengths, dtype=torch.int32),
        train_index=torch.tensor(train_indices_array, dtype=torch.int32),
        split_review_ord=torch.tensor(split_review_ord_array, dtype=torch.int32),
        train_split_lengths=torch.tensor(train_split_lengths, dtype=torch.int32),
    )


def get_user_keys(user_id: int) -> list[str]:
    keys: list[bytes] = []
    for field_name in USER_TENSOR_FIELDS:
        keys.extend(tensor_field_keys(user_tensor_prefix(user_id, field_name)))
    keys.append(user_done_key(user_id))
    return [key.decode() for key in keys]


def process_user(user_id: int) -> int:
    if lmdb_env is None:
        raise RuntimeError("LMDB environment was not initialized.")
    if worker_config is None:
        raise RuntimeError("Worker config was not initialized.")

    user_keys = get_user_keys(user_id)
    with lmdb_env.begin(write=False) as txn:
        if all(txn.get(key.encode()) is not None for key in user_keys):
            try:
                blob = load_user_blob_fields(txn, user_id)
            except (KeyError, TypeError, RuntimeError):
                blob = None
            if blob is not None and is_current_user_blob(blob, worker_config):
                return get_max_train_split_length(blob)

    df = load_user_parquet(worker_config.data_path, user_id)
    # print(dds"].min())
    # print("temp prune")
    # df = df[df["card_id"] <= 1]
    # print(df)
    assert len(df) > 0
    raw_layout = build_raw_tensors(df)
    try:
        benchmark_tensors = build_benchmark_tensors(
            df,
            worker_config,
            raw_layout.raw_to_grouped_index,
        )
    except ValueError as err:
        if "No data after handling outliers" in str(err):
            return user_id
        raise

    with lmdb_env.begin(write=True) as txn:
        blob = pack_user_tensors(raw_layout.tensors, benchmark_tensors)
        save_user_tensors(txn, user_id, blob)
        txn.put(user_done_key(user_id), b"true")

    return get_max_train_split_length(blob)


def main() -> None:
    mp.set_start_method("spawn", force=True)

    parser = create_parser()
    parser.set_defaults(algo="FSRS-7", short=True, secs=True)
    args, _ = parser.parse_known_args()
    config = Config(args)
    user_ids = list(range(1, 10001))

    executor = ProcessPoolExecutor(
        max_workers=config.num_processes,
        initializer=init_worker,
        initargs=(LMDB_PATH, LMDB_SIZE, config),
    )
    futures = []
    user_max_train_split_lengths = [0 for _ in user_ids]
    try:
        futures = [
            executor.submit(process_user, user_id)
            for user_id in user_ids
        ]
        future_to_index = {
            future: idx
            for idx, future in enumerate(futures)
        }

        for future in tqdm(as_completed(futures), total=len(futures), smoothing=0.1):
            user_max_train_split_lengths[future_to_index[future]] = future.result()
    except KeyboardInterrupt:
        stop_executor_now(executor, futures)
        os._exit(130)
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown()
        save_global_metadata(user_max_train_split_lengths)


if __name__ == "__main__":
    main()
