from __future__ import annotations

import multiprocessing as mp
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from models.model_factory import create_model
from rwkv.utils import save_tensor
from utils import get_bin


LMDB_PATH = Path("parallel_db")
LMDB_SIZE = 50_000_000_000
SECONDS_PER_DAY = 86_400
BATCH_LOADER_SEED = 2023

lmdb_env: lmdb.Environment | None = None
worker_config: Config | None = None


class BenchmarkTensors(NamedTuple):
    test_index: torch.Tensor
    rmse_bins: torch.Tensor
    split: torch.Tensor
    batch_order: torch.Tensor
    train_index: torch.Tensor
    train_batch_lengths: torch.Tensor
    train_split_lengths: torch.Tensor


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


def save_user_tensors(txn: lmdb.Transaction, user_id: int, df: pd.DataFrame) -> None:
    ratings = torch.tensor(df["rating"].to_numpy(), dtype=torch.int8)
    elapsed_days_int = torch.tensor(df["elapsed_days"].to_numpy(), dtype=torch.int32)
    elapsed_days_real = torch.tensor(
        df["elapsed_seconds"].to_numpy() / SECONDS_PER_DAY,
        dtype=torch.float32,
    )

    save_tensor(txn, f"{user_id}_ratings", ratings)
    save_tensor(txn, f"{user_id}_elapsed_days_int", elapsed_days_int)
    save_tensor(txn, f"{user_id}_elapsed_days_real", elapsed_days_real)


def empty_benchmark_tensors() -> BenchmarkTensors:
    empty_int32 = torch.tensor([], dtype=torch.int32)
    return BenchmarkTensors(
        test_index=empty_int32,
        rmse_bins=torch.tensor([], dtype=torch.int8),
        split=empty_int32,
        batch_order=empty_int32,
        train_index=empty_int32,
        train_batch_lengths=empty_int32,
        train_split_lengths=empty_int32,
    )


def get_training_layout(
    train_set: pd.DataFrame,
    batch_size: int,
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_size = max(1, int(batch_size))
    if train_set.empty:
        empty = np.array([], dtype=np.int32)
        return empty, empty, empty

    train_set = train_set.copy()
    train_set["_seq_len"] = train_set["tensor"].map(len)
    train_set = train_set[train_set["_seq_len"] <= max_seq_len]
    if train_set.empty:
        empty = np.array([], dtype=np.int32)
        return empty, empty, empty

    train_set = train_set.sort_values(by=["_seq_len"], kind="stable")
    train_index = (train_set["review_th"].to_numpy() - 1).astype(np.int32)

    batch_lengths = []
    for start in range(0, len(train_index), batch_size):
        end = min(start + batch_size, len(train_index))
        batch_lengths.append(end - start)
    batch_lengths_array = np.array(batch_lengths, dtype=np.int32)

    generator = torch.Generator()
    generator.manual_seed(BATCH_LOADER_SEED)
    batch_order = torch.randperm(len(batch_lengths), generator=generator).numpy()
    batch_order = batch_order.astype(np.int32)

    return train_index, batch_lengths_array, batch_order


def concat_int32(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=np.int32)
    return np.concatenate(arrays).astype(np.int32)


def build_benchmark_tensors(df: pd.DataFrame, config: Config) -> BenchmarkTensors:
    feature_df = create_features(df.copy(), config=config)
    if len(feature_df) == 0:
        return empty_benchmark_tensors()

    model = create_model(config)
    batch_size = getattr(model, "batch_size", config.batch_size)
    max_seq_len = config.max_seq_len

    bins = feature_df.apply(get_bin, axis=1)
    bin_codes = bins.astype("category").cat.codes.to_numpy()
    test_index_values = feature_df["review_th"].to_numpy() - 1

    test_indices = []
    rmse_bins = []
    split_test_ranges = []
    train_indices = []
    train_batch_lengths = []
    batch_orders = []
    train_split_lengths = []
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    for train_index, test_index in tscv.split(feature_df):
        test_indices.append(test_index_values[test_index])
        rmse_bins.append(bin_codes[test_index])
        split_test_ranges.append(min(test_index_values[test_index]))

        train_set = feature_df.iloc[train_index]
        train_set = model.filter_training_data(train_set)
        split_train_index, split_batch_lengths, split_batch_order = get_training_layout(
            train_set,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        train_indices.append(split_train_index)
        train_batch_lengths.append(split_batch_lengths)
        batch_orders.append(split_batch_order)
        train_split_lengths.append(len(split_batch_lengths))

    split_test_ranges.append(int(1e9))
    test_indices = np.concatenate(test_indices)
    rmse_bins = np.concatenate(rmse_bins)
    train_indices_array = concat_int32(train_indices)
    train_batch_lengths_array = concat_int32(train_batch_lengths)
    batch_order_array = concat_int32(batch_orders)
    assert np.array_equal(np.sort(test_indices), test_indices)
    return BenchmarkTensors(
        test_index=torch.tensor(test_indices, dtype=torch.int32),
        rmse_bins=torch.tensor(rmse_bins, dtype=torch.int8),
        split=torch.tensor(split_test_ranges, dtype=torch.int32),
        batch_order=torch.tensor(batch_order_array, dtype=torch.int32),
        train_index=torch.tensor(train_indices_array, dtype=torch.int32),
        train_batch_lengths=torch.tensor(train_batch_lengths_array, dtype=torch.int32),
        train_split_lengths=torch.tensor(train_split_lengths, dtype=torch.int32),
    )


def get_user_keys(user_id: int) -> list[str]:
    return [
        f"{user_id}_ratings",
        f"{user_id}_elapsed_days_int",
        f"{user_id}_elapsed_days_real",
        f"{user_id}_test_index",
        f"{user_id}_rmse_bins",
        f"{user_id}_split",
        f"{user_id}_batch_order",
        f"{user_id}_train_index",
        f"{user_id}_train_batch_lengths",
        f"{user_id}_train_split_lengths",
        f"{user_id}_done",
    ]


def process_user(user_id: int) -> int:
    if lmdb_env is None:
        raise RuntimeError("LMDB environment was not initialized.")
    if worker_config is None:
        raise RuntimeError("Worker config was not initialized.")

    user_keys = get_user_keys(user_id)
    # with lmdb_env.begin(write=False) as txn:
    #     if all(txn.get(key.encode()) is not None for key in user_keys):
    #         return user_id

    df = load_user_parquet(worker_config.data_path, user_id)
    try:
        benchmark_tensors = build_benchmark_tensors(df, worker_config)
    except ValueError as err:
        if "No data after handling outliers" in str(err):
            return user_id
        raise

    with lmdb_env.begin(write=True) as txn:
        save_user_tensors(txn, user_id, df)
        save_tensor(txn, f"{user_id}_test_index", benchmark_tensors.test_index)
        save_tensor(txn, f"{user_id}_rmse_bins", benchmark_tensors.rmse_bins)
        save_tensor(txn, f"{user_id}_split", benchmark_tensors.split)
        save_tensor(txn, f"{user_id}_batch_order", benchmark_tensors.batch_order)
        save_tensor(txn, f"{user_id}_train_index", benchmark_tensors.train_index)
        save_tensor(
            txn,
            f"{user_id}_train_batch_lengths",
            benchmark_tensors.train_batch_lengths,
        )
        save_tensor(
            txn,
            f"{user_id}_train_split_lengths",
            benchmark_tensors.train_split_lengths,
        )
        txn.put(f"{user_id}_done".encode(), b"true")

    return user_id


def main() -> None:
    mp.set_start_method("spawn", force=True)

    parser = create_parser()
    args, _ = parser.parse_known_args()
    config = Config(args)
    # user_ids = list(range(1, 10_001))
    user_ids = list(range(1, 2))

    executor = ProcessPoolExecutor(
        max_workers=config.num_processes,
        initializer=init_worker,
        initargs=(LMDB_PATH, LMDB_SIZE, config),
    )
    futures = []
    try:
        futures = [
            executor.submit(process_user, user_id)
            for user_id in user_ids
        ]

        for future in tqdm(as_completed(futures), total=len(futures), smoothing=0.03):
            future.result()
    except KeyboardInterrupt:
        stop_executor_now(executor, futures)
        os._exit(130)
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown()


if __name__ == "__main__":
    main()
