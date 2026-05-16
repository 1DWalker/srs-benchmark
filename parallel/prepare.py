from __future__ import annotations

import multiprocessing as mp
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


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
from rwkv.utils import save_tensor
from utils import get_bin


LMDB_PATH = Path("parallel_db")
LMDB_SIZE = 50_000_000_000
SECONDS_PER_DAY = 86_400

lmdb_env: lmdb.Environment | None = None
worker_config: Config | None = None


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


def build_benchmark_tensors(
    df: pd.DataFrame,
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_df = create_features(df.copy(), config=config)
    if len(feature_df) == 0:
        empty = torch.tensor([], dtype=torch.int32)
        return empty, empty, empty

    bins = feature_df.apply(get_bin, axis=1)
    bin_codes = bins.astype("category").cat.codes.to_numpy()
    test_index_values = feature_df["review_th"].to_numpy() - 1

    test_indices = []
    rmse_bins = []
    split_test_ranges = []
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    for _, test_index in tscv.split(feature_df):
        test_indices.append(test_index_values[test_index])
        rmse_bins.append(bin_codes[test_index])
        split_test_ranges.append(min(test_index_values[test_index]))

    split_test_ranges.append(int(1e9))
    test_indices = np.concatenate(test_indices)
    rmse_bins = np.concatenate(rmse_bins)
    assert np.array_equal(np.sort(test_indices), test_indices)
    test_index_tensor = torch.tensor(test_indices, dtype=torch.int32)
    rmse_bins_tensor = torch.tensor(rmse_bins, dtype=torch.int8)
    split_tensor = torch.tensor(split_test_ranges, dtype=torch.int32)
    return test_index_tensor, rmse_bins_tensor, split_tensor


def get_user_keys(user_id: int) -> list[str]:
    return [
        f"{user_id}_ratings",
        f"{user_id}_elapsed_days_int",
        f"{user_id}_elapsed_days_real",
        f"{user_id}_test_index",
        f"{user_id}_rmse_bins",
        f"{user_id}_split",
        f"{user_id}_done",
    ]


def process_user(user_id: int) -> int:
    if lmdb_env is None:
        raise RuntimeError("LMDB environment was not initialized.")
    if worker_config is None:
        raise RuntimeError("Worker config was not initialized.")

    user_keys = get_user_keys(user_id)
    with lmdb_env.begin(write=False) as txn:
        if all(txn.get(key.encode()) is not None for key in user_keys):
            return user_id

    df = load_user_parquet(worker_config.data_path, user_id)
    try:
        test_index_tensor, rmse_bins_tensor, split_tensor = build_benchmark_tensors(
            df,
            worker_config,
        )
    except ValueError as err:
        if "No data after handling outliers" in str(err):
            return user_id
        raise

    with lmdb_env.begin(write=True) as txn:
        save_user_tensors(txn, user_id, df)
        save_tensor(txn, f"{user_id}_test_index", test_index_tensor)
        save_tensor(txn, f"{user_id}_rmse_bins", rmse_bins_tensor)
        save_tensor(txn, f"{user_id}_split", split_tensor)
        txn.put(f"{user_id}_done".encode(), b"true")

    return user_id


def main() -> None:
    mp.set_start_method("spawn", force=True)

    parser = create_parser()
    args, _ = parser.parse_known_args()
    config = Config(args)
    user_ids = list(range(1, 10_001))

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
