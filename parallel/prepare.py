from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lmdb
import pandas as pd
import torch
from tqdm.auto import tqdm  # type: ignore

from config import Config, create_parser
from rwkv.utils import save_tensor


LMDB_PATH = Path("prepare_db")
LMDB_SIZE = 50_000_000_000
SECONDS_PER_DAY = 86_400

lmdb_env: lmdb.Environment | None = None


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
) -> None:
    global lmdb_env
    lmdb_env = _open_lmdb_env(lmdb_path, lmdb_size)


def load_user_parquet(data_path: Path, user_id: int) -> pd.DataFrame:
    return pd.read_parquet(data_path / "revlogs" / f"{user_id=}")


def save_user_tensors(txn: lmdb.Transaction, user_id: int, df: pd.DataFrame) -> None:
    ratings = torch.tensor(df["rating"].to_numpy(), dtype=torch.int8)
    elapsed_days = torch.tensor(df["elapsed_days"].to_numpy(), dtype=torch.int32)
    elapsed_time = torch.tensor(
        df["elapsed_seconds"].to_numpy() / SECONDS_PER_DAY,
        dtype=torch.float32,
    )

    save_tensor(txn, f"{user_id}_ratings", ratings)
    save_tensor(txn, f"{user_id}_elapsed_days", elapsed_days)
    save_tensor(txn, f"{user_id}_elapsed_time", elapsed_time)


def process_user(data_path: Path, user_id: int) -> int:
    if lmdb_env is None:
        raise RuntimeError("LMDB environment was not initialized.")

    done_key = f"{user_id}_done"
    with lmdb_env.begin(write=False) as txn:
        if txn.get(done_key.encode()) is not None:
            return user_id

    df = load_user_parquet(data_path, user_id)

    with lmdb_env.begin(write=True) as txn:
        save_user_tensors(txn, user_id, df)
        txn.put(done_key.encode(), b"true")

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
        initargs=(LMDB_PATH, LMDB_SIZE),
    )
    futures = []
    try:
        futures = [
            executor.submit(process_user, config.data_path, user_id)
            for user_id in user_ids
        ]

        for future in tqdm(as_completed(futures), total=len(futures), smoothing=0.03):
            future.result()
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown()


if __name__ == "__main__":
    main()
