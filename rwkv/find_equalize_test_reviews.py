import numpy as np
import psutil
from multiprocessing import Pool
import torch
from tqdm import tqdm
from config import Config, create_parser
from utils import get_bin
from features import create_features
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
import lmdb

from rwkv.parse_toml import parse_toml
from rwkv.utils import save_tensor

rwkv_config = parse_toml()
lmdb_env = None

parser = create_parser()
args, _ = parser.parse_known_args()
config = Config(args)
config.model_name = rwkv_config.ALGO
config.include_short_term = bool(rwkv_config.SHORT)
config.use_secs_intervals = bool(rwkv_config.SECS)


def _open_lmdb_env():
    max_readers = getattr(rwkv_config, "LMDB_MAX_READERS", None)
    if max_readers is None:
        processes = getattr(rwkv_config, "PROCESSES", 1)
        max_readers = max(128, processes * 8)
    return lmdb.open(
        rwkv_config.LABEL_FILTER_LMDB_PATH,
        map_size=rwkv_config.LABEL_FILTER_LMDB_SIZE,
        max_readers=max_readers,
    )


def process(user_id):
    global lmdb_env
    if lmdb_env is None:
        lmdb_env = _open_lmdb_env()

    key_review_ths = f"{user_id}_review_ths"
    key_rmse_bins = f"{user_id}_rmse_bins"
    key_split = f"{user_id}_split"
    with lmdb_env.begin(write=False) as txn:
        if txn.get(f"{user_id}_done".encode()) is not None:
            print(f"Already done for {user_id}.")
            return

    df = pd.read_parquet(rwkv_config.DATA_PATH / "revlogs" / f"{user_id=}")
    try:
        df = create_features(df.copy(), config=config)
    except ValueError as err:
        # Some users can lose every row during outlier/non-continuity filtering; skip those users.
        if "No data after handling outliers" in str(err):
            print(f"Skipping {user_id}: {err}")
            return
        raise
    if len(df) == 0:  # that one user
        return

    bins = df.apply(get_bin, axis=1)

    # map bins to contiguous integer indices
    bin_codes = bins.astype("category").cat.codes.to_numpy()

    # extract column once
    review_th_values = df["review_th"].to_numpy()

    tscv = TimeSeriesSplit(n_splits=5)

    test_label_review_th = []
    test_label_rmse_bins = []
    test_label_split_test_range = []
    for _, test_index in tscv.split(df):
        test_label_review_th.append(review_th_values[test_index])
        test_label_rmse_bins.append(bin_codes[test_index])
        test_label_split_test_range.append(min(review_th_values[test_index]))

    test_label_split_test_range.append(int(1e9))
    test_label_review_th = np.concatenate(test_label_review_th)
    test_label_rmse_bins = np.concatenate(test_label_rmse_bins)

    review_ths_tensor = torch.tensor(test_label_review_th, dtype=torch.int32)
    rmse_bins_tensor = torch.tensor(test_label_rmse_bins, dtype=torch.int32)
    split_tensor = torch.tensor(test_label_split_test_range, dtype=torch.int32)

    with lmdb_env.begin(write=True) as txn:
        save_tensor(txn, key_review_ths, review_ths_tensor)
        save_tensor(txn, key_rmse_bins, rmse_bins_tensor)
        save_tensor(txn, key_split, split_tensor)
        txn.put(f"{user_id}_done".encode(), "true".encode())

    print("Done:", user_id, "Size:", len(test_label_review_th))


def set_low_priority():
    try:
        p = psutil.Process()
        if hasattr(psutil, "IDLE_PRIORITY_CLASS"):
            p.nice(psutil.IDLE_PRIORITY_CLASS)
        else:
            # POSIX: nice level 19 is the lowest priority
            p.nice(19)
    except Exception as e:
        print(f"Failed to set priority: {e}")


def init_worker():
    global lmdb_env
    set_low_priority()
    lmdb_env = _open_lmdb_env()


def main():
    user_ids = list(range(rwkv_config.USER_START, rwkv_config.USER_END + 1))

    with Pool(processes=rwkv_config.PROCESSES, initializer=init_worker) as pool:
        _ = list(tqdm(pool.imap(process, user_ids), total=len(user_ids), smoothing=0.005))


if __name__ == "__main__":
    main()
