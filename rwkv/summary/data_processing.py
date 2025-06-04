
import lmdb
import numpy as np
import pandas as pd
import torch
from rwkv.summary.summary_model import ToFSRSParams
from rwkv.utils import get_number_of_trainable_parameters
from utils import parse_toml


def process(user_id, config):
    df = pd.read_parquet(
        config.DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    df["review_th"] = range(1, df.shape[0] + 1)
    len_before = len(df)
    df.drop(df[~df["rating"].isin([1, 2, 3, 4])].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    df["elapsed_seconds"] = df["elapsed_seconds"].map(lambda x: max(0, x))
    df["elapsed_days_int"] = df["elapsed_days"].map(lambda x: max(0, x))
    assert len_before == len(df), f"{user_id} has invalid ratings, review_th might be incorrect"

    card_locs = df.groupby("card_id")["review_th"].apply(list).to_dict()
    ordered_card_ids = sorted(card_locs, key=lambda k: len(card_locs[k]), reverse=True)

    perm = []
    indices = []
    for card_id in ordered_card_ids:
        indices.append(len(perm))
        for review_th in card_locs[card_id]:
            perm.append(review_th - 1)

    perm_inv = [-1 for _ in range(len(perm))]
    for i in range(len(perm)):
        perm_inv[perm[i]] = i

    rating_onehot_T4 = torch.nn.functional.one_hot(torch.tensor(df["rating"], dtype=torch.long) - 1, num_classes=4).to(config.DTYPE)
    scaled_seconds_T1 = ((torch.log(torch.tensor(df["elapsed_seconds"], dtype=config.DTYPE) + 1) - 10) / 5).unsqueeze(-1)
    scaled_elapsed_days_T1 = ((torch.log(torch.tensor(df["elapsed_days_int"], dtype=config.DTYPE) + 1) - 1.5) / 1.5).unsqueeze(-1)
    scaled_state_T1 = (torch.tensor(df["state"], dtype=config.DTYPE) - 2).unsqueeze(-1)
    scaled_duration_T1 = (torch.log(10 + torch.tensor(df["duration"], dtype=config.DTYPE)) - 9).unsqueeze(-1)
    df["day_offset_diff"] = df["day_offset"].diff().fillna(0)
    scaled_day_offset_diff_T1 = torch.log(torch.log(np.e + torch.tensor(df["day_offset_diff"], dtype=config.DTYPE))).unsqueeze(-1)
    # features_T9 = torch.cat((rating_onehot_T4, scaled_seconds_T1, scaled_elapsed_days_T1, scaled_duration_T1, scaled_state_T1, scaled_day_offset_diff_T1), dim=-1)

    perm_tensor = torch.tensor(perm, dtype=torch.int)
    perm_inv_tensor = torch.tensor(perm_inv, dtype=torch.int)
    indices_I = torch.tensor(indices, dtype=torch.int)

    return rating_onehot_T4, scaled_seconds_T1, scaled_elapsed_days_T1, scaled_state_T1, scaled_duration_T1, scaled_day_offset_diff_T1, perm_tensor, perm_inv_tensor, indices_I

def job(user_id, config, writer_queue, progress_queue):
    writer_queue.put((user_id, process(user_id, config)))
    progress_queue.put(1)

if __name__ == '__main__':
    env = lmdb.open("fsrs_evaluate_db", readonly=True, lock=False)

    with env.begin() as txn:
        info = env.info()
        print(f"Map size (bytes): {info['map_size']}")
        print(f"Map size (MB): {info['map_size'] / (1024 * 1024):.2f} MB")
    exit()

    dtype = torch.bfloat16
    fsrs = ToFSRSParams(in_dim=32)
    fsrs = fsrs.selective_cast(dtype)
    for _ in range(10):
        x = torch.randn((1, 32), dtype=dtype)
        y = fsrs(x)
        print(y.tolist())
    # config = parse_toml()
    # main(1, config)