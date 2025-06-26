
from argparse import Namespace
from dataclasses import dataclass
import logging
import multiprocessing
import lmdb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import torch
from tqdm import tqdm
from features import create_features
from models.model_factory import create_model
from utils import load_tensor, parse_toml, save_tensor
try:
    from fsrs_optimizer import BatchDataset, BatchLoader, plot_brier, Optimizer  # type: ignore
except Exception as e:
    logging.exception("Failed to import fsrs_optimizer: %s", e)

class MinimalBatchLoader:
    def __init__(self, batch_nums, shuffle: bool = True, seed: int = 2023):
        self.batch_nums = batch_nums
        self.shuffle = shuffle
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            yield from (
                idx
                for idx in torch.randperm(
                    self.batch_nums, generator=self.generator
                ).tolist()
            )
        else:
            yield from (self.dataset[idx] for idx in range(self.batch_nums))

    def __len__(self):
        return self.batch_nums

@dataclass
class SplitInfo:
    pretrain_params: torch.Tensor
    epochs: torch.Tensor
    review_ths_all: list
    batch_lens: torch.Tensor
    lrs: torch.Tensor
    train_set_review_ths: torch.Tensor
    test_set_review_ths: torch.Tensor

def get_fsrs_training_info(df) -> SplitInfo:
    fsrs_config = {
        "use_secs_intervals": False,
        "equalize_test_with_non_secs": False,
        "model_name": "FSRS-6",
        "two_buttons": False,
        "max_seq_len": 64,
        "include_short_term": True,
        "n_splits": 5,
        "batch_size": 512,
        "n_epoch": 5,
        "device": torch.device("cpu"),
        "s_min": 0.001,
        "init_s_max": 100.0,
        "s_max": 36500.0,
        "verbose": False,
        "verbose_inadequate_data": False,
    }
    fsrs_config = Namespace(**fsrs_config)
    df = create_features(df, config=fsrs_config)
    tscv = TimeSeriesSplit(n_splits=fsrs_config.n_splits)

    split_infos = []
    for split_i, (train_index, test_index) in enumerate(tscv.split(df)):
        train_set = df.iloc[train_index]
        test_set = df.iloc[test_index]

        # Get and store the initial FSRS stability values
        model = create_model(fsrs_config)
        model.pretrain(train_set)
        pretrain_params = torch.tensor(model.state_dict(), dtype=torch.float)[:4]

        # Get the training batches
        generator = torch.Generator()
        generator.manual_seed(2023)

        batch_num, remainder = divmod(len(train_set), max(1, fsrs_config.batch_size))
        batch_num = batch_num + 1 if remainder > 0 else batch_num
        data_loader = MinimalBatchLoader(batch_num)
        # random tensor to setup a scheduler
        optim_tensor = torch.tensor([1.0], requires_grad=True)
        optim = torch.optim.AdamW([optim_tensor], lr=4e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=batch_num * fsrs_config.n_epoch)

        epochs = []
        review_ths_all = []
        batch_lens = []
        lrs = []
        for epoch in range(fsrs_config.n_epoch):
            for batch_i, batch_bin in enumerate(data_loader):
                lrs.append(scheduler.get_last_lr()[0])
                batch_l = batch_bin * fsrs_config.batch_size
                batch_r = min(len(train_set), (batch_bin + 1) * fsrs_config.batch_size)
                df_slice = train_set.iloc[batch_l:batch_r]
                review_ths = df_slice["review_th"]
                epochs.append(epoch)
                review_ths_all.append(review_ths.to_numpy())
                batch_lens.append(len(review_ths))
                scheduler.step()

        split_infos.append(SplitInfo(
            pretrain_params=pretrain_params,
            epochs=torch.tensor(epochs, dtype=torch.int),
            review_ths_all=review_ths_all,
            batch_lens=torch.tensor(batch_lens, dtype=torch.int),
            lrs=torch.tensor(lrs, dtype=torch.float),
            train_set_review_ths=torch.tensor(train_set["review_th"].to_numpy(), dtype=torch.int),
            test_set_review_ths=torch.tensor(test_set["review_th"].to_numpy(), dtype=torch.int),
        ))
    
    return split_infos

@dataclass
class PackedSplitInfo:
    pretrain_params: torch.Tensor
    epochs: torch.Tensor
    locs: torch.Tensor
    locs_lens: torch.Tensor
    keys: torch.Tensor
    keys_lens: torch.Tensor
    train_set_locs: torch.Tensor
    train_set_keys: torch.Tensor
    test_set_locs: torch.Tensor
    test_set_keys: torch.Tensor

def process(user_id, config):
    df = pd.read_parquet(
        config.DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    split_infos = get_fsrs_training_info(df.copy())

    df["review_th"] = range(1, df.shape[0] + 1)
    len_before = len(df)
    df.drop(df[~df["rating"].isin([1, 2, 3, 4])].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    assert len_before == len(df), f"{user_id} has invalid ratings, review_th might be incorrect"

    # equalize_env = lmdb.open(config.LABEL_FILTER_DB_PATH, readonly=True, lock=False)
    # with equalize_env.begin(write=False) as txn:
    #     equalize_review_ths = load_tensor(txn, f"{user_id}_review_ths", device="cpu")
    #     # rmse_bins = load_tensor(txn, f"{user_id}_rmse_bins", device="cpu")
    # equalize_review_ths_set = set(equalize_review_ths.tolist())

    # df["is_equalize_review"] = df["review_th"].isin(equalize_review_ths_set).astype(int)
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
    perm = np.array(perm)

    perm_inv = [-1 for _ in range(len(perm))]
    for i in range(len(perm)):
        perm_inv[perm[i]] = i
    perm_inv = np.array(perm_inv)

    def review_ths_to_packed_query(review_ths):
        rows = df.iloc[review_ths - 1].sort_values('card_id')
        rows["start_loc"] = rows["card_id"].map(card_locs_dict)
        rows["loc"] = perm_inv[rows["review_th"] - 1]
        rows["req_L"] = rows["loc"] - rows["start_loc"]
        rows["batch_loc"] = range(0, len(rows))

        start_locs = rows.groupby("card_id", sort=False)["start_loc"].min().to_numpy()
        ls = rows.groupby("card_id", sort=False)["batch_loc"].min().to_numpy()
        rs = rows.groupby("card_id", sort=False)["batch_loc"].max().to_numpy()
        Ls = rows.groupby("card_id", sort=False)["req_L"].max().to_numpy()
        locs = torch.tensor(rows["loc"].to_list(), dtype=torch.int)
        keys = torch.stack(tuple(map(lambda x: torch.tensor(x, dtype=torch.int), (start_locs, ls, rs, Ls))), dim=-1)
        return locs, keys

    packed_split_infos = []
    with torch.no_grad():
        for split_info in split_infos:
            locs_all = []
            keys_all = []
            locs_lens = []
            keys_lens = []
            for review_ths in split_info.review_ths_all:
                locs, keys = review_ths_to_packed_query(review_ths)
                locs_all.append(locs)
                keys_all.append(keys)
                locs_lens.append(locs.size(0))
                keys_lens.append(keys.size(0))

            train_set_locs, train_set_keys = review_ths_to_packed_query(split_info.train_set_review_ths)
            test_set_locs, test_set_keys = review_ths_to_packed_query(split_info.test_set_review_ths)

            packed_split_infos.append(PackedSplitInfo(
                pretrain_params=split_info.pretrain_params,
                epochs=split_info.epochs,
                locs=torch.cat(locs_all),
                locs_lens=torch.tensor(locs_lens, dtype=torch.int),
                keys=torch.cat(keys_all),
                keys_lens=torch.tensor(keys_lens, dtype=torch.int),
                train_set_locs=train_set_locs,
                train_set_keys=train_set_keys,
                test_set_locs=test_set_locs,
                test_set_keys=test_set_keys,
            ))

        packed_review_th_T = torch.tensor(df["review_th"], dtype=torch.int)[perm]
        packed_rating_T = torch.tensor(df["rating"], dtype=torch.int)[perm]
        packed_elapsed_days_real_T = torch.tensor(df["elapsed_days_real"], dtype=config.DTYPE)[perm]
        packed_elapsed_days_int_T = torch.tensor(df["elapsed_days_int"], dtype=config.DTYPE)[perm]
        packed_label_elapsed_days_real_T = torch.tensor(df["label_elapsed_days_real"], dtype=config.DTYPE)[perm]
        packed_label_elapsed_days_int_T = torch.tensor(df["label_elapsed_days_int"], dtype=config.DTYPE)[perm]

    return packed_split_infos, packed_review_th_T, packed_rating_T, packed_elapsed_days_real_T, packed_elapsed_days_int_T, packed_label_elapsed_days_real_T, packed_label_elapsed_days_int_T

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
                packed_split_infos,
                packed_review_th_T,
                packed_rating_T,
                packed_elapsed_days_real_T,
                packed_elapsed_days_int_T,
                packed_label_elapsed_days_real_T,
                packed_label_elapsed_days_int_T,
            ) = tensors
            assert len(packed_split_infos) == 5
            for split_i, packed_split_info in enumerate(packed_split_infos):
                save_tensor(txn, f"{user_id}_split_{split_i}_pretrain_params", packed_split_info.pretrain_params)   
                save_tensor(txn, f"{user_id}_split_{split_i}_epochs", packed_split_info.epochs)   
                save_tensor(txn, f"{user_id}_split_{split_i}_locs", packed_split_info.locs)   
                save_tensor(txn, f"{user_id}_split_{split_i}_locs_lens", packed_split_info.locs_lens)   
                save_tensor(txn, f"{user_id}_split_{split_i}_keys", packed_split_info.keys)   
                save_tensor(txn, f"{user_id}_split_{split_i}_keys_lens", packed_split_info.keys_lens)   
                save_tensor(txn, f"{user_id}_split_{split_i}_train_set_locs", packed_split_info.train_set_locs)   
                save_tensor(txn, f"{user_id}_split_{split_i}_train_set_keys", packed_split_info.train_set_keys)   
                save_tensor(txn, f"{user_id}_split_{split_i}_test_set_locs", packed_split_info.test_set_locs)   
                save_tensor(txn, f"{user_id}_split_{split_i}_test_set_keys", packed_split_info.test_set_keys)   
            save_tensor(txn, f"{user_id}_packed_review_th_T", packed_review_th_T)
            save_tensor(txn, f"{user_id}_packed_rating_T", packed_rating_T)
            save_tensor(txn, f"{user_id}_packed_elapsed_days_real_T", packed_elapsed_days_real_T)
            save_tensor(txn, f"{user_id}_packed_elapsed_days_int_T", packed_elapsed_days_int_T)
            save_tensor(txn, f"{user_id}_packed_label_elapsed_days_real_T", packed_label_elapsed_days_real_T)
            save_tensor(txn, f"{user_id}_packed_label_elapsed_days_int_T", packed_label_elapsed_days_int_T)
            txn.put(f"{user_id}_done".encode(), "true".encode())

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
    # process(1, config)
    main(config)