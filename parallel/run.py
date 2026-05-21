from __future__ import annotations

from io import BytesIO

import lmdb
import numpy as np
from sklearn.metrics import log_loss
import torch
from tqdm import tqdm
import time

from parallel import adamw, scheduler
from parallel.config import (
    BATCH_SIZE,
    LMDB_PATH,
    LMDB_SIZE,
    DEVICE,
    N_EPOCHS,
    N_SPLITS,
    TEST_BATCH_SIZE_MAX,
)
from parallel.models import fsrs_v7
from parallel.randperm import segmented_feistel_permutation
from parallel.tensors import Data, ParamKey, ReviewData, UserTensorBlob

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def load_blob_bytes(blob_bytes: bytes) -> UserTensorBlob:
    buffer = BytesIO(blob_bytes)
    tensors = torch.load(buffer, weights_only=True, map_location=DEVICE)
    return UserTensorBlob.from_dict(tensors)

def load_metadata_tensor(txn: lmdb.Transaction, key: str) -> torch.Tensor:
    tensor_bytes = txn.get(key.encode())
    if tensor_bytes is None:
        raise KeyError(f"Missing LMDB metadata key: {key}")
    return torch.load(
        BytesIO(tensor_bytes),
        weights_only=True,
        map_location=DEVICE,
    )


def load_user_blob(txn: lmdb.Transaction, user_id: int) -> UserTensorBlob:
    blob_bytes = txn.get(f"{user_id}_packed".encode())
    if blob_bytes is None:
        raise LookupError(f"Packed blob not found for user {user_id}.")
    return load_blob_bytes(blob_bytes)

def next_power(n: int, base: int = 2) -> int:
    assert n > 0
    assert base >= 2

    p = 1
    while p < n:
        p *= base
    return p

def predict(review_data: ReviewData, index, params):
    assert index.size(0) == params.size(0)
    seq_lens = review_data.seq_len[index]
    max_seq_len = seq_lens.max()
    card_start_index = index - seq_lens + 1
    data_len = review_data.elapsed_days_real.size(0)
    take_indices = (card_start_index.unsqueeze(-1) + torch.arange(max_seq_len.item(), device=index.device).repeat(index.size(0), 1)).clamp_max(data_len - 1)

    rating_bl = review_data.rating[take_indices]
    elapsed_time_real_bl = review_data.elapsed_days_real[take_indices]

    base = 3  # choose 2, 3, 4, etc.
    L = rating_bl.size(-1)
    # L_pad = next_power(L, base=base)
    L_pad = 65
    pad = L_pad - L

    if pad > 0:
        rating_bl = torch.cat(
            [rating_bl, rating_bl.new_full((*rating_bl.shape[:-1], pad), 0)],
            dim=-1,
        )

        elapsed_time_real_bl = torch.cat(
            [elapsed_time_real_bl, elapsed_time_real_bl.new_full((*elapsed_time_real_bl.shape[:-1], pad), 0.0)],
            dim=-1,
        )

    fsrs_params = fsrs_v7.nn_vec_to_fsrs7_params(params)
    print("shape", rating_bl.shape)
    return fsrs_v7.forward(fsrs_params, elapsed_time_real_bl, rating_bl, seq_lens)
    

def train(fsrs_params: torch.Tensor, users: list[int], data: Data):
    print("start train")
    print(data.train_split_lengths)
    train_split_lengths_cat = torch.cat(data.train_split_lengths)
    num_training_steps_per_epoch_cat = (train_split_lengths_cat + BATCH_SIZE - 1) // BATCH_SIZE
    num_training_steps_cat = N_EPOCHS * num_training_steps_per_epoch_cat
    print(num_training_steps_per_epoch_cat)
    
    batch_perm_cat = torch.cat([torch.randperm(n) for n in num_training_steps_per_epoch_cat.cpu().numpy() for _ in range(N_EPOCHS)]).to(DEVICE)
    # Map: (user, split) -> batch_perm_cat start index
    batch_perm_user_flat_offset = torch.nn.functional.pad(
        torch.cumsum(num_training_steps_cat, dim=-1)[:-1],
        (1, 0),
    ).view(len(users), N_SPLITS)
    # print(batch_perm_user_flat_offset)
    # print(batch_perm_cat, batch_perm_cat.size())
    # print(batch_perm_cat.cpu().tolist())
    # exit()

    # Map: (user, split) -> data.train_index start index
    train_split_lengths_offset = torch.nn.functional.pad(
        torch.cumsum(train_split_lengths_cat, dim=-1)[:-1],
        (1, 0),
    ).view(len(users), N_SPLITS)

    print(batch_perm_user_flat_offset)
    print(batch_perm_cat, len(batch_perm_cat))
    train_splits_length_cat_sum = int(num_training_steps_cat.sum().item())
    train_splits_length_cat_max = int(num_training_steps_cat.max().item())
    batch_num_inner_batches = ceil_div(train_splits_length_cat_sum, train_splits_length_cat_max)
    
    print(num_training_steps_cat)
    print("Inner batches per batch:", batch_num_inner_batches, "Train iters (max):", train_splits_length_cat_max)
    step_i_cat = torch.zeros_like(num_training_steps_cat)
    optim_state = adamw.init_adamw_state(fsrs_params)
    for iter in tqdm(range(train_splits_length_cat_max), desc="Training", smoothing=0.03):
        ts = time.time()
        remaining = num_training_steps_cat - step_i_cat
        # print("remaining:", remaining)
        _, indices = torch.topk(remaining, k=int(min(batch_num_inner_batches, (remaining > 0).sum().item())))
        step_i = step_i_cat[indices]
        # epoch_step_i = step_i % num_training_steps_per_epoch_cat[indices]
        # print(step_i, epoch_step_i)

        user_indices = indices // N_SPLITS
        split_indices = indices % N_SPLITS
        perm_i = batch_perm_cat[batch_perm_user_flat_offset[user_indices, split_indices] + step_i]
        assert (perm_i <= num_training_steps_per_epoch_cat[indices]).all()

        train_l = perm_i * BATCH_SIZE
        train_r = torch.minimum((perm_i + 1) * BATCH_SIZE - 1, train_split_lengths_cat[indices] - 1)
        assert (train_l <= train_r).all()

        indices_range = indices.unsqueeze(-1).repeat(1, BATCH_SIZE)
        train_range = \
            + train_l.unsqueeze(-1) \
            + torch.arange(BATCH_SIZE, device=train_l.device).view(1, -1).repeat(train_l.size(0), 1)

        legal = train_range <= train_r.unsqueeze(-1)
        review_data_indices = data.train_index[(train_split_lengths_offset[user_indices, split_indices].unsqueeze(-1) + train_range).clamp_max(data.train_index.size(0) - 1)]
        # print(review_data_indices, review_data_indices.shape)
        # print(legal)
        # print(train_split_lengths_cat[indices] - 1)

        indices_flat = indices_range.view(-1)
        review_data_indices_flat = review_data_indices.view(-1)
        train_range_flat = train_range.view(-1)
        legal_flat = legal.view(-1)
        assert legal_flat.shape == indices_flat.shape

        indices_filtered = indices_flat[legal_flat]
        # train_range_filtered = train_range_flat[legal_flat]
        review_data_indices_filtered = review_data_indices_flat[legal_flat]
        # review_data_indices_filtered = data.train_index[train_split_lengths_offset[indices_filtered // N_SPLITS, indices_filtered % N_SPLITS].unsqueeze(-1) + train_range_filtered]
        # print(review_data_indices_filtered.shape)
        # exit()

        # Sort by seq len
        seq_lens = data.review_data.seq_len[review_data_indices_filtered]
        sorted_seq_lens, sorted_seq_lens_indices = torch.sort(seq_lens, stable=True)
        seq_lens = sorted_seq_lens
        indices_filtered = indices_filtered[sorted_seq_lens_indices]
        review_data_indices_filtered = review_data_indices_filtered[sorted_seq_lens_indices]
        print(time.time() - ts)

        def print_power2_bucket_counts(x: torch.Tensor):
            # buckets: [0], [1], [2,3], [4,7], ..., [32,63], [64]
            bounds = torch.tensor([0, 1, 2, 4, 8, 16, 32, 64, 65], device=x.device)

            # x is sorted, so searchsorted is efficient
            starts = torch.searchsorted(x, bounds[:-1], right=False)
            ends   = torch.searchsorted(x, bounds[1:],  right=False)
            counts = ends - starts

            for lo, hi, c in zip(bounds[:-1].tolist(), bounds[1:].tolist(), counts.tolist()):
                if hi == lo + 1:
                    print(f"{lo}: {c}")
                else:
                    print(f"{lo}-{hi - 1}: {c}")
        
        print(print_power2_bucket_counts(seq_lens))

        def load_balancer():
            b = indices_filtered.size(0)
            # return [(0, b // 2), (b // 2, b)]
            return [(0, b)]
        # Run
        batches = load_balancer()
        for (l, re) in batches:
        #    slice = torch.arange(l, re, re - l, device=indices.device)
            # batch_fsrs_params = torch.tensor(fsrs_params.view(-1, fsrs_params.size(-1))[indices_filtered[l:re]], requires_grad=True, device=indices.device)
            batch_fsrs_params = fsrs_params.view(-1, fsrs_params.size(-1))[indices_filtered[l:re]]
            batch_review_data_indices = review_data_indices_filtered[l:re]
            ts = time.time()
            p = predict(data.review_data, batch_review_data_indices, batch_fsrs_params)
            print(time.time() - ts)
            label = data.review_data.rating[batch_review_data_indices] > 1
            loss = torch.nn.functional.binary_cross_entropy(p, label.float(), reduction='sum')
            loss.backward()


        active_params_mask = torch.zeros(step_i_cat.size(0), device=step_i_cat.device, dtype=torch.bool)
        active_params_mask[indices] = True
        lr_schedule_multi = scheduler.scheduler(step_i_cat, num_training_steps_cat)

        fsrs_params.grad = None
        step_i_cat[indices] += 1


    assert (step_i_cat == num_training_steps_cat).all()
    print("------------------done train-----------------")
    return fsrs_params

def evaluate_on_test_set(fsrs_params: torch.Tensor, users: list[int], data: Data):
    print("eval on test")
    param_keys = data.get_test_index_param_key()

    test_seq_len = data.review_data.seq_len[data.test_index]
    sorted_test_index_permutation = torch.argsort(test_seq_len, stable=True)
    print("sort done")

    # TODO use the same load balancing as training
    N = data.test_index.size(0)
    num_batches = ceil_div(N, TEST_BATCH_SIZE_MAX)
    batch_size = ceil_div(N, num_batches)
    gather_p = []
    for perm_slice in sorted_test_index_permutation.split(batch_size):
        batch_fsrs_params = fsrs_params[param_keys.user_index[perm_slice], param_keys.split_index[perm_slice]]
        p = predict(data.review_data, data.test_index[perm_slice], batch_fsrs_params)
        gather_p.append(p)
    
    restore_test_index_permutation = torch.empty_like(sorted_test_index_permutation)
    restore_test_index_permutation[sorted_test_index_permutation] = torch.arange(
        sorted_test_index_permutation.numel(),
        device=sorted_test_index_permutation.device,
    )
    p_concat = torch.cat(gather_p, dim=-1)[restore_test_index_permutation]
    del sorted_test_index_permutation
    del restore_test_index_permutation

    p_by_user = p_concat.split(data.test_index_lens)
    label_by_user = (data.review_data.rating[data.test_index] > 1).split(data.test_index_lens)

    for user, pred, label in tqdm(zip(users, p_by_user, label_by_user), smoothing=0.03):
        logloss = log_loss(y_true=label.cpu().numpy(), y_pred=pred.cpu().numpy(), labels=[0, 1])
        print(f"User: {user}, logloss={logloss:.3f}")

def run(
    env: lmdb.Environment,
    users: list[int],
) -> None:
    with env.begin(write=False) as txn:
        blobs = [
            load_user_blob(txn, user_id)
            for user_id in tqdm(users, total=len(users), smoothing=0.03, desc="Loading user data")
        ]
        data = Data(blobs)
        del blobs

    # TODO train


    # fsrs_params = torch.zeros((len(users), N_SPLITS, 35), device=DEVICE)
    fsrs_params = torch.tensor(fsrs_v7.get_initial_params_for_optimization().to(DEVICE).view(1, 1, -1).repeat(len(users), N_SPLITS, 1), requires_grad=True)
    fsrs_params = train(fsrs_params, users, data)

    print("skip test")
    # # evaluate
    # with torch.no_grad():
    #     evaluate_on_test_set(fsrs_params, users, data)

def main() -> None:
    env = lmdb.open(
        str(LMDB_PATH),
        map_size=LMDB_SIZE,
        readonly=True,
        lock=False,
    )
    
    users = list(range(1, 3000))
    # users = [1, 2]
    # TODO get length metadata, sort by users, run

    run(env, users)

    # with env.begin(write=False) as txn:
    #     user_max_train_split_lengths = load_metadata_tensor(
    #         txn,
    #         USER_MAX_TRAIN_SPLIT_LENGTHS_KEY,
    #     )
    # env.close()

    # values, indices = torch.sort(user_max_train_split_lengths, descending=True)

    # for index, value in zip(indices.tolist(), values.tolist()):
    #     user_id = USER_IDS[index] if index < len(USER_IDS) else index
    #     print(f"{index}\t{user_id}\t{value}")


if __name__ == "__main__":
    main()
