from __future__ import annotations

import importlib
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
    DATA_BUILD_DEVICE,
    LMDB_PATH,
    LMDB_SIZE,
    DEVICE,
    N_EPOCHS,
    N_SPLITS,
    TEST_BATCH_SIZE_MAX,
)
from parallel.load_balancer import get_batches_test, get_batches_train
from parallel.models import fsrs_v7_constants
from parallel.tensors import Data, DataBuilder, ParamKey, ReviewData, UserTensorBlob

enzyme_sample = importlib.import_module("parallel._enzyme_torch_sample")

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def load_blob_bytes(blob_bytes: bytes, map_location: str | torch.device = "cpu") -> UserTensorBlob:
    buffer = BytesIO(blob_bytes)
    tensors = torch.load(buffer, weights_only=True, map_location=map_location)
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

def load_user_blob(
    txn: lmdb.Transaction,
    user_id: int,
    map_location: str | torch.device = "cpu",
) -> UserTensorBlob:
    blob_bytes = txn.get(f"{user_id}_packed".encode())
    if blob_bytes is None:
        raise LookupError(f"Packed blob not found for user {user_id}.")
    return load_blob_bytes(blob_bytes, map_location=map_location)

# def next_power(n: int, base: int = 2) -> int:
#     assert n > 0
#     assert base >= 2

#     p = 1
#     while p < n:
#         p *= base
#     return p

# def _prepare_prediction_inputs(review_data: ReviewData, index):
#     ts = time.time()
#     seq_lens = review_data.seq_len[index]
#     max_seq_len = seq_lens.max()

#     card_start_index = index - seq_lens + 1
#     data_len = review_data.elapsed_days_real.size(0)

#     torch.cuda.synchronize()
#     to = time.time()
#     B = index.size(0)

#     take_indices = (
#         # card_start_index.unsqueeze(-1) + torch.minimum(torch.arange(max_seq_len.item(), device=index.device).unsqueeze(0).repeat(B, 1), seq_lens.unsqueeze(-1) - 1)
#         card_start_index.unsqueeze(-1) + torch.arange(max_seq_len.item(), device=index.device).unsqueeze(0)
#     ).clamp_max(data_len - 1)
#     print("prepare before take", time.time() - ts)
#     ts = time.time()

#     rating_bl = review_data.rating[take_indices]
#     elapsed_time_real_bl = review_data.elapsed_days_real[take_indices]
#     print("prepare take", time.time() - ts)

#     # print("shape", rating_bl.shape, max_seq_len)
#     # print(review_data.elapsed_days_real[index])
#     # print(elapsed_time_real_bl[:, seq_lens - 1], elapsed_time_real_bl[:, seq_lens - 1].shape)
#     # assert (review_data.elapsed_days_real[index] == elapsed_time_real_bl[torch.arange(index.size(0)), seq_lens - 1]).all()
#     # assert (review_data.elapsed_days_real[index] > 0).all()
#     B = elapsed_time_real_bl.size(0)
#     return elapsed_time_real_bl, rating_bl, seq_lens

# def predict(review_data: ReviewData, index, params):
#     assert index.size(0) == index.size(0)
#     elapsed_time_real_bl, rating_bl, seq_lens = _prepare_prediction_inputs(
#         review_data, index
#     )
#     return fsrs_v7_jax_adapter.prediction(params, elapsed_time_real_bl, rating_bl, seq_lens)


# def predict_loss_grad(review_data: ReviewData, index, params, epoch_lens):
#     torch.cuda.synchronize()
#     ts = time.time()
#     assert index.size(0) == params.size(0)
#     elapsed_time_real_bl, rating_bl, seq_lens = _prepare_prediction_inputs(
#         review_data, index
#     )
#     torch.cuda.synchronize()
#     print("pred loss grad prep", time.time() - ts)
#     return None, None, None
#     # return fsrs_v7_jax_adapter.prediction_loss_grad(params, elapsed_time_real_bl, rating_bl, seq_lens, epoch_lens) 

def train(fsrs_params: torch.Tensor, users: list[int], data: Data):
    print("start train")
    print(data.train_split_lengths)
    train_split_lengths_cat = data.train_split_lengths
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
    # exit()
    step_i_cat = torch.zeros_like(num_training_steps_cat)
    optim_state = adamw.init_adamw_state(fsrs_params.view(-1, fsrs_params.size(-1)))
    for iter in tqdm(range(train_splits_length_cat_max), desc="Training", smoothing=0.03):
        torch.cuda.synchronize()
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
        print("TODO Try reverse sort")
        sorted_seq_lens, sorted_seq_lens_indices = torch.sort(seq_lens, stable=True)
        seq_lens = sorted_seq_lens
        indices_filtered = indices_filtered[sorted_seq_lens_indices]
        review_data_indices_filtered = review_data_indices_filtered[sorted_seq_lens_indices]

        torch.cuda.synchronize()
        tb = time.time()
        batches = get_batches_train(seq_lens)
        print("get batches", time.time() - tb)
        torch.cuda.synchronize()
        print("prepare iteration", time.time() - ts)
        # print(batches)
        for (l, re) in batches:
            ts = time.time()
            # slice = torch.arange(l, re, re - l, device=indices.device)
            # batch_fsrs_params = torch.tensor(fsrs_params.view(-1, fsrs_params.size(-1))[indices_filtered[l:re]], requires_grad=True, device=indices.device)
            # batch_fsrs_params = fsrs_params.view(-1, fsrs_params.size(-1))[indices_filtered[l:re]].detach().clone().requires_grad_(True)
            batch_indices = indices_filtered[l:re]
            batch_fsrs_params = fsrs_params.view(-1, fsrs_params.size(-1))[batch_indices]
            batch_epoch_lens = train_split_lengths_cat[batch_indices]
            assert (batch_epoch_lens > 0).all()
            # batch_review_data_indices = review_data_indices_filtered[slice]
            batch_review_data_indices = review_data_indices_filtered[l:re]
            torch.cuda.synchronize()
            print("intermediate", time.time() - ts)
            p, loss, grad = predict_loss_grad(data.review_data, batch_review_data_indices, batch_fsrs_params, batch_epoch_lens)
            # assert not p.isnan().any()
            # torch.cuda.synchronize()
            # print("intermediate2", time.time() - ts)
            # # print("Jax adapter output:")
            # # print(p)

            # # batch_fsrs_params.grad = grad
            # # print(batch_fsrs_params)
            # print(batch_epoch_lens)
            # batch_fsrs_params.backward(grad)

            # print(batch_fsrs_params.shape)
            # torch.cuda.synchronize()
            # tl2 = time.time()
            # l2_grad = fsrs_v7_constants.l2_penalty_per_review(batch_fsrs_params, batch_epoch_lens)
            # print(l2_grad)
            # l2_grad.sum().backward()
            # torch.cuda.synchronize()
            # print("L2 time", time.time() - tl2)

            # print(fsrs_params.grad)
            # exit()
            torch.cuda.synchronize()
            print("total it", time.time() - ts)
            print()

        t_opt = time.time()
        active_params_mask = torch.zeros(step_i_cat.size(0), device=step_i_cat.device, dtype=torch.bool)
        active_params_mask[indices] = True
        lr_schedule_multi = 2e-2 * scheduler.scheduler(step_i_cat, num_training_steps_cat)

        # TODO ADAMW betas

        # with torch.no_grad():
        #     if fsrs_params.grad.isnan().any():
        #         print(fsrs_params.grad.cpu().detach().tolist())
        #     assert not fsrs_params.grad.isnan().any()
        #     flat_fsrs_params = fsrs_params.view(-1, fsrs_params.size(-1))
        #     flat_grad = fsrs_params.grad.view_as(flat_fsrs_params)
        #     new_flat_fsrs_params, optim_state = adamw.adamw_step(
        #         flat_fsrs_params,
        #         flat_grad,
        #         optim_state,
        #         lr=lr_schedule_multi,
        #         mask=active_params_mask,
        #     )
        #     new_flat_fsrs_params_clipped = fsrs_v7_constants.apply_parameter_clipper(new_flat_fsrs_params)
        #     flat_fsrs_params.copy_(new_flat_fsrs_params_clipped)
        #     assert not flat_fsrs_params.isnan().any()

        fsrs_params.grad = None
        step_i_cat[indices] += 1

        torch.cuda.synchronize()
        print("opt time", time.time() - t_opt)
        print("----------------------------------------------------------")


    assert (step_i_cat == num_training_steps_cat).all()
    print("------------------done train-----------------")
    return fsrs_params

def evaluate_on_test_set(fsrs_params: torch.Tensor, users: list[int], data: Data):
    print("eval on test")
    torch.cuda.synchronize()
    tz = time.time()
    ts = time.time()
    param_keys = data.get_test_index_param_key()

    test_seq_len = data.review_data.seq_len[data.test_index]
    sorted_test_seq_len, sorted_test_index_permutation = torch.sort(test_seq_len, stable=True, descending=True)
    torch.cuda.synchronize()
    print("sort done", time.time() - ts)

    # TODO use the same load balancing as training
    N = data.test_index.size(0)
    num_batches = ceil_div(N, TEST_BATCH_SIZE_MAX)
    batch_size = ceil_div(N, num_batches)
    gather_p = []
    # batches = get_batches_test(sorted_test_seq_len)
    # print(batches)

    # enzyme_sample.fsrs7_forward(
    #     data.review_data.elapsed_days_real, 
    #     data.review_data.rating, 
    #     data.review_data.seq_len, 
    #     sorted_test_index_permutation,
    # )
    # exit()
    # for (l, re) in tqdm(batches, desc="Test set", smoothing=0.03):
    for perm_slice in tqdm(sorted_test_index_permutation.split(batch_size), desc="Test set", smoothing=0.03):
        # perm_slice = sorted_test_index_permutation[l:re]
        batch_fsrs_params = fsrs_params[param_keys.user_index[perm_slice], param_keys.split_index[perm_slice]]
        # p = predict(data.review_data, data.test_index[perm_slice], batch_fsrs_params)
        test_index_perm_slice = data.test_index[perm_slice]
        seq_lens = data.review_data.seq_len[test_index_perm_slice]
        start_indices = test_index_perm_slice - seq_lens + 1
        # print(seq_lens.size(0))
        p = enzyme_sample.fsrs7_forward(
                data.review_data.elapsed_days_real, 
                data.review_data.rating, 
                start_indices,
                seq_lens,
                batch_fsrs_params,
            )
        gather_p.append(p)

    torch.cuda.synchronize()
    print("took", time.time() - tz)
    
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
        data_build_device = torch.device(DATA_BUILD_DEVICE)
        data_builder = DataBuilder(device=data_build_device)
        for user_id in tqdm(users, total=len(users), smoothing=0.03, desc="Loading user data"):
            blob = load_user_blob(txn, user_id, map_location=data_build_device)
            data_builder.append(blob)
            del blob
        data = data_builder.finish(device=DEVICE)
        del data_builder

    # fsrs_params = torch.zeros((len(users), N_SPLITS, 35), device=DEVICE)
    initial_params = fsrs_v7_constants.get_initial_params_for_optimization().to(DEVICE)
    fsrs_params = initial_params.view(1, 1, -1).repeat(len(users), N_SPLITS, 1)
    # fsrs_params = train(fsrs_params, users, data)

    print("skip test")
    # evaluate
    with torch.no_grad():
        evaluate_on_test_set(fsrs_params, users, data)

def main() -> None:
    env = lmdb.open(
        str(LMDB_PATH),
        map_size=LMDB_SIZE,
        readonly=True,
        lock=False,
    )
    
    users = list(range(1, 10))
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
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"
    torch.manual_seed(123)
    main()
