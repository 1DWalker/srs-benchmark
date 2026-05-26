from __future__ import annotations

from io import BytesIO

import lmdb
import numpy as np
from sklearn.metrics import log_loss
import torch
from tqdm import tqdm
import time

from parallel import adamw, scheduler, srs_ops
from parallel.config import (
    BATCH_SIZE,
    DATA_BUILD_DEVICE,
    LMDB_PATH,
    LMDB_SIZE,
    DEVICE,
    N_EPOCHS,
    N_SPLITS,
    TEST_BATCH_SIZE_MAX,
    TRAIN_BUFFER_SIZE_GB,
)
from parallel.load_balancer import get_batches_test, get_batches_train
from parallel.models import fsrs_v7, fsrs_v7_constants
from parallel.tensors import Data, DataBuilder, ParamKey, ReviewData, UserTensorBlob

enzyme_sample = srs_ops.enzyme_sample
THREADS_PER_BLOCK = srs_ops.THREADS_PER_BLOCK
TRAIN_BUFFER_FLOAT_SIZE = int(TRAIN_BUFFER_SIZE_GB * 1_000_000_000) // 4

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

def run_cpp_train_pass(
    elapsed_days_real,
    rating,
    start_indices,
    seq_lens,
    batch_fsrs_params,
    state_buffer,
):
    U, B = seq_lens.shape
    seq_lens_UxT = seq_lens.view(U, B // THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    seq_lens_Ux_max = seq_lens_UxT.max(dim=-1).values
    flat = seq_lens_Ux_max.view(-1)
    seq_lens_Ux_max_cumsum_inc = (flat * THREADS_PER_BLOCK).cumsum(dim=0, dtype=torch.int32)
    seq_lens_Ux_max_cumsum = torch.nn.functional.pad(
        seq_lens_Ux_max_cumsum_inc[:-1],
        (1, 0),
        value=0,
    ).view(U, B // THREADS_PER_BLOCK)
    return torch.ops.srs.fsrs7_train(
        elapsed_days_real, 
        rating, 
        start_indices.view_as(seq_lens_UxT),
        seq_lens_UxT,
        seq_lens_Ux_max,
        seq_lens_Ux_max_cumsum,
        batch_fsrs_params,
        state_buffer,
    )


@torch.compile(fullgraph=True)
def train_iter(
    flat_fsrs_params: torch.Tensor,
    optim_state: adamw.AdamWState,
    step_i_cat: torch.Tensor,
    batch_perm_cat: torch.Tensor,
    train_split_lengths_cat: torch.Tensor,
    num_training_steps_cat: torch.Tensor,
    num_training_steps_per_epoch_cat: torch.Tensor,
    batch_perm_user_flat_offset: torch.Tensor,
    train_split_lengths_offset: torch.Tensor,
    train_index: torch.Tensor,
    elapsed_days_real: torch.Tensor,
    rating: torch.Tensor,
    seq_len: torch.Tensor,
    state_buffer: torch.Tensor,
    batch_num_inner_batches: int,
) -> tuple[torch.Tensor, adamw.AdamWState, torch.Tensor]:
    remaining = num_training_steps_cat - step_i_cat
    _, indices = torch.topk(remaining, k=batch_num_inner_batches)
    active = remaining[indices] > 0

    step_i = step_i_cat[indices]
    max_step_i = (num_training_steps_cat[indices] - 1).clamp_min(0)
    safe_step_i = torch.minimum(step_i, max_step_i)

    user_indices = indices // N_SPLITS
    split_indices = indices % N_SPLITS
    perm_offset = (
        batch_perm_user_flat_offset[user_indices, split_indices] + safe_step_i
    ).clamp_max(batch_perm_cat.size(0) - 1)
    perm_i = batch_perm_cat[perm_offset]

    train_l = perm_i * BATCH_SIZE
    train_r = torch.minimum(
        (perm_i + 1) * BATCH_SIZE - 1,
        train_split_lengths_cat[indices] - 1,
    )
    train_range = train_l.unsqueeze(-1) + torch.arange(
        BATCH_SIZE,
        device=train_l.device,
    ).view(1, -1).expand(train_l.size(0), -1)

    legal = (train_range <= train_r.unsqueeze(-1)) & active.unsqueeze(-1)
    review_data_indices = train_index[
        (
            train_split_lengths_offset[user_indices, split_indices].unsqueeze(-1)
            + train_range
        ).clamp_max(train_index.size(0) - 1)
    ]
    batch_seq_lens = seq_len[review_data_indices]
    start_indices = review_data_indices - batch_seq_lens + 1
    batch_fsrs_params = flat_fsrs_params[indices]

    per_example_grad = run_cpp_train_pass(
        elapsed_days_real,
        rating,
        start_indices,
        batch_seq_lens,
        batch_fsrs_params,
        state_buffer,
    )
    selected_grad = (per_example_grad * legal.unsqueeze(-1)).sum(dim=1)
    flat_grad = torch.zeros_like(flat_fsrs_params).scatter_add(
        0,
        indices.unsqueeze(-1).expand_as(selected_grad),
        selected_grad,
    )

    active_i = active.to(dtype=step_i_cat.dtype)
    active_params_mask_i = torch.zeros_like(step_i_cat).scatter_add(
        0,
        indices,
        active_i,
    )
    active_params_mask = active_params_mask_i > 0
    lr_schedule_multi = 2e-2 * scheduler.scheduler(step_i_cat, num_training_steps_cat)

    new_flat_fsrs_params, new_optim_state = adamw.adamw_step(
        flat_fsrs_params,
        flat_grad,
        optim_state,
        lr=lr_schedule_multi,
        mask=active_params_mask,
    )
    new_flat_fsrs_params = fsrs_v7_constants.apply_parameter_clipper(new_flat_fsrs_params)
    new_step_i_cat = step_i_cat + active_params_mask_i

    return new_flat_fsrs_params, new_optim_state, new_step_i_cat


def train(fsrs_params: torch.Tensor, users: list[int], data: Data):
    print("start train")
    train_split_lengths_cat = data.train_split_lengths
    num_training_steps_per_epoch_cat = (train_split_lengths_cat + BATCH_SIZE - 1) // BATCH_SIZE
    num_training_steps_cat = N_EPOCHS * num_training_steps_per_epoch_cat
    batch_perm_cat = torch.cat([torch.randperm(n) for n in num_training_steps_per_epoch_cat.cpu().numpy() for _ in range(N_EPOCHS)]).to(DEVICE)
    # Map: (user, split) -> batch_perm_cat start index
    batch_perm_user_flat_offset = torch.nn.functional.pad(
        torch.cumsum(num_training_steps_cat, dim=-1)[:-1],
        (1, 0),
    ).view(len(users), N_SPLITS)

    # Map: (user, split) -> data.train_index start index
    train_split_lengths_offset = torch.nn.functional.pad(
        torch.cumsum(train_split_lengths_cat, dim=-1)[:-1],
        (1, 0),
    ).view(len(users), N_SPLITS)
    train_splits_length_cat_sum = int(num_training_steps_cat.sum().item())
    train_splits_length_cat_max = int(num_training_steps_cat.max().item())
    batch_num_inner_batches = ceil_div(train_splits_length_cat_sum, train_splits_length_cat_max)
    
    step_i_cat = torch.zeros_like(num_training_steps_cat)
    flat_fsrs_params = fsrs_params.detach().view(-1, fsrs_params.size(-1))
    optim_state = adamw.init_adamw_state(flat_fsrs_params)
    state_buffer = torch.empty(TRAIN_BUFFER_FLOAT_SIZE, dtype=torch.float32, device=DEVICE)
    for iter in tqdm(range(train_splits_length_cat_max), desc="Training", smoothing=0.03):
        flat_fsrs_params, optim_state, step_i_cat = train_iter(
            flat_fsrs_params,
            optim_state,
            step_i_cat,
            batch_perm_cat,
            train_split_lengths_cat,
            num_training_steps_cat,
            num_training_steps_per_epoch_cat,
            batch_perm_user_flat_offset,
            train_split_lengths_offset,
            data.train_index,
            data.review_data.elapsed_days_real,
            data.review_data.rating,
            data.review_data.seq_len,
            state_buffer,
            batch_num_inner_batches,
        )


    assert (step_i_cat == num_training_steps_cat).all()
    print("------------------done train-----------------")
    return flat_fsrs_params.view_as(fsrs_params)

def evaluate_on_test_set(fsrs_params: torch.Tensor, users: list[int], data: Data):
    print("eval on test")
    torch.cuda.synchronize()
    tz = time.time()
    ts = time.time()
    param_keys = data.get_test_index_param_key()

    test_seq_len = data.review_data.seq_len[data.test_index]
    _, sorted_test_index_permutation = torch.sort(test_seq_len, stable=True, descending=True)
    torch.cuda.synchronize()
    print("sort done", time.time() - ts)

    # TODO use the same load balancing as training
    N = data.test_index.size(0)
    num_batches = ceil_div(N, TEST_BATCH_SIZE_MAX)
    batch_size = ceil_div(N, num_batches)
    gather_p = []

    for perm_slice in tqdm(sorted_test_index_permutation.split(batch_size), desc="Test set", smoothing=0.03):
        batch_fsrs_params = fsrs_params[param_keys.user_index[perm_slice], param_keys.split_index[perm_slice]]
        test_index_perm_slice = data.test_index[perm_slice]
        seq_lens = data.review_data.seq_len[test_index_perm_slice]
        start_indices = test_index_perm_slice - seq_lens + 1
        p = enzyme_sample.fsrs7_test(
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

    for user, pred, label in zip(users, p_by_user, label_by_user):
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
    fsrs_params = initial_params.view(1, 1, -1).repeat(len(users), N_SPLITS, 1).requires_grad_(True)
    fsrs_params = train(fsrs_params, users, data)

    print("skip test")
    # evaluate
    # with torch.no_grad():
    #     evaluate_on_test_set(fsrs_params, users, data)

def main() -> None:
    assert DEVICE == "cuda", "Only cuda is supported."
    env = lmdb.open(
        str(LMDB_PATH),
        map_size=LMDB_SIZE,
        readonly=True,
        lock=False,
    )
    
    users = list(range(1, 5000))
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
    torch.manual_seed(123)
    main()
