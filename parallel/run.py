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
    LMDB_PATH,
    LMDB_SIZE,
    DEVICE,
    N_EPOCHS,
    N_SPLITS,
    TEST_BATCH_SIZE_MAX,
    USER_END,
    USER_MAX_TRAIN_SPLIT_LENGTHS_KEY,
    USER_START,
)
from parallel.load_balancer import get_batches_test, get_batches_train
from parallel.models import fsrs_v7, fsrs_v7_constants
from parallel.tensor_cache import (
    TrainSetup,
    build_batch_perm_cat_for_users,
    load_cached_review_data,
    load_cached_test_only,
    load_cached_train_only,
    load_or_rebuild_tensor_cache,
)
from parallel.tensors import Data, ParamKey

enzyme_sample = srs_ops.enzyme_sample
THREADS_PER_BLOCK = srs_ops.THREADS_PER_BLOCK

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def load_metadata_tensor(
    txn: lmdb.Transaction,
    key: str,
    map_location: str | torch.device = "cpu",
) -> torch.Tensor:
    tensor_bytes = txn.get(key.encode())
    if tensor_bytes is None:
        raise KeyError(f"Missing LMDB metadata key: {key}")
    return torch.load(
        BytesIO(tensor_bytes),
        weights_only=True,
        map_location=map_location,
    )


def split_users_by_train_length(
    users: list[int],
    user_max_train_split_lengths: torch.Tensor,
    k: int,
) -> list[list[int]]:
    if k <= 0:
        raise ValueError("k must be positive.")
    if not users:
        return []

    lengths = user_max_train_split_lengths.cpu().to(dtype=torch.int64)
    user_tensor = torch.tensor(users, dtype=torch.int64)
    if int(user_tensor.min().item()) < 1 or int(user_tensor.max().item()) > lengths.numel():
        raise ValueError("users must be 1-indexed into user_max_train_split_lengths.")

    selected_lengths = lengths[user_tensor - 1]
    sorted_lengths, order = torch.sort(selected_lengths, descending=True, stable=True)
    sorted_users = user_tensor[order]
    n = sorted_users.numel()
    section_count = min(k, n)

    if int(sorted_lengths.sum().item()) == 0:
        return [
            sorted_users[
                round(n * i / section_count) : round(n * (i + 1) / section_count)
            ].tolist()
            for i in range(section_count)
        ]

    prefix = torch.cumsum(sorted_lengths, dim=0)
    prefix_float = prefix.to(dtype=torch.float64)
    total = int(prefix[-1].item())
    boundaries = [0]
    for section in range(1, section_count):
        target = total * section / section_count
        boundary = int(
            torch.searchsorted(
                prefix_float,
                torch.tensor(target, dtype=prefix_float.dtype),
                right=False,
            ).item()
        ) + 1
        if boundary > 1:
            prev_sum = int(prefix[boundary - 2].item())
            cur_sum = int(prefix[boundary - 1].item())
            if abs(prev_sum - target) <= abs(cur_sum - target):
                boundary -= 1
        boundary = max(boundaries[-1] + 1, boundary)
        boundary = min(boundary, n - (section_count - section))
        boundaries.append(boundary)
    boundaries.append(n)

    return [
        sorted_users[boundaries[i] : boundaries[i + 1]].tolist()
        for i in range(section_count)
    ]

def run_cpp_train_pass(
    elapsed_days_real,
    rating,
    start_indices,
    seq_lens,
    batch_fsrs_params,
    threads_per_block: int,
):
    U, B = seq_lens.shape
    seq_lens_UxT = seq_lens.view(U, B // threads_per_block, threads_per_block)
    seq_lens_Ux_max = seq_lens_UxT.max(dim=-1).values
    flat = seq_lens_Ux_max.view(-1)
    seq_lens_Ux_max_cumsum_inc = (flat * threads_per_block).cumsum(dim=0, dtype=torch.int32)
    seq_lens_Ux_max_cumsum = torch.nn.functional.pad(
        seq_lens_Ux_max_cumsum_inc[:-1],
        (1, 0),
        value=0,
    ).view(U, B // threads_per_block)
    return torch.ops.srs.fsrs7_train(
        elapsed_days_real, 
        rating, 
        start_indices.view_as(seq_lens_UxT),
        seq_lens_UxT,
        seq_lens_Ux_max,
        seq_lens_Ux_max_cumsum,
        batch_fsrs_params,
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
    threads_per_block: int,
    batch_num_inner_batches: int,
) -> tuple[torch.Tensor, adamw.AdamWState, torch.Tensor]:
    remaining = num_training_steps_cat - step_i_cat
    _, indices = torch.topk(remaining, k=batch_num_inner_batches)
    active_mask = remaining[indices] > 0

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

    legal = (train_range <= train_r.unsqueeze(-1)) & active_mask.unsqueeze(-1)
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
        threads_per_block,
    )
    selected_grad = (per_example_grad * legal.unsqueeze(-1)).sum(dim=1)
    flat_grad = torch.zeros_like(flat_fsrs_params).scatter_add(
        0,
        indices.unsqueeze(-1).expand_as(selected_grad),
        selected_grad,
    )

    lr_schedule_multi = 2e-2 * scheduler.scheduler(step_i_cat, num_training_steps_cat)
    lr_schedule_multi = lr_schedule_multi.unsqueeze(-1).expand(-1, flat_fsrs_params.size(-1))

    active_params_mask_i = torch.zeros_like(step_i_cat).scatter_add(
        0,
        indices,
        torch.ones_like(indices, dtype=step_i_cat.dtype),
    )
    active_params_mask = torch.where(active_params_mask_i > 0, remaining > 0, torch.zeros_like(remaining, dtype=torch.bool))

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


def build_train_setup(data: Data, users: list[int]) -> TrainSetup:
    train_split_lengths_cat = data.train_split_lengths
    num_training_steps_per_epoch_cat = (train_split_lengths_cat + BATCH_SIZE - 1) // BATCH_SIZE
    num_training_steps_cat = N_EPOCHS * num_training_steps_per_epoch_cat

    batch_perm_cat = torch.from_numpy(
        build_batch_perm_cat_for_users(
            users,
            num_training_steps_per_epoch_cat.cpu().numpy(),
        )
    ).to(DEVICE)

    batch_perm_user_flat_offset = torch.nn.functional.pad(
        torch.cumsum(num_training_steps_cat.to(dtype=torch.int64), dim=-1)[:-1],
        (1, 0),
    ).view(len(users), N_SPLITS)
    train_split_lengths_offset = torch.nn.functional.pad(
        torch.cumsum(train_split_lengths_cat.to(dtype=torch.int64), dim=-1)[:-1],
        (1, 0),
    ).view(len(users), N_SPLITS)

    train_splits_length_cat_sum = int(num_training_steps_cat.sum().item())
    train_splits_length_cat_max = (
        int(num_training_steps_cat.max().item())
        if num_training_steps_cat.numel() > 0
        else 0
    )
    batch_num_inner_batches = (
        ceil_div(train_splits_length_cat_sum, train_splits_length_cat_max)
        if train_splits_length_cat_max > 0
        else 0
    )
    return TrainSetup(
        num_training_steps_per_epoch_cat=num_training_steps_per_epoch_cat,
        num_training_steps_cat=num_training_steps_cat,
        batch_perm_cat=batch_perm_cat,
        batch_perm_user_flat_offset=batch_perm_user_flat_offset,
        train_split_lengths_offset=train_split_lengths_offset,
        batch_num_inner_batches=batch_num_inner_batches,
    )


def train(
    fsrs_params: torch.Tensor,
    users: list[int],
    data: Data,
    train_setup: TrainSetup | None = None,
):
    print("start train")
    train_split_lengths_cat = data.train_split_lengths
    if train_setup is None:
        train_setup = build_train_setup(data, users)

    num_training_steps_per_epoch_cat = train_setup.num_training_steps_per_epoch_cat
    num_training_steps_cat = train_setup.num_training_steps_cat
    batch_perm_cat = train_setup.batch_perm_cat
    batch_perm_user_flat_offset = train_setup.batch_perm_user_flat_offset
    train_split_lengths_offset = train_setup.train_split_lengths_offset
    batch_num_inner_batches = train_setup.batch_num_inner_batches
    train_splits_length_cat_max = (
        int(num_training_steps_cat.max().item())
        if num_training_steps_cat.numel() > 0
        else 0
    )
    
    step_i_cat = torch.zeros_like(num_training_steps_cat)
    flat_fsrs_params = fsrs_params.detach().view(-1, fsrs_params.size(-1))
    optim_state = adamw.init_adamw_state(flat_fsrs_params)
    print("Inner batches:", batch_num_inner_batches)
    for iter in tqdm(range(train_splits_length_cat_max), desc="Training", smoothing=0.06):
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
            THREADS_PER_BLOCK,
            batch_num_inner_batches,
        )

    assert (step_i_cat >= num_training_steps_cat).all()
    assert (step_i_cat == num_training_steps_cat).any()
    print("------------------done train-----------------")
    return flat_fsrs_params.view_as(fsrs_params)

def evaluate_on_test_set(fsrs_params: torch.Tensor, users: list[int], data: Data):
    print("eval on test")
    # time.sleep(10)
    # exit()
    # print(fsrs_params)
    param_keys = data.get_test_index_param_key()
    # print(data.split_counts, data.split_counts.shape)
    # assert (data.split_counts == data.split_counts[0]).all()
    # print(data.splits, data.splits.view(-1, data.split_counts[0].item()))
    assert data.split_counts.size(0) == len(users)
    # test_dataset_size_by_user = data.splits.view(-1, N_SPLITS).sum(dim=-1)
    test_index_lens = torch.tensor(data.test_index_lens, dtype=torch.int, device=fsrs_params.device)
    per_user_weight_flat = torch.repeat_interleave(1.0 / per_user_weight, test_index_lens)
    # print(test_dataset_size_by_user, test_dataset_size_by_user.sum())
    # print(torch.tensor(data.test_index_lens))
    print(per_user_weight_flat)

    test_seq_len = data.test_index.size(0)
    # _, sorted_test_index_permutation = torch.sort(test_seq_len, stable=True, descending=True)

    # TODO use the same load balancing as training
    N = data.test_index.size(0)
    num_batches = ceil_div(N, TEST_BATCH_SIZE_MAX)
    batch_size = ceil_div(N, num_batches)
    p_concat = torch.empty(
        (test_seq_len,),
        device=fsrs_params.device,
        dtype=fsrs_params.dtype,
    )

    # for perm_slice in tqdm(torch.arange(test_seq_len).split(batch_size), desc="Test set", smoothing=0.03):
    for l in tqdm(range(0, test_seq_len, batch_size)):
        re = min(test_seq_len, l + batch_size)
        batch_fsrs_params = fsrs_params[param_keys.user_index[l:re], param_keys.split_index[l:re]]
        test_index_perm_slice = data.test_index[l:re]
        seq_lens = data.review_data.seq_len[test_index_perm_slice]
        start_indices = test_index_perm_slice - seq_lens + 1
        p = enzyme_sample.fsrs7_test(
                data.review_data.elapsed_days_real, 
                data.review_data.rating, 
                start_indices,
                seq_lens,
                batch_fsrs_params,
            )
        p_concat[l:re].copy_(p)
    
    # restore_test_index_permutation = torch.empty_like(sorted_test_index_permutation)
    # restore_test_index_permutation[sorted_test_index_permutation] = torch.arange(
    #     sorted_test_index_permutation.numel(),
    #     device=sorted_test_index_permutation.device,
    # )
    del param_keys
    del test_seq_len
    # del sorted_test_index_permutation
    # del restore_test_index_permutation
    torch.cuda.empty_cache()
    # time.sleep(10)
    # exit()

    label = (data.review_data.rating[data.test_index] > 1).float()
    loss = torch.nn.functional.binary_cross_entropy(p_concat, label, reduction='none')
    logloss_weighted_by_reviews = loss.mean()
    logloss_weighted_by_user = (loss * per_user_weight_flat).sum() / len(users)
    print("Log loss avg:", logloss_weighted_by_reviews, label.size(0))
    print("Log loss avg by user:", logloss_weighted_by_user, label.size(0))

    # p_by_user = p_concat.split(data.test_index_lens)
    # label_by_user = label.split(data.test_index_lens)
    # for user, pred, label in zip(users, p_by_user, label_by_user):
    #     logloss = log_loss(y_true=label.cpu().numpy(), y_pred=pred.cpu().numpy(), labels=[0, 1])
    #     print(f"User: {user}, logloss={logloss:.3f}")

def run(
    users: list[int],
    data: Data,
    train_setup: TrainSetup,
) -> torch.Tensor:
    return fsrs_params

def main() -> None:
    assert DEVICE == "cuda", "Only cuda is supported."
    env = lmdb.open(
        str(LMDB_PATH),
        map_size=LMDB_SIZE,
        readonly=True,
        lock=False,
    )
    
    users = list(range(USER_START, USER_END + 1))

    split_factor_k = 1
    with env.begin(write=False) as txn:
        user_max_train_split_lengths = load_metadata_tensor(
            txn,
            USER_MAX_TRAIN_SPLIT_LENGTHS_KEY,
        )

    user_splits = split_users_by_train_length(
        users,
        user_max_train_split_lengths,
        split_factor_k,
    )
    user_splits.reverse()
    for l in user_splits:
        l.sort()
    # user_splits = [users]  # overwrite
    cache_env = load_or_rebuild_tensor_cache(env, user_splits)

    try:
        for split_i, user_subset in enumerate(user_splits):
            user_indices = torch.tensor(user_subset, dtype=torch.int64) - 1
            split_work = int(user_max_train_split_lengths[user_indices].sum().item())
            print(
                f"Run split {split_i + 1}/{len(user_splits)}: "
                f"users={len(user_subset)}, max_train_split_length_sum={split_work}"
            )
            torch.cuda.empty_cache()
            review_data = load_cached_review_data(cache_env, split_i, DEVICE)
            # train_data, train_setup = load_cached_train_only(
            #     cache_env,
            #     split_i,
            #     DEVICE,
            #     review_data,
            # )

            initial_params = fsrs_v7_constants.get_initial_params_for_optimization().to(DEVICE)
            fsrs_params = initial_params.view(1, 1, -1).repeat(len(users), N_SPLITS, 1).requires_grad_(True)
            # fsrs_params = train(fsrs_params, users, train_data, train_setup)

            # del train_data
            # del train_setup
            # torch.cuda.empty_cache()

            test_data = load_cached_test_only(cache_env, split_i, DEVICE, review_data)
            with torch.no_grad():
                evaluate_on_test_set(fsrs_params, user_subset, test_data)
            del test_data
            del review_data
            del fsrs_params
            torch.cuda.empty_cache()
    finally:
        cache_env.close()
    env.close()


if __name__ == "__main__":
    torch.manual_seed(123)
    main()
