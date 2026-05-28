from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import math

import lmdb
import torch
from tqdm import tqdm
import time

from parallel import srs_ops
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
    WRITE_RESULT,
    WRITE_RESULT_FILE,
)
from parallel.fsrs import fsrs_v7_scheduler
from parallel.fsrs import fsrs_v7_constants, fsrs_v7_helpers, fsrs_v7_optimizer
from parallel.result_metrics import (
    FsrsParamSummary,
    flatten_fsrs_params_for_summary,
    format_param_stat,
    summarize_fsrs_param_parts,
    write_user_result_jsonl,
)
from parallel.tensor_cache import (
    TrainSetup,
    build_batch_perm_cat_for_users,
    load_cached_review_data,
    load_cached_test_only,
    load_cached_train_only,
    load_or_rebuild_tensor_cache,
)
from parallel.tensors import Data, ParamKey

fsrs_extension = srs_ops.fsrs_extension
THREADS_PER_BLOCK = srs_ops.THREADS_PER_BLOCK


@dataclass(frozen=True)
class EvaluationResult:
    logloss_by_review: float
    logloss_by_user: float
    review_count: int
    user_count: int
    p_by_user: dict[int, torch.Tensor]
    label_by_user: dict[int, torch.Tensor]
    rmse_bins_by_user: dict[int, torch.Tensor]
    fsrs_params_by_user: dict[int, torch.Tensor]
    fsrs_param_rows: torch.Tensor


@dataclass
class EvaluationAggregate:
    review_loss_sum: float = 0.0
    review_count: int = 0
    user_loss_sum: float = 0.0
    user_count: int = 0
    p_by_user: dict[int, torch.Tensor] = field(default_factory=dict)
    label_by_user: dict[int, torch.Tensor] = field(default_factory=dict)
    rmse_bins_by_user: dict[int, torch.Tensor] = field(default_factory=dict)
    fsrs_params_by_user: dict[int, torch.Tensor] = field(default_factory=dict)
    fsrs_param_rows_parts: list[torch.Tensor] = field(default_factory=list)

    def add(self, result: EvaluationResult) -> None:
        self.review_loss_sum += result.logloss_by_review * result.review_count
        self.review_count += result.review_count
        self.user_loss_sum += result.logloss_by_user * result.user_count
        self.user_count += result.user_count
        self.p_by_user.update(result.p_by_user)
        self.label_by_user.update(result.label_by_user)
        self.rmse_bins_by_user.update(result.rmse_bins_by_user)
        self.fsrs_params_by_user.update(result.fsrs_params_by_user)
        self.fsrs_param_rows_parts.append(result.fsrs_param_rows)

    @property
    def logloss_by_review(self) -> float:
        if self.review_count == 0:
            return float("nan")
        return self.review_loss_sum / self.review_count

    @property
    def logloss_by_user(self) -> float:
        if self.user_count == 0:
            return float("nan")
        return self.user_loss_sum / self.user_count

    def fsrs_param_summary(self) -> FsrsParamSummary:
        return summarize_fsrs_param_parts(self.fsrs_param_rows_parts)


def write_evaluation_results(
    aggregate: EvaluationAggregate,
    output_file: str = WRITE_RESULT_FILE,
) -> None:
    write_user_result_jsonl(
        output_file,
        list(aggregate.p_by_user),
        aggregate.p_by_user,
        aggregate.label_by_user,
        aggregate.rmse_bins_by_user,
        aggregate.fsrs_params_by_user,
    )


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

    lengths = user_max_train_split_lengths.cpu().to(dtype=torch.int32)
    user_tensor = torch.tensor(users, dtype=torch.int32)
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

    prefix = torch.cumsum(sorted_lengths, dim=0, dtype=torch.int32)
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
    grad_weight,
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
        grad_weight.view_as(seq_lens_UxT),
        seq_lens_UxT,
        seq_lens_Ux_max,
        seq_lens_Ux_max_cumsum,
        batch_fsrs_params,
    )

def masked_penalty(parameters_kp, mask_k, batch_size_k, training_set_size_k):
    return (mask_k * fsrs_v7_helpers.penalty_loss(parameters_kp, batch_size_k, training_set_size_k)).sum()

_penalty_grad = torch.func.grad(masked_penalty, argnums=0)

@torch.compile(fullgraph=True, dynamic=False)
def train_iter(
    flat_fsrs_params: torch.Tensor,
    optim_state: fsrs_v7_optimizer.AdamWState,
    step_i_cat: torch.Tensor,
    batch_perm_cat: torch.Tensor,
    train_split_lengths_cat: torch.Tensor,
    num_training_steps_cat: torch.Tensor,
    num_training_steps_per_epoch_cat: torch.Tensor,
    batch_perm_user_flat_offset: torch.Tensor,
    train_split_lengths_offset: torch.Tensor,
    train_index: torch.Tensor,
    split_review_ord: torch.Tensor,
    elapsed_days_real: torch.Tensor,
    rating: torch.Tensor,
    seq_len: torch.Tensor,
    threads_per_block: int,
    batch_num_inner_batches: int,
) -> tuple[torch.Tensor, fsrs_v7_optimizer.AdamWState, torch.Tensor]:
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
        dtype=train_l.dtype,
    ).view(1, -1).expand(train_l.size(0), -1)

    legal = (train_range <= train_r.unsqueeze(-1)) & active_mask.unsqueeze(-1)
    index_within_flat = (
            train_split_lengths_offset[user_indices, split_indices].unsqueeze(-1)
            + train_range
        ).clamp_max(train_index.size(0) - 1)
    review_data_indices = train_index[index_within_flat]
    batch_seq_lens = seq_len[review_data_indices]
    start_indices = review_data_indices - batch_seq_lens + 1
    batch_fsrs_params = flat_fsrs_params[indices]
    grad_weight = fsrs_v7_helpers.gradient_weight(split_review_ord[index_within_flat], train_split_lengths_cat[indices])

    per_example_grad = run_cpp_train_pass(
        elapsed_days_real,
        rating,
        start_indices,
        batch_seq_lens,
        grad_weight,
        batch_fsrs_params,
        threads_per_block,
    )
    selected_grad = (per_example_grad * legal.unsqueeze(-1)).sum(dim=1)
    penalty_grad = _penalty_grad(batch_fsrs_params, active_mask, train_r - train_l + 1, train_split_lengths_cat[indices])
    grad = selected_grad + penalty_grad
    flat_grad = torch.zeros_like(flat_fsrs_params).scatter_add(
        0,
        indices.unsqueeze(-1).expand_as(grad),
        grad,
    )

    lr_schedule_multi = fsrs_v7_constants.LR * fsrs_v7_scheduler.scheduler(step_i_cat, num_training_steps_cat)
    lr_schedule_multi = lr_schedule_multi.unsqueeze(-1).expand(-1, flat_fsrs_params.size(-1))

    active_params_mask_i = torch.zeros_like(step_i_cat).scatter_add(
        0,
        indices,
        torch.ones_like(indices, dtype=step_i_cat.dtype),
    )
    active_params_mask = torch.where(active_params_mask_i > 0, remaining > 0, torch.zeros_like(remaining, dtype=torch.bool))

    new_flat_fsrs_params, new_optim_state = fsrs_v7_optimizer.adamw_step(
        flat_fsrs_params,
        flat_grad,
        optim_state,
        lr=lr_schedule_multi,
        betas=fsrs_v7_constants.BETAS,
        mask=active_params_mask,
    )
    new_flat_fsrs_params = fsrs_v7_helpers.apply_parameter_clipper(new_flat_fsrs_params)
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
        torch.cumsum(num_training_steps_cat.to(dtype=torch.int32), dim=-1, dtype=torch.int32)[:-1],
        (1, 0),
    ).view(len(users), N_SPLITS)
    train_split_lengths_offset = torch.nn.functional.pad(
        torch.cumsum(train_split_lengths_cat.to(dtype=torch.int32), dim=-1, dtype=torch.int32)[:-1],
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
        split_review_ord=data.split_review_ord,
        batch_num_inner_batches=batch_num_inner_batches,
    )


def train(
    fsrs_params: torch.Tensor,
    data: Data,
    train_setup: TrainSetup,
):
    train_split_lengths_cat = data.train_split_lengths
    num_training_steps_per_epoch_cat = train_setup.num_training_steps_per_epoch_cat
    num_training_steps_cat = train_setup.num_training_steps_cat
    batch_perm_cat = train_setup.batch_perm_cat
    batch_perm_user_flat_offset = train_setup.batch_perm_user_flat_offset
    train_split_lengths_offset = train_setup.train_split_lengths_offset
    split_review_ord = train_setup.split_review_ord
    batch_num_inner_batches = train_setup.batch_num_inner_batches
    train_splits_length_cat_max = (
        int(num_training_steps_cat.max().item())
        if num_training_steps_cat.numel() > 0
        else 0
    )
    if train_splits_length_cat_max == 0:
        return fsrs_params
    
    step_i_cat = torch.zeros_like(num_training_steps_cat)
    flat_fsrs_params = fsrs_params.view(-1, fsrs_params.size(-1))
    optim_state = fsrs_v7_optimizer.init_adamw_state(flat_fsrs_params)
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
            split_review_ord,
            data.review_data.elapsed_days_real,
            data.review_data.rating,
            data.review_data.seq_len,
            THREADS_PER_BLOCK,
            batch_num_inner_batches,
        )

    assert (step_i_cat >= num_training_steps_cat).all()
    assert (step_i_cat == num_training_steps_cat).any()
    return flat_fsrs_params.view_as(fsrs_params)

def predict_test_set(fsrs_params: torch.Tensor, data: Data) -> torch.Tensor:
    param_keys = data.get_test_index_param_key()
    test_seq_len = data.test_index.size(0)
    N = data.test_index.size(0)
    num_batches = ceil_div(N, TEST_BATCH_SIZE_MAX)
    batch_size = ceil_div(N, num_batches)
    p_concat = torch.empty(
        (test_seq_len,),
        device=fsrs_params.device,
        dtype=fsrs_params.dtype,
    )

    for l in tqdm(range(0, test_seq_len, batch_size)):
        re = min(test_seq_len, l + batch_size)
        batch_fsrs_params = fsrs_params[param_keys.user_index[l:re], param_keys.split_index[l:re]]
        test_index_perm_slice = data.test_index[l:re]
        seq_lens = data.review_data.seq_len[test_index_perm_slice]
        start_indices = test_index_perm_slice - seq_lens + 1
        p = fsrs_extension.fsrs7_test(
                data.review_data.elapsed_days_real, 
                data.review_data.rating, 
                start_indices,
                seq_lens,
                batch_fsrs_params,
            )
        p_concat[l:re].copy_(p)
    return p_concat


def evaluate_on_test_set(fsrs_params: torch.Tensor, users: list[int], data: Data) -> EvaluationResult:
    print("eval on test")
    assert data.split_counts.size(0) == len(users)
    assert fsrs_params.size(0) == len(users)

    p_concat = predict_test_set(fsrs_params, data)
    torch.cuda.empty_cache()

    label = (data.review_data.rating[data.test_index] > 1).float()
    loss = torch.nn.functional.binary_cross_entropy(p_concat, label, reduction='none')
    logloss_weighted_by_reviews = loss.mean()
    logloss_weighted_by_user = \
        (loss * torch.repeat_interleave(1.0 / data.test_index_lens, data.test_index_lens)).sum() / len(users)
    p_by_user: dict[int, torch.Tensor] = {}
    label_by_user: dict[int, torch.Tensor] = {}
    rmse_bins_by_user: dict[int, torch.Tensor] = {}
    fsrs_params_by_user: dict[int, torch.Tensor] = {}
    fsrs_param_rows = flatten_fsrs_params_for_summary(fsrs_params)
    if WRITE_RESULT:
        test_index_lens = data.test_index_lens.cpu().tolist()
        p_by_user_parts = p_concat.detach().cpu().split(test_index_lens)
        label_by_user_parts = label.detach().cpu().split(test_index_lens)
        rmse_bins_by_user_parts = data.rmse_bins.detach().cpu().split(test_index_lens)
        p_by_user = dict(zip(users, p_by_user_parts))
        label_by_user = dict(zip(users, label_by_user_parts))
        rmse_bins_by_user = dict(zip(users, rmse_bins_by_user_parts))
        fsrs_params_cpu = fsrs_param_rows.view_as(fsrs_params)
        fsrs_params_by_user = dict(zip(users, fsrs_params_cpu.unbind(0)))

    result = EvaluationResult(
        logloss_by_review=float(logloss_weighted_by_reviews.item()),
        logloss_by_user=float(logloss_weighted_by_user.item()),
        review_count=int(label.size(0)),
        user_count=len(users),
        p_by_user=p_by_user,
        label_by_user=label_by_user,
        rmse_bins_by_user=rmse_bins_by_user,
        fsrs_params_by_user=fsrs_params_by_user,
        fsrs_param_rows=fsrs_param_rows,
    )
    # print(f"n: {result.review_count}")
    # print(f"Log loss avg by review: {result.logloss_by_review:.5f}")
    # print(f"Log loss avg by user: {result.logloss_by_user:.5f}")

    # for user, pred, label in zip(users, p_by_user, label_by_user):
    #     logloss = log_loss(y_true=label.cpu().numpy(), y_pred=pred.cpu().numpy(), labels=[0, 1])
    #     print(f"User: {user}, logloss={logloss:.3f}")
    return result


def make_initial_fsrs_params(user_count: int) -> torch.Tensor:
    initial_params = fsrs_v7_helpers.get_initial_params_for_optimization().to(DEVICE)
    return initial_params.view(1, 1, -1).repeat(user_count, N_SPLITS, 1)


def run(
    users: list[int],
    data: Data,
    train_setup: TrainSetup,
) -> torch.Tensor:
    fsrs_params = make_initial_fsrs_params(len(users))
    fsrs_params = train(fsrs_params, data, train_setup)
    return fsrs_params


def train_cached_split(
    cache_env: lmdb.Environment,
    split_i: int,
    users: list[int],
    review_data,
) -> torch.Tensor:
    train_data, train_setup = load_cached_train_only(
        cache_env,
        split_i,
        DEVICE,
        review_data,
    )
    return run(users, train_data, train_setup).detach()


def evaluate_cached_split(
    cache_env: lmdb.Environment,
    split_i: int,
    users: list[int],
    review_data,
    fsrs_params: torch.Tensor,
) -> EvaluationResult:
    test_data = load_cached_test_only(
        cache_env,
        split_i,
        DEVICE,
        review_data,
        load_rmse_bins=WRITE_RESULT,
    )
    with torch.no_grad():
        return evaluate_on_test_set(fsrs_params, users, test_data)


def run_cached_split(
    cache_env: lmdb.Environment,
    split_i: int,
    split_count: int,
    user_subset: list[int],
    user_max_train_split_lengths: torch.Tensor,
) -> EvaluationResult:
    user_indices = torch.tensor(user_subset, dtype=torch.int32) - 1
    split_work = int(user_max_train_split_lengths[user_indices].sum().item())
    print(
        f"Run split {split_i + 1}/{split_count}: "
        f"users={len(user_subset)}, max_train_split_length_sum={split_work}"
    )

    torch.cuda.empty_cache()
    review_data = load_cached_review_data(cache_env, split_i, DEVICE)
    fsrs_params = train_cached_split(cache_env, split_i, user_subset, review_data)

    torch.cuda.empty_cache()
    result = evaluate_cached_split(cache_env, split_i, user_subset, review_data, fsrs_params)
    torch.cuda.empty_cache()
    return result

def get_split_factor_k(req_size):
    props = torch.cuda.get_device_properties(0)
    cuda_mem_GB = props.total_memory / 1000 ** 3 - 0.5
    gb_req = req_size / 510540097 * 22.9
    # smallest x such that cuda_mem_GB * x >= gb_req
    return math.ceil(gb_req / cuda_mem_GB)

def main() -> None:
    assert DEVICE == "cuda", "Only cuda is supported."
    env = lmdb.open(
        str(LMDB_PATH),
        map_size=LMDB_SIZE,
        readonly=True,
        lock=False,
    )
    
    users = list(range(USER_START, USER_END + 1))

    with env.begin(write=False) as txn:
        user_max_train_split_lengths = load_metadata_tensor(
            txn,
            USER_MAX_TRAIN_SPLIT_LENGTHS_KEY,
        )

    rel_user_sum = user_max_train_split_lengths[torch.tensor(users) - 1]
    split_factor_k = get_split_factor_k(rel_user_sum.sum())
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

    eval_aggregate = EvaluationAggregate()
    try:
        for split_i, user_subset in enumerate(user_splits):
            result = run_cached_split(
                cache_env,
                split_i,
                len(user_splits),
                user_subset,
                user_max_train_split_lengths,
            )
            eval_aggregate.add(result)
    finally:
        cache_env.close()
    env.close()

    print(f"Users: {eval_aggregate.user_count}")
    print(f"n reviews: {eval_aggregate.review_count}")
    print(f"Log loss avg by review: {eval_aggregate.logloss_by_review:.5f}")
    print(f"Log loss avg by user: {eval_aggregate.logloss_by_user:.5f}")
    fsrs_param_summary = eval_aggregate.fsrs_param_summary()
    print(f"FSRS params count: {fsrs_param_summary.count}")
    print(f"FSRS params mean: {format_param_stat(fsrs_param_summary.mean)}")
    print(f"FSRS params median: {format_param_stat(fsrs_param_summary.median)}")
    print(f"FSRS params bottom 10%: {format_param_stat(fsrs_param_summary.bottom_10)}")
    print(f"FSRS params top 10%: {format_param_stat(fsrs_param_summary.top_10)}")
    print(f"FSRS params std: {format_param_stat(fsrs_param_summary.std)}")

    if WRITE_RESULT:
        write_evaluation_results(eval_aggregate)



if __name__ == "__main__":
    torch.manual_seed(123)
    main()
