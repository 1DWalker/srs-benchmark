from __future__ import annotations

from io import BytesIO

import jax
import jax.numpy as jnp
import lmdb
import numpy as np
from sklearn.metrics import log_loss
import torch
from tqdm import tqdm
import time

from parallel import adamw_jax as adamw
from parallel import scheduler_jax as scheduler
from parallel.config import (
    BATCH_SIZE,
    LMDB_PATH,
    LMDB_SIZE,
    N_EPOCHS,
    N_SPLITS,
    TEST_BATCH_SIZE_MAX,
)
from parallel.load_balancer import get_batches_test, get_batches_train
from parallel.models import fsrs_v7_jax
from parallel.models import fsrs_v7_jax_adapter
from parallel.tensors import (
    Data,
    ReviewData,
    UserTensorBlob,
    torch_tensor_to_jax_array,
)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _sync_jax(*arrays: jax.Array) -> None:
    if arrays:
        jax.block_until_ready(arrays)


def _assert_all(value: jax.Array) -> None:
    assert bool(jnp.all(value).item())


def _mask_select(value: jax.Array, mask: jax.Array) -> jax.Array:
    size = int(jnp.sum(mask).item())
    if size == 0:
        return jnp.empty((0,), dtype=value.dtype)
    indices = jnp.nonzero(mask, size=size)[0]
    return value[indices]


def _make_batch_permutation(num_training_steps_per_epoch_cat: jax.Array) -> jax.Array:
    sizes = np.asarray(jax.device_get(num_training_steps_per_epoch_cat), dtype=np.int64)
    if sizes.size == 0 or int(sizes.sum()) == 0:
        return jnp.empty((0,), dtype=num_training_steps_per_epoch_cat.dtype)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    perms = []
    for n in sizes:
        for _ in range(N_EPOCHS):
            perms.append(
                torch.randperm(int(n), generator=generator, dtype=torch.int32)
            )
    return torch_tensor_to_jax_array(torch.cat(perms, dim=0))


def load_blob_bytes(blob_bytes: bytes) -> UserTensorBlob:
    buffer = BytesIO(blob_bytes)
    tensors = torch.load(buffer, weights_only=True, map_location="cpu")
    return UserTensorBlob.from_dict(tensors)


def load_metadata_tensor(txn: lmdb.Transaction, key: str) -> jax.Array:
    tensor_bytes = txn.get(key.encode())
    if tensor_bytes is None:
        raise KeyError(f"Missing LMDB metadata key: {key}")
    tensor = torch.load(
        BytesIO(tensor_bytes),
        weights_only=True,
        map_location="cpu",
    )
    return torch_tensor_to_jax_array(tensor)


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


def _prepare_prediction_inputs(review_data: ReviewData, index):
    ts = time.time()
    seq_lens = review_data.seq_len[index]
    max_seq_len = seq_lens.max()

    card_start_index = index - seq_lens + 1
    data_len = review_data.elapsed_days_real.shape[0]

    _sync_jax(max_seq_len)
    to = time.time()
    arange_l = jnp.arange(int(max_seq_len.item()), dtype=index.dtype)
    _sync_jax(arange_l)
    print("make arange_l", time.time() - to)

    take_indices = jnp.minimum(
        card_start_index[:, None] + arange_l[None, :],
        data_len - 1,
    )
    _sync_jax(take_indices)
    print("prepare before take", time.time() - ts)
    ts = time.time()

    rating_bl = review_data.rating[take_indices]
    elapsed_time_real_bl = review_data.elapsed_days_real[take_indices]
    _sync_jax(rating_bl, elapsed_time_real_bl)
    print("prepare take", time.time() - ts)

    # print("shape", rating_bl.shape, max_seq_len)

    return elapsed_time_real_bl, rating_bl, seq_lens


def predict(review_data: ReviewData, index, params):
    assert index.shape[0] == index.shape[0]
    elapsed_time_real_bl, rating_bl, seq_lens = _prepare_prediction_inputs(
        review_data, index
    )
    return fsrs_v7_jax_adapter.prediction(params, elapsed_time_real_bl, rating_bl, seq_lens)


def predict_loss_grad(review_data: ReviewData, index, params):
    _sync_jax(index, params)
    ts = time.time()
    assert index.shape[0] == params.shape[0]
    elapsed_time_real_bl, rating_bl, seq_lens = _prepare_prediction_inputs(
        review_data, index
    )
    _sync_jax(elapsed_time_real_bl, rating_bl, seq_lens)
    print("pred loss grad prep", time.time() - ts)
    return fsrs_v7_jax_adapter.prediction_loss_grad(params, elapsed_time_real_bl, rating_bl, seq_lens)


def train(fsrs_params: jax.Array, users: list[int], data: Data):
    print("start train")
    train_split_lengths_cat = data.train_split_lengths
    num_training_steps_per_epoch_cat = (train_split_lengths_cat + BATCH_SIZE - 1) // BATCH_SIZE
    num_training_steps_cat = N_EPOCHS * num_training_steps_per_epoch_cat
    
    batch_perm_cat = _make_batch_permutation(num_training_steps_per_epoch_cat)
    # Map: (user, split) -> batch_perm_cat start index
    batch_perm_user_flat_offset = jnp.pad(
        jnp.cumsum(num_training_steps_cat, axis=-1)[:-1],
        (1, 0),
    ).reshape(len(users), N_SPLITS)
    # print(batch_perm_user_flat_offset)
    # print(batch_perm_cat, batch_perm_cat.size())
    # print(batch_perm_cat.cpu().tolist())
    # exit()

    # Map: (user, split) -> data.train_index start index
    train_split_lengths_offset = jnp.pad(
        jnp.cumsum(train_split_lengths_cat, axis=-1)[:-1],
        (1, 0),
    ).reshape(len(users), N_SPLITS)

    train_splits_length_cat_sum = int(jnp.sum(num_training_steps_cat).item())
    train_splits_length_cat_max = int(jnp.max(num_training_steps_cat).item())
    batch_num_inner_batches = ceil_div(train_splits_length_cat_sum, train_splits_length_cat_max)
    
    print("Inner batches per batch:", batch_num_inner_batches, "Train iters (max):", train_splits_length_cat_max)
    step_i_cat = jnp.zeros_like(num_training_steps_cat)
    optim_state = adamw.init_adamw_state(fsrs_params.reshape(-1, fsrs_params.shape[-1]))
    for iter in tqdm(range(train_splits_length_cat_max), desc="Training", smoothing=0.03):
        _sync_jax(step_i_cat)
        ts = time.time()
        remaining = num_training_steps_cat - step_i_cat
        # print("remaining:", remaining)
        active_count = int(jnp.sum(remaining > 0).item())
        _, indices = jax.lax.top_k(
            remaining,
            k=int(min(batch_num_inner_batches, active_count)),
        )
        step_i = step_i_cat[indices]
        # epoch_step_i = step_i % num_training_steps_per_epoch_cat[indices]
        # print(step_i, epoch_step_i)

        user_indices = indices // N_SPLITS
        split_indices = indices % N_SPLITS
        perm_i = batch_perm_cat[batch_perm_user_flat_offset[user_indices, split_indices] + step_i]
        _assert_all(perm_i <= num_training_steps_per_epoch_cat[indices])

        train_l = perm_i * BATCH_SIZE
        train_r = jnp.minimum((perm_i + 1) * BATCH_SIZE - 1, train_split_lengths_cat[indices] - 1)
        _assert_all(train_l <= train_r)

        indices_range = jnp.repeat(indices[:, None], BATCH_SIZE, axis=1)
        train_range = \
            + train_l[:, None] \
            + jnp.arange(BATCH_SIZE, dtype=train_l.dtype)[None, :]

        legal = train_range <= train_r[:, None]
        review_data_indices = data.train_index[
            jnp.minimum(
                train_split_lengths_offset[user_indices, split_indices][:, None] + train_range,
                data.train_index.shape[0] - 1,
            )
        ]
        # print(review_data_indices, review_data_indices.shape)
        # print(legal)
        # print(train_split_lengths_cat[indices] - 1)

        indices_flat = indices_range.reshape(-1)
        review_data_indices_flat = review_data_indices.reshape(-1)
        train_range_flat = train_range.reshape(-1)
        legal_flat = legal.reshape(-1)
        assert legal_flat.shape == indices_flat.shape

        indices_filtered = _mask_select(indices_flat, legal_flat)
        # train_range_filtered = train_range_flat[legal_flat]
        review_data_indices_filtered = _mask_select(review_data_indices_flat, legal_flat)
        # review_data_indices_filtered = data.train_index[train_split_lengths_offset[indices_filtered // N_SPLITS, indices_filtered % N_SPLITS].unsqueeze(-1) + train_range_filtered]
        # print(review_data_indices_filtered.shape)
        # exit()

        # Sort by seq len
        seq_lens = data.review_data.seq_len[review_data_indices_filtered]
        sorted_seq_lens_indices = jnp.argsort(seq_lens, stable=True)
        seq_lens = seq_lens[sorted_seq_lens_indices]
        indices_filtered = indices_filtered[sorted_seq_lens_indices]
        review_data_indices_filtered = review_data_indices_filtered[sorted_seq_lens_indices]

        _sync_jax(seq_lens, indices_filtered, review_data_indices_filtered)
        tb = time.time()
        batches = get_batches_train(seq_lens)
        print("get batches", time.time() - tb)
        _sync_jax(seq_lens)
        print("prepare iteration", time.time() - ts)
        # print(batches)
        flat_fsrs_params = fsrs_params.reshape(-1, fsrs_params.shape[-1])
        flat_grad = jnp.zeros_like(flat_fsrs_params)
        for (l, re) in batches:
            ts = time.time()
            # slice = torch.arange(l, re, re - l, device=indices.device)
            # batch_fsrs_params = torch.tensor(fsrs_params.view(-1, fsrs_params.size(-1))[indices_filtered[l:re]], requires_grad=True, device=indices.device)
            # batch_fsrs_params = fsrs_params.view(-1, fsrs_params.size(-1))[indices_filtered[l:re]].detach().clone().requires_grad_(True)
            batch_fsrs_params = flat_fsrs_params[indices_filtered[l:re]]
            # batch_review_data_indices = review_data_indices_filtered[slice]
            batch_review_data_indices = review_data_indices_filtered[l:re]
            _sync_jax(batch_fsrs_params, batch_review_data_indices)
            print("intermediate", time.time() - ts)
            p, loss, grad = predict_loss_grad(data.review_data, batch_review_data_indices, batch_fsrs_params)
            _sync_jax(p, loss, grad)
            print("intermediate2", time.time() - ts)
            # print("Jax adapter output:")
            # print(p)

            # batch_fsrs_params.grad = grad
            # print(batch_fsrs_params)
            flat_grad = flat_grad.at[indices_filtered[l:re]].add(grad)
            # print(fsrs_params.grad)
            # exit()
            _sync_jax(flat_grad)
            print("total it", time.time() - ts)
            print()

        t_opt = time.time()
        active_params_mask = jnp.zeros(step_i_cat.shape, dtype=jnp.bool_)
        active_params_mask = active_params_mask.at[indices].set(True)
        lr_schedule_multi = scheduler.scheduler(step_i_cat, num_training_steps_cat)

        flat_fsrs_params = fsrs_params.reshape(-1, fsrs_params.shape[-1])
        new_flat_fsrs_params, optim_state = adamw.adamw_step(
            flat_fsrs_params,
            flat_grad,
            optim_state,
            lr=lr_schedule_multi,
            mask=active_params_mask,
        )
        new_flat_fsrs_params_clipped = fsrs_v7_jax.apply_parameter_clipper(new_flat_fsrs_params)
        fsrs_params = new_flat_fsrs_params_clipped.reshape(fsrs_params.shape)

        step_i_cat = step_i_cat.at[indices].add(1)

        _sync_jax(fsrs_params, step_i_cat, optim_state.step)
        print("opt time", time.time() - t_opt)
        print("----------------------------------------------------------")


    _assert_all(step_i_cat == num_training_steps_cat)
    print("------------------done train-----------------")
    return fsrs_params


def evaluate_on_test_set(fsrs_params: jax.Array, users: list[int], data: Data):
    print("eval on test")
    param_keys = data.get_test_index_param_key()

    test_seq_len = data.review_data.seq_len[data.test_index]
    sorted_test_index_permutation = jnp.argsort(test_seq_len, stable=True)
    sorted_test_seq_len = test_seq_len[sorted_test_index_permutation]
    print("sort done")

    # TODO use the same load balancing as training
    N = data.test_index.shape[0]
    num_batches = ceil_div(N, TEST_BATCH_SIZE_MAX)
    batch_size = ceil_div(N, num_batches)
    gather_p = []
    batches = get_batches_test(sorted_test_seq_len)
    print(batches)
    for (l, re) in tqdm(batches, desc="Test set", smoothing=0.03):
    # for perm_slice in sorted_test_index_permutation.split(batch_size):
        perm_slice = sorted_test_index_permutation[l:re]
        batch_fsrs_params = fsrs_params[param_keys.user_index[perm_slice], param_keys.split_index[perm_slice]]
        p = predict(data.review_data, data.test_index[perm_slice], batch_fsrs_params)
        gather_p.append(p)
    
    restore_test_index_permutation = jnp.empty_like(sorted_test_index_permutation)
    restore_test_index_permutation = restore_test_index_permutation.at[sorted_test_index_permutation].set(
        jnp.arange(
            sorted_test_index_permutation.size,
            dtype=sorted_test_index_permutation.dtype,
        )
    )
    p_concat = jnp.concatenate(gather_p, axis=-1)[restore_test_index_permutation]
    del sorted_test_index_permutation
    del restore_test_index_permutation

    split_points = np.cumsum(data.test_index_lens)[:-1]
    p_by_user = np.split(np.asarray(jax.device_get(p_concat)), split_points)
    label = data.review_data.rating[data.test_index] > 1
    label_by_user = np.split(np.asarray(jax.device_get(label)), split_points)

    for user, pred, label in tqdm(zip(users, p_by_user, label_by_user), smoothing=0.03):
        logloss = log_loss(y_true=label, y_pred=pred, labels=[0, 1])
        print(f"User: {user}, logloss={logloss:.3f}")


def run(
    env: lmdb.Environment,
    users: list[int],
) -> None:
    with env.begin(write=False) as txn:
        ts = time.time()
        blobs = [
            load_user_blob(txn, user_id)
            for user_id in tqdm(users, total=len(users), smoothing=0.03, desc="Loading user data")
        ]
        print("load blobs", time.time() - ts)
        ts = time.time()
        data = Data(blobs)
        _sync_jax(
            data.review_data.rating,
            data.review_data.elapsed_days_real,
            data.review_data.seq_len,
            data.train_index,
            data.train_split_lengths,
            data.test_index,
        )
        print("build data", time.time() - ts)
        del blobs

    # fsrs_params = torch.zeros((len(users), N_SPLITS, 35), device=DEVICE)
    fsrs_params = jnp.tile(
        fsrs_v7_jax.get_initial_params_for_optimization().reshape(1, 1, -1),
        (len(users), N_SPLITS, 1),
    )
    fsrs_params = train(fsrs_params, users, data)

    print("skip test")
    # evaluate
    evaluate_on_test_set(fsrs_params, users, data)


def main() -> None:
    env = lmdb.open(
        str(LMDB_PATH),
        map_size=LMDB_SIZE,
        readonly=True,
        lock=False,
    )
    
    users = list(range(1, 1000))
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
