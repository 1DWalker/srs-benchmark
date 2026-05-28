from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import shutil
from pathlib import Path

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from parallel.config import (
    BATCH_PERM_SEED,
    BATCH_SIZE,
    N_EPOCHS,
    N_SPLITS,
    TENSOR_CACHE_PATH,
    TENSOR_CACHE_SIZE,
    TENSOR_CACHE_VERSION,
)
from parallel.tensor_lmdb import (
    get_array,
    get_tensor,
    get_tensor_meta,
    put_array_to_env,
    user_tensor_prefix,
)
from parallel.tensors import Data, ReviewData


@dataclass(frozen=True)
class TrainSetup:
    num_training_steps_per_epoch_cat: torch.Tensor
    num_training_steps_cat: torch.Tensor
    batch_perm_cat: torch.Tensor
    batch_perm_user_flat_offset: torch.Tensor
    train_split_lengths_offset: torch.Tensor
    split_review_ord: torch.Tensor
    batch_num_inner_batches: int


@dataclass(frozen=True)
class _SourceUserInfo:
    user_id: int
    review_len: int
    train_len: int
    test_len: int
    split_len: int
    train_split_len: int


_MANIFEST_KEY = b"manifest"
_TRAIN_SETUP_MANIFEST_KEY = b"train_setup_manifest"
_BATCH_PERM_MANIFEST_KEY = b"batch_perm_manifest"


def _cache_tensor_prefix(split_i: int, name: str) -> str:
    return f"split:{split_i}:tensor:{name}"


def _cache_json_key(split_i: int, name: str) -> bytes:
    return f"split:{split_i}:json:{name}".encode()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def build_user_batch_perm(
    user_id: int,
    num_training_steps_per_epoch: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    if seed is None:
        seed = BATCH_PERM_SEED
    steps = np.asarray(num_training_steps_per_epoch, dtype=np.int32).reshape(-1)
    out = np.empty(int(steps.sum()) * N_EPOCHS, dtype=np.int32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + int(user_id))

    offset = 0
    for steps_per_epoch in steps:
        n = int(steps_per_epoch)
        for _ in range(N_EPOCHS):
            next_offset = offset + n
            if n > 0:
                out[offset:next_offset] = torch.randperm(n, generator=generator).numpy()
            offset = next_offset

    return out


def build_batch_perm_cat_for_users(
    user_ids: Sequence[int],
    num_training_steps_per_epoch: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    if seed is None:
        seed = BATCH_PERM_SEED
    steps = np.asarray(num_training_steps_per_epoch, dtype=np.int32).reshape(-1)
    if len(user_ids) == 0:
        return np.empty(0, dtype=np.int32)

    steps_by_user = steps.reshape(len(user_ids), N_SPLITS)
    out = np.empty(int(steps.sum()) * N_EPOCHS, dtype=np.int32)

    offset = 0
    for user_id, user_steps in zip(user_ids, steps_by_user, strict=True):
        user_perm = build_user_batch_perm(int(user_id), user_steps, seed)
        next_offset = offset + user_perm.size
        out[offset:next_offset] = user_perm
        offset = next_offset

    return out


def _open_cache_env(
    cache_path: Path = TENSOR_CACHE_PATH,
    map_size: int = TENSOR_CACHE_SIZE,
) -> lmdb.Environment:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return lmdb.open(
        str(cache_path),
        map_size=map_size,
        writemap=True,
        subdir=True,
    )


def _expected_manifest(user_splits: list[list[int]]) -> dict[str, object]:
    return {
        "version": TENSOR_CACHE_VERSION,
        "user_splits": [[int(user_id) for user_id in split] for split in user_splits],
        "n_splits": N_SPLITS,
    }


def _expected_train_setup_manifest(user_splits: list[list[int]]) -> dict[str, object]:
    return {
        "version": TENSOR_CACHE_VERSION,
        "user_splits": [[int(user_id) for user_id in split] for split in user_splits],
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "n_splits": N_SPLITS,
    }


def _expected_batch_perm_manifest(user_splits: list[list[int]]) -> dict[str, object]:
    return {
        "version": TENSOR_CACHE_VERSION,
        "batch_perm_seed": BATCH_PERM_SEED,
        "user_splits": [[int(user_id) for user_id in split] for split in user_splits],
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "n_splits": N_SPLITS,
    }


def _read_manifest(cache_env: lmdb.Environment) -> dict[str, object] | None:
    with cache_env.begin(write=False) as txn:
        raw = txn.get(_MANIFEST_KEY)
    if raw is None:
        return None
    return json.loads(raw.decode())


def _main_manifest_matches(actual: dict[str, object] | None, expected: dict[str, object]) -> bool:
    if actual is None:
        return False
    actual = dict(actual)
    actual.pop("n_epochs", None)
    actual.pop("batch_size", None)
    return actual == expected


def _write_manifest(cache_env: lmdb.Environment, manifest: dict[str, object]) -> None:
    with cache_env.begin(write=True) as txn:
        txn.put(_MANIFEST_KEY, json.dumps(manifest, separators=(",", ":")).encode())


def _read_train_setup_manifest(cache_env: lmdb.Environment) -> dict[str, object] | None:
    with cache_env.begin(write=False) as txn:
        raw = txn.get(_TRAIN_SETUP_MANIFEST_KEY)
    if raw is None:
        return None
    return json.loads(raw.decode())


def _write_train_setup_manifest(cache_env: lmdb.Environment, manifest: dict[str, object]) -> None:
    with cache_env.begin(write=True) as txn:
        txn.put(_TRAIN_SETUP_MANIFEST_KEY, json.dumps(manifest, separators=(",", ":")).encode())


def _read_batch_perm_manifest(cache_env: lmdb.Environment) -> dict[str, object] | None:
    with cache_env.begin(write=False) as txn:
        raw = txn.get(_BATCH_PERM_MANIFEST_KEY)
    if raw is None:
        return None
    return json.loads(raw.decode())


def _write_batch_perm_manifest(cache_env: lmdb.Environment, manifest: dict[str, object]) -> None:
    with cache_env.begin(write=True) as txn:
        txn.put(_BATCH_PERM_MANIFEST_KEY, json.dumps(manifest, separators=(",", ":")).encode())


def _clear_cache_path(cache_path: Path) -> None:
    if cache_path.exists():
        shutil.rmtree(cache_path)


def load_or_rebuild_tensor_cache(
    source_env: lmdb.Environment,
    user_splits: list[list[int]],
    cache_path: Path = TENSOR_CACHE_PATH,
    map_size: int = TENSOR_CACHE_SIZE,
) -> lmdb.Environment:
    expected = _expected_manifest(user_splits)
    cache_env = _open_cache_env(cache_path, map_size)
    if _main_manifest_matches(_read_manifest(cache_env), expected):
        _ensure_train_setup_cache(cache_env, user_splits)
        _ensure_batch_perm_cache(cache_env, user_splits)
        return cache_env

    print("tensor cache miss; rebuilding")
    cache_env.close()
    _clear_cache_path(cache_path)
    cache_env = _open_cache_env(cache_path, map_size)
    _rebuild_tensor_cache(source_env, cache_env, user_splits)
    _write_manifest(cache_env, expected)
    _ensure_train_setup_cache(cache_env, user_splits)
    _ensure_batch_perm_cache(cache_env, user_splits)
    return cache_env


def load_cached_split(
    cache_env: lmdb.Environment,
    split_i: int,
    device: torch.device | str,
) -> tuple[Data, TrainSetup]:
    device = torch.device(device)
    review_data = load_cached_review_data(cache_env, split_i, device)
    data, setup = load_cached_train_only(cache_env, split_i, device, review_data)
    test_data = load_cached_test_only(cache_env, split_i, device, review_data)
    with cache_env.begin(write=False, buffers=True) as txn:
        data.user_flat_offset = get_tensor(txn, _cache_tensor_prefix(split_i, "user_flat_offset"), device)
    data.test_index = test_data.test_index
    data.rmse_bins = test_data.rmse_bins
    data.splits = test_data.splits
    data.split_counts = test_data.split_counts
    data.test_index_lens = test_data.test_index_lens
    return data, setup


def load_cached_review_data(
    cache_env: lmdb.Environment,
    split_i: int,
    device: torch.device | str,
) -> ReviewData:
    device = torch.device(device)
    with cache_env.begin(write=False, buffers=True) as txn:
        return ReviewData(
            rating=get_tensor(txn, _cache_tensor_prefix(split_i, "rating"), device),
            elapsed_days_real=get_tensor(txn, _cache_tensor_prefix(split_i, "elapsed_days_real"), device),
            seq_len=get_tensor(txn, _cache_tensor_prefix(split_i, "seq_len"), device),
        )


def _new_partial_data(review_data: ReviewData) -> Data:
    data = object.__new__(Data)
    data.review_data = review_data
    data.device = review_data.rating.device
    return data


def load_cached_train_only(
    cache_env: lmdb.Environment,
    split_i: int,
    device: torch.device | str,
    review_data: ReviewData,
) -> tuple[Data, TrainSetup]:
    device = torch.device(device)
    with cache_env.begin(write=False, buffers=True) as txn:
        data = _new_partial_data(review_data)
        data.train_index = get_tensor(txn, _cache_tensor_prefix(split_i, "train_index"), device)
        data.train_split_lengths = get_tensor(txn, _cache_tensor_prefix(split_i, "train_split_lengths"), device)

        setup = TrainSetup(
            num_training_steps_per_epoch_cat=get_tensor(
                txn,
                _cache_tensor_prefix(split_i, "num_training_steps_per_epoch_cat"),
                device,
            ),
            num_training_steps_cat=get_tensor(
                txn,
                _cache_tensor_prefix(split_i, "num_training_steps_cat"),
                device,
            ),
            batch_perm_cat=get_tensor(txn, _cache_tensor_prefix(split_i, "batch_perm_cat"), device),
            batch_perm_user_flat_offset=get_tensor(
                txn,
                _cache_tensor_prefix(split_i, "batch_perm_user_flat_offset"),
                device,
            ),
            train_split_lengths_offset=get_tensor(
                txn,
                _cache_tensor_prefix(split_i, "train_split_lengths_offset"),
                device,
            ),
            split_review_ord=get_tensor(txn, _cache_tensor_prefix(split_i, "split_review_ord"), device),
            batch_num_inner_batches=int(
                json.loads(bytes(txn.get(_cache_json_key(split_i, "batch_num_inner_batches"))).decode())
            ),
        )
    return data, setup


def load_cached_test_only(
    cache_env: lmdb.Environment,
    split_i: int,
    device: torch.device | str,
    review_data: ReviewData,
    load_rmse_bins: bool = True,
) -> Data:
    device = torch.device(device)
    with cache_env.begin(write=False, buffers=True) as txn:
        data = _new_partial_data(review_data)
        data.test_index = get_tensor(txn, _cache_tensor_prefix(split_i, "test_index"), device)
        if load_rmse_bins:
            data.rmse_bins = get_tensor(txn, _cache_tensor_prefix(split_i, "rmse_bins"), device)
        data.splits = get_tensor(txn, _cache_tensor_prefix(split_i, "splits"), device)
        data.split_counts = get_tensor(txn, _cache_tensor_prefix(split_i, "split_counts"), device)
        data.test_index_lens = get_tensor(
            txn,
            _cache_tensor_prefix(split_i, "test_index_lens"),
            device,
        )
    return data


def _rebuild_tensor_cache(
    source_env: lmdb.Environment,
    cache_env: lmdb.Environment,
    user_splits: list[list[int]],
) -> None:
    build_steps = [
        ("rating", lambda split_i, infos: _build_review_field(source_env, cache_env, split_i, infos, "rating")),
        (
            "elapsed_days_real",
            lambda split_i, infos: _build_review_field(
                source_env,
                cache_env,
                split_i,
                infos,
                "elapsed_days_real",
            ),
        ),
        ("seq_len", lambda split_i, infos: _build_review_field(source_env, cache_env, split_i, infos, "seq_len")),
        (
            "train_index",
            lambda split_i, infos: _build_offset_index_field(
                source_env,
                cache_env,
                split_i,
                infos,
                "train_index",
                "train_index",
            ),
        ),
        (
            "split_review_ord",
            lambda split_i, infos: _build_flat_field(
                source_env,
                cache_env,
                split_i,
                infos,
                "split_review_ord",
                "split_review_ord",
            ),
        ),
        (
            "train_split_lengths",
            lambda split_i, infos: _build_train_split_lengths(source_env, cache_env, split_i, infos),
        ),
        (
            "test_index",
            lambda split_i, infos: _build_offset_index_field(
                source_env,
                cache_env,
                split_i,
                infos,
                "test_index",
                "test_index",
            ),
        ),
        (
            "rmse_bins",
            lambda split_i, infos: _build_flat_field(
                source_env,
                cache_env,
                split_i,
                infos,
                "rmse_bins",
                "rmse_bins",
            ),
        ),
        (
            "splits",
            lambda split_i, infos: _build_flat_field(source_env, cache_env, split_i, infos, "split", "splits"),
        ),
        ("derived", lambda split_i, infos: _build_small_derived_tensors(cache_env, split_i, infos)),
    ]

    for split_i, users in enumerate(tqdm(user_splits, desc="Tensor cache splits", smoothing=0.03)):
        tqdm.write(f"rebuilding tensor cache split {split_i + 1}/{len(user_splits)} ({len(users)} users)")
        with source_env.begin(write=False) as source_txn:
            infos = [
                _read_user_info(source_txn, user_id)
                for user_id in users
            ]

        step_progress = tqdm(
            build_steps,
            desc=f"Split {split_i + 1} tensors",
            leave=False,
            smoothing=0.03,
        )
        for step_name, build_step in step_progress:
            step_progress.set_postfix_str(step_name)
            build_step(split_i, infos)


def _ensure_train_setup_cache(
    cache_env: lmdb.Environment,
    user_splits: list[list[int]],
) -> None:
    expected = _expected_train_setup_manifest(user_splits)
    if _read_train_setup_manifest(cache_env) == expected:
        return

    print("train setup cache miss; rebuilding")
    for split_i, _ in enumerate(user_splits):
        _build_train_setup(cache_env, split_i)
    _write_train_setup_manifest(cache_env, expected)


def _ensure_batch_perm_cache(
    cache_env: lmdb.Environment,
    user_splits: list[list[int]],
) -> None:
    expected = _expected_batch_perm_manifest(user_splits)
    if _read_batch_perm_manifest(cache_env) == expected:
        return

    print("batch perm cache miss; rebuilding")
    for split_i, users in enumerate(user_splits):
        with cache_env.begin(write=False, buffers=True) as cache_txn:
            num_training_steps_per_epoch = get_array(
                cache_txn,
                _cache_tensor_prefix(split_i, "num_training_steps_per_epoch_cat"),
            ).astype(np.int32, copy=True)
            num_training_steps = get_array(
                cache_txn,
                _cache_tensor_prefix(split_i, "num_training_steps_cat"),
            ).astype(np.int32, copy=True)
        _build_batch_perm_cat(
            cache_env,
            split_i,
            users,
            num_training_steps_per_epoch,
            num_training_steps,
            BATCH_PERM_SEED,
        )
    _write_batch_perm_manifest(cache_env, expected)


def _read_user_info(txn: lmdb.Transaction, user_id: int) -> _SourceUserInfo:
    return _SourceUserInfo(
        user_id=user_id,
        review_len=_numel(get_tensor_meta(txn, user_tensor_prefix(user_id, "rating")).shape),
        train_len=_numel(get_tensor_meta(txn, user_tensor_prefix(user_id, "train_index")).shape),
        test_len=_numel(get_tensor_meta(txn, user_tensor_prefix(user_id, "test_index")).shape),
        split_len=_numel(get_tensor_meta(txn, user_tensor_prefix(user_id, "split")).shape),
        train_split_len=_numel(get_tensor_meta(txn, user_tensor_prefix(user_id, "train_split_lengths")).shape),
    )


def _put_cache_array(cache_env: lmdb.Environment, split_i: int, name: str, array: np.ndarray) -> None:
    def on_chunk_write(chunk_i: int, chunk_count: int, chunk_bytes: int) -> None:
        pass

    put_array_to_env(
        cache_env,
        _cache_tensor_prefix(split_i, name),
        array,
        on_chunk_write=on_chunk_write,
    )


def _build_review_field(
    source_env: lmdb.Environment,
    cache_env: lmdb.Environment,
    split_i: int,
    infos: list[_SourceUserInfo],
    field: str,
) -> None:
    total = sum(info.review_len for info in infos)
    with source_env.begin(write=False, buffers=True) as source_txn:
        dtype = get_array(source_txn, user_tensor_prefix(infos[0].user_id, field)).dtype if infos else np.int8
        out = np.empty(total, dtype=dtype)
        offset = 0
        for info in infos:
            source = get_array(source_txn, user_tensor_prefix(info.user_id, field)).reshape(-1)
            next_offset = offset + source.size
            out[offset:next_offset] = source
            offset = next_offset
    _put_cache_array(cache_env, split_i, field, out)
    del out


def _build_flat_field(
    source_env: lmdb.Environment,
    cache_env: lmdb.Environment,
    split_i: int,
    infos: list[_SourceUserInfo],
    source_field: str,
    cache_name: str,
) -> None:
    total = sum(_numel_for_source_field(info, source_field) for info in infos)
    with source_env.begin(write=False, buffers=True) as source_txn:
        dtype = get_array(source_txn, user_tensor_prefix(infos[0].user_id, source_field)).dtype if infos else np.int32
        out = np.empty(total, dtype=dtype)
        offset = 0
        for info in infos:
            source = get_array(source_txn, user_tensor_prefix(info.user_id, source_field)).reshape(-1)
            next_offset = offset + source.size
            out[offset:next_offset] = source
            offset = next_offset
    _put_cache_array(cache_env, split_i, cache_name, out)
    del out


def _numel_for_source_field(info: _SourceUserInfo, field: str) -> int:
    if field in ("train_index", "split_review_ord"):
        return info.train_len
    if field in ("test_index", "rmse_bins"):
        return info.test_len
    if field == "split":
        return info.split_len
    if field == "train_split_lengths":
        return info.train_split_len
    return info.review_len


def _build_offset_index_field(
    source_env: lmdb.Environment,
    cache_env: lmdb.Environment,
    split_i: int,
    infos: list[_SourceUserInfo],
    source_field: str,
    cache_name: str,
) -> None:
    total = sum(_numel_for_source_field(info, source_field) for info in infos)
    out = np.empty(total, dtype=np.int32)
    review_offset = 0
    write_offset = 0
    with source_env.begin(write=False, buffers=True) as source_txn:
        for info in infos:
            source = get_array(source_txn, user_tensor_prefix(info.user_id, source_field)).reshape(-1)
            next_write_offset = write_offset + source.size
            np.add(source, review_offset, out=out[write_offset:next_write_offset], casting="unsafe")
            write_offset = next_write_offset
            review_offset += info.review_len
    _put_cache_array(cache_env, split_i, cache_name, out)
    del out


def _build_train_split_lengths(
    source_env: lmdb.Environment,
    cache_env: lmdb.Environment,
    split_i: int,
    infos: list[_SourceUserInfo],
) -> None:
    _build_flat_field(
        source_env,
        cache_env,
        split_i,
        infos,
        "train_split_lengths",
        "train_split_lengths",
    )


def _build_small_derived_tensors(
    cache_env: lmdb.Environment,
    split_i: int,
    infos: list[_SourceUserInfo],
) -> None:
    split_counts = np.array([info.split_len for info in infos], dtype=np.int32)
    test_index_lens = np.array([info.test_len for info in infos], dtype=np.int32)
    user_lengths = np.array([info.review_len for info in infos], dtype=np.int32)
    if user_lengths.size == 0:
        user_flat_offset = np.empty(0, dtype=np.int32)
    else:
        user_flat_offset = np.empty_like(user_lengths)
        user_flat_offset[0] = 0
        if user_lengths.size > 1:
            user_flat_offset[1:] = np.cumsum(user_lengths[:-1], dtype=np.int32)

    _put_cache_array(cache_env, split_i, "split_counts", split_counts)
    del split_counts
    _put_cache_array(cache_env, split_i, "test_index_lens", test_index_lens)
    del test_index_lens
    _put_cache_array(cache_env, split_i, "user_flat_offset", user_flat_offset)
    del user_flat_offset


def _build_train_setup(
    cache_env: lmdb.Environment,
    split_i: int,
) -> None:
    with cache_env.begin(write=False, buffers=True) as cache_txn:
        train_split_lengths = get_array(
            cache_txn,
            _cache_tensor_prefix(split_i, "train_split_lengths"),
        ).astype(np.int32, copy=True)

    num_training_steps_per_epoch = (
        (train_split_lengths + BATCH_SIZE - 1) // BATCH_SIZE
    ).astype(np.int32, copy=False)
    num_training_steps = (N_EPOCHS * num_training_steps_per_epoch).astype(np.int32, copy=False)

    _put_cache_array(cache_env, split_i, "num_training_steps_per_epoch_cat", num_training_steps_per_epoch)
    _put_cache_array(cache_env, split_i, "num_training_steps_cat", num_training_steps)

    batch_perm_user_flat_offset = _offsets_from_lengths(num_training_steps)
    _put_cache_array(
        cache_env,
        split_i,
        "batch_perm_user_flat_offset",
        batch_perm_user_flat_offset.reshape(-1, N_SPLITS),
    )
    del batch_perm_user_flat_offset

    train_split_lengths_offset = _offsets_from_lengths(train_split_lengths)
    _put_cache_array(
        cache_env,
        split_i,
        "train_split_lengths_offset",
        train_split_lengths_offset.reshape(-1, N_SPLITS),
    )
    del train_split_lengths_offset

    batch_num_inner_batches = _batch_num_inner_batches(num_training_steps)
    with cache_env.begin(write=True) as cache_txn:
        cache_txn.put(
            _cache_json_key(split_i, "batch_num_inner_batches"),
            json.dumps(batch_num_inner_batches).encode(),
        )

    del train_split_lengths
    del num_training_steps_per_epoch
    del num_training_steps


def _offsets_from_lengths(lengths: np.ndarray) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int32)
    offsets = np.empty_like(lengths, dtype=np.int32)
    if lengths.size == 0:
        return offsets
    offsets[0] = 0
    if lengths.size > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int32)
    return offsets


def _batch_num_inner_batches(num_training_steps: np.ndarray) -> int:
    total = int(num_training_steps.sum())
    max_steps = int(num_training_steps.max()) if num_training_steps.size else 0
    if max_steps == 0:
        return 0
    return _ceil_div(total, max_steps)


def _build_batch_perm_cat(
    cache_env: lmdb.Environment,
    split_i: int,
    user_ids: Sequence[int],
    num_training_steps_per_epoch: np.ndarray,
    num_training_steps: np.ndarray,
    seed: int,
) -> None:
    out = build_batch_perm_cat_for_users(user_ids, num_training_steps_per_epoch, seed)
    assert out.size == int(num_training_steps.sum())
    _put_cache_array(cache_env, split_i, "batch_perm_cat", out)
    del out
