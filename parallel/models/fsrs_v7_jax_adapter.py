from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import time

from parallel.models import fsrs_v7_jax
from parallel.utils import _next_power_of_2


MIN_PADDED_BATCH_SIZE = 2**16
DEFAULT_CACHE_DIR = Path(".jax-cache") / "fsrs_v7"


def _configure_compilation_cache(cache_dir: Path | None) -> None:
    if cache_dir is None:
        return
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


def _sync_jax(*arrays: jax.Array) -> None:
    if arrays:
        jax.block_until_ready(arrays)


def _astype(tensor: jax.Array, dtype: jnp.dtype) -> jax.Array:
    if tensor.dtype == dtype:
        return tensor
    return tensor.astype(dtype)


def _pad_rows_repeat_last(tensor: jax.Array, size: int) -> jax.Array:
    if tensor.shape[0] > size:
        raise ValueError(f"Cannot pad {tensor.shape[0]} rows down to {size}.")
    if tensor.shape[0] == size:
        return tensor

    fill_shape = (size - tensor.shape[0], *tensor.shape[1:])
    fill = jnp.broadcast_to(tensor[-1:], fill_shape)
    return jnp.concatenate((tensor, fill), axis=0)


def _pad_cols_zero(tensor: jax.Array, size: int) -> jax.Array:
    if tensor.shape[1] > size:
        raise ValueError(f"Cannot pad {tensor.shape[1]} columns down to {size}.")
    if tensor.shape[1] == size:
        return tensor

    fill = jnp.zeros((tensor.shape[0], size - tensor.shape[1]), dtype=tensor.dtype)
    return jnp.concatenate((tensor, fill), axis=1)


def _prepare_prediction_batch(
    parameters_bp: jax.Array,
    feature_elapsed_days_real_bl: jax.Array,
    feature_rating_bl: jax.Array,
    seq_lens: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, int]:
    actual_size = parameters_bp.shape[0]
    padded_b = max(MIN_PADDED_BATCH_SIZE, _next_power_of_2(actual_size))
    padded_l = 1 + _next_power_of_2(feature_elapsed_days_real_bl.shape[1] - 1)

    elapsed = _pad_rows_repeat_last(
        _pad_cols_zero(feature_elapsed_days_real_bl, padded_l),
        padded_b,
    )
    rating = _pad_rows_repeat_last(
        _pad_cols_zero(feature_rating_bl, padded_l),
        padded_b,
    )

    return (
        _pad_rows_repeat_last(parameters_bp, padded_b),
        _astype(elapsed, jnp.float32),
        _astype(rating, jnp.int32),
        _astype(_pad_rows_repeat_last(seq_lens, padded_b), jnp.int32),
        actual_size,
    )


def _prepare_batch(
    parameters_bp: jax.Array,
    feature_elapsed_days_real_bl: jax.Array,
    feature_rating_bl: jax.Array,
    seq_lens: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int]:
    *batch, actual_size = _prepare_prediction_batch(
        parameters_bp,
        feature_elapsed_days_real_bl,
        feature_rating_bl,
        seq_lens,
    )
    mask = jnp.zeros((batch[0].shape[0],), dtype=jnp.float32)
    mask = mask.at[:actual_size].set(1.0)
    return (*batch, mask, actual_size)


def _validate_inputs(
    parameters_bp: jax.Array,
    feature_elapsed_days_real_bl: jax.Array,
    feature_rating_bl: jax.Array,
    seq_lens: jax.Array,
) -> None:
    if parameters_bp.ndim != 2 or parameters_bp.shape[1] != 35:
        raise ValueError("parameters_bp must have shape (B, 35).")
    if feature_elapsed_days_real_bl.ndim != 2 or feature_rating_bl.ndim != 2:
        raise ValueError("feature tensors must have shape (B, L).")
    if feature_elapsed_days_real_bl.shape != feature_rating_bl.shape:
        raise ValueError("feature elapsed and rating tensors must have the same shape.")
    if parameters_bp.shape[0] != feature_elapsed_days_real_bl.shape[0]:
        raise ValueError("parameters and feature tensors must have the same B.")
    if seq_lens.shape != (parameters_bp.shape[0],):
        raise ValueError("seq_lens must have shape (B,).")


class FSRS7JaxAdapter:
    def __init__(
        self,
        batch_size: int | None = None,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
    ) -> None:
        if batch_size is not None and batch_size != MIN_PADDED_BATCH_SIZE:
            raise ValueError(f"batch_size must be {MIN_PADDED_BATCH_SIZE}.")
        _configure_compilation_cache(Path(cache_dir) if cache_dir is not None else None)

    def prediction_loss_grad(
        self,
        parameters_bp: jax.Array,
        feature_elapsed_days_real_bl: jax.Array,
        feature_rating_bl: jax.Array,
        seq_lens: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        _sync_jax(parameters_bp, feature_elapsed_days_real_bl, feature_rating_bl, seq_lens)
        to = time.time()
        _validate_inputs(
            parameters_bp,
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            seq_lens,
        )
        if parameters_bp.shape[0] == 0:
            return (
                jnp.empty((0,), dtype=parameters_bp.dtype),
                jnp.zeros((), dtype=parameters_bp.dtype),
                jnp.empty(parameters_bp.shape, dtype=parameters_bp.dtype),
            )

        *batch, actual_size = _prepare_batch(
            parameters_bp,
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            seq_lens,
        )
        ts = time.time()
        _sync_jax(*batch)
        print("adapter start", time.time() - to)
        (loss, prediction), parameters_grad = fsrs_v7_jax.loss_and_prediction_and_grad(
            *batch
        )

        _sync_jax(prediction, loss, parameters_grad)
        print(loss.shape)  # force sync?
        print("just func", time.time() - ts, time.time() - to, batch[1].shape, prediction.shape)
        # exit()
        return prediction[:actual_size], loss, parameters_grad[:actual_size]

    def prediction(
        self,
        parameters_bp: jax.Array,
        feature_elapsed_days_real_bl: jax.Array,
        feature_rating_bl: jax.Array,
        seq_lens: jax.Array,
    ) -> jax.Array:
        _validate_inputs(
            parameters_bp,
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            seq_lens,
        )
        if parameters_bp.shape[0] == 0:
            return jnp.empty((0,), dtype=parameters_bp.dtype)

        *batch, actual_size = _prepare_prediction_batch(
            parameters_bp,
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            seq_lens,
        )
        prediction = fsrs_v7_jax.forward(*batch)
        return prediction[:actual_size]

_DEFAULT_ADAPTER = FSRS7JaxAdapter()

def prediction(
    parameters_bp: jax.Array,
    feature_elapsed_days_real_bl: jax.Array,
    feature_rating_bl: jax.Array,
    seq_lens: jax.Array,
) -> jax.Array:
    return _DEFAULT_ADAPTER.prediction(
        parameters_bp,
        feature_elapsed_days_real_bl,
        feature_rating_bl,
        seq_lens,
    )

def prediction_loss_grad(
    parameters_bp: jax.Array,
    feature_elapsed_days_real_bl: jax.Array,
    feature_rating_bl: jax.Array,
    seq_lens: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _DEFAULT_ADAPTER.prediction_loss_grad(
        parameters_bp,
        feature_elapsed_days_real_bl,
        feature_rating_bl,
        seq_lens,
    )
