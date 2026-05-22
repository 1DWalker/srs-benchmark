from __future__ import annotations

from pathlib import Path

import jax
import torch
import time

from parallel.models import fsrs_v7_jax
from parallel.utils import _next_power_of_2


MIN_PADDED_BATCH_SIZE = 2**15
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


def _torch_to_jax(tensor: torch.Tensor) -> jax.Array:
    tensor = tensor.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    try:
        return jax.dlpack.from_dlpack(tensor, copy=False)
    except ValueError as exc:
        if "requires a copy" not in str(exc):
            raise
        return jax.dlpack.from_dlpack(tensor, copy=None)


def _jax_to_torch(array: jax.Array, like: torch.Tensor | None = None) -> torch.Tensor:
    return torch.from_dlpack(array, copy=False)


def _astype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype=dtype)


def _pad_rows_repeat_last(tensor: torch.Tensor, size: int) -> torch.Tensor:
    if tensor.shape[0] > size:
        raise ValueError(f"Cannot pad {tensor.shape[0]} rows down to {size}.")
    if tensor.shape[0] == size:
        return tensor.contiguous()

    fill = tensor[-1:].expand(size - tensor.shape[0], *tensor.shape[1:])
    return torch.cat((tensor, fill), dim=0).contiguous()


def _pad_cols_zero(tensor: torch.Tensor, size: int) -> torch.Tensor:
    if tensor.shape[1] > size:
        raise ValueError(f"Cannot pad {tensor.shape[1]} columns down to {size}.")
    if tensor.shape[1] == size:
        return tensor.contiguous()

    fill = tensor.new_zeros((tensor.shape[0], size - tensor.shape[1]))
    return torch.cat((tensor, fill), dim=1).contiguous()


def _prepare_prediction_batch(
    parameters_bp: torch.Tensor,
    feature_elapsed_days_real_bl: torch.Tensor,
    feature_rating_bl: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
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
        _astype(elapsed, torch.float32),
        _astype(rating, torch.int32),
        _astype(_pad_rows_repeat_last(seq_lens, padded_b), torch.int32),
        actual_size,
    )


def _prepare_batch(
    parameters_bp: torch.Tensor,
    feature_elapsed_days_real_bl: torch.Tensor,
    feature_rating_bl: torch.Tensor,
    seq_lens: torch.Tensor,
    epoch_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    *batch, actual_size = _prepare_prediction_batch(
        parameters_bp,
        feature_elapsed_days_real_bl,
        feature_rating_bl,
        seq_lens,
    )
    mask = feature_elapsed_days_real_bl.new_zeros(
        (batch[0].shape[0],),
        dtype=torch.float32,
    )
    mask[:actual_size] = 1.0
    epoch_lens = _astype(
        _pad_rows_repeat_last(epoch_lens, batch[0].shape[0]),
        torch.float32,
    )
    return (*batch, mask.contiguous(), epoch_lens, actual_size)


def _validate_inputs(
    parameters_bp: torch.Tensor,
    feature_elapsed_days_real_bl: torch.Tensor,
    feature_rating_bl: torch.Tensor,
    seq_lens: torch.Tensor,
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
        parameters_bp: torch.Tensor,
        feature_elapsed_days_real_bl: torch.Tensor,
        feature_rating_bl: torch.Tensor,
        seq_lens: torch.Tensor,
        epoch_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.cuda.synchronize()
        to = time.time()
        _validate_inputs(
            parameters_bp,
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            seq_lens,
        )
        if parameters_bp.shape[0] == 0:
            return (
                parameters_bp.new_empty((0,)),
                parameters_bp.new_zeros(()),
                parameters_bp.new_empty(parameters_bp.shape),
            )

        *batch, actual_size = _prepare_batch(
            parameters_bp,
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            seq_lens,
            epoch_lens,
        )
        torch.cuda.synchronize()
        t_before_move = time.time()
        jax_arrays = [_torch_to_jax(tensor) for tensor in batch]
        for array in jax_arrays:
            array.block_until_ready()
        ts = time.time()
        print("adapter start", time.time() - to, time.time() - t_before_move)
        (loss, prediction), parameters_grad = fsrs_v7_jax.loss_and_prediction_and_grad(
            *jax_arrays
        )

        prediction.block_until_ready()
        loss.block_until_ready()
        parameters_grad.block_until_ready()
        print(loss.shape)  # force sync?
        print("just func", time.time() - ts, time.time() - to, batch[1].shape)
        ts = time.time()
        prediction_b = _jax_to_torch(prediction)[:actual_size]
        loss_t = _jax_to_torch(loss, parameters_bp)
        parameters_grad_bp = _jax_to_torch(parameters_grad, parameters_bp)[:actual_size]
        torch.cuda.synchronize()
        print("tranfer back", time.time() - ts)
        return prediction_b, loss_t, parameters_grad_bp

    def prediction(
        self,
        parameters_bp: torch.Tensor,
        feature_elapsed_days_real_bl: torch.Tensor,
        feature_rating_bl: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        _validate_inputs(
            parameters_bp,
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            seq_lens,
        )
        if parameters_bp.shape[0] == 0:
            return parameters_bp.new_empty((0,))

        *batch, actual_size = _prepare_prediction_batch(
            parameters_bp,
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            seq_lens,
        )
        prediction = fsrs_v7_jax.forward(*(_torch_to_jax(tensor) for tensor in batch))
        return _jax_to_torch(prediction)[:actual_size]

_DEFAULT_ADAPTER = FSRS7JaxAdapter()

def prediction(
    parameters_bp: torch.Tensor,
    feature_elapsed_days_real_bl: torch.Tensor,
    feature_rating_bl: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    return _DEFAULT_ADAPTER.prediction(
        parameters_bp,
        feature_elapsed_days_real_bl,
        feature_rating_bl,
        seq_lens,
    )

def prediction_loss_grad(
    parameters_bp: torch.Tensor,
    feature_elapsed_days_real_bl: torch.Tensor,
    feature_rating_bl: torch.Tensor,
    seq_lens: torch.Tensor,
    epoch_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _DEFAULT_ADAPTER.prediction_loss_grad(
        parameters_bp,
        feature_elapsed_days_real_bl,
        feature_rating_bl,
        seq_lens,
        epoch_lens,
    )
