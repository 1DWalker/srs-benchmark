from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import torch

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

from parallel.models import fsrs_v7 as torch_fsrs_v7
from parallel.models import fsrs_v7_jax


DEFAULT_BATCH_SIZE = 2**16
DEFAULT_CACHE_DIR = Path(".jax-cache") / "fsrs_v7"

FSRS7_DEFAULT_35 = torch_fsrs_v7.FSRS7_DEFAULT_35
FSRS_MIN = torch_fsrs_v7.FSRS_MIN
FSRS_MAX = torch_fsrs_v7.FSRS_MAX
get_initial_params_for_optimization = torch_fsrs_v7.get_initial_params_for_optimization
nn_vec_to_fsrs7_params = torch_fsrs_v7.nn_vec_to_fsrs7_params


def _is_power_of_2(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _next_power_of_2(value: int) -> int:
    if value < 1:
        raise ValueError("seq_lens values must be at least 1.")
    return 1 << (value - 1).bit_length()


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
    tensor = torch.from_dlpack(array)
    if like is None:
        return tensor
    if tensor.device != like.device or tensor.dtype != like.dtype:
        tensor = tensor.to(device=like.device, dtype=like.dtype)
    return tensor


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
        return tensor[:, :size].contiguous()
    if tensor.shape[1] == size:
        return tensor.contiguous()

    fill = tensor.new_zeros((tensor.shape[0], size - tensor.shape[1]))
    return torch.cat((tensor, fill), dim=1).contiguous()


def _iter_power2_batches(
    seq_lens: torch.Tensor,
    batch_size: int,
) -> Iterator[tuple[int, int, int]]:
    n_rows = seq_lens.numel()
    start = 0
    while start < n_rows:
        seq_len = int(seq_lens[start].item())
        l_pad = _next_power_of_2(seq_len)
        bucket_end = int(
            torch.searchsorted(
                seq_lens,
                seq_lens.new_tensor(l_pad + 1),
                right=False,
            ).item()
        )
        if bucket_end <= start:
            raise ValueError("seq_lens must be sorted in ascending order.")

        while start < bucket_end:
            end = min(start + batch_size, bucket_end)
            yield start, end, l_pad
            start = end


def _prepare_batch(
    parameters_bp: torch.Tensor,
    feature_elapsed_days_real_bl: torch.Tensor,
    feature_rating_bl: torch.Tensor,
    seq_lens: torch.Tensor,
    start: int,
    end: int,
    batch_size: int,
    l_pad: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actual_size = end - start

    parameters = _pad_rows_repeat_last(parameters_bp[start:end], batch_size)
    elapsed = _pad_cols_zero(feature_elapsed_days_real_bl[start:end], l_pad)
    rating = _pad_cols_zero(feature_rating_bl[start:end], l_pad)
    lens = seq_lens[start:end]

    elapsed = _pad_rows_repeat_last(elapsed, batch_size)
    rating = _pad_rows_repeat_last(rating, batch_size)
    lens = _pad_rows_repeat_last(lens, batch_size)

    mask = feature_elapsed_days_real_bl.new_zeros((batch_size,), dtype=torch.float32)
    mask[:actual_size] = 1.0

    return (
        parameters.contiguous(),
        _astype(elapsed, torch.float32).contiguous(),
        _astype(rating, torch.int32).contiguous(),
        _astype(lens, torch.int32).contiguous(),
        mask.contiguous(),
    )


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
    if seq_lens.numel() and not bool(torch.all(seq_lens[:-1] <= seq_lens[1:]).item()):
        raise ValueError("seq_lens must be sorted in ascending order.")


class FSRS7JaxAdapter:
    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
    ) -> None:
        if not _is_power_of_2(batch_size):
            raise ValueError("batch_size must be a power of 2.")
        self.batch_size = batch_size
        _configure_compilation_cache(Path(cache_dir) if cache_dir is not None else None)

    def forward(
        self,
        parameters_bp: torch.Tensor,
        feature_elapsed_days_real_bl: torch.Tensor,
        feature_rating_bl: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        seq_lens = _astype(seq_lens, torch.int32)
        predictions: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        parameter_grads: list[torch.Tensor] = []

        for start, end, l_pad in _iter_power2_batches(seq_lens, self.batch_size):
            batch = _prepare_batch(
                parameters_bp,
                feature_elapsed_days_real_bl,
                feature_rating_bl,
                seq_lens,
                start,
                end,
                self.batch_size,
                l_pad,
            )
            (loss, prediction), parameters_grad = fsrs_v7_jax.loss_and_prediction_and_grad(
                *(_torch_to_jax(tensor) for tensor in batch)
            )

            actual_size = end - start
            predictions.append(_jax_to_torch(prediction)[:actual_size])
            losses.append(_jax_to_torch(loss, parameters_bp))
            parameter_grads.append(
                _jax_to_torch(parameters_grad, parameters_bp)[:actual_size]
            )

        prediction_b = torch.cat(predictions, dim=0)
        loss = torch.stack(losses).sum()
        parameters_grad_bp = torch.cat(parameter_grads, dim=0)
        return prediction_b, loss, parameters_grad_bp

    def __call__(
        self,
        parameters_bp: torch.Tensor,
        feature_elapsed_days_real_bl: torch.Tensor,
        feature_rating_bl: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(
            parameters_bp,
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            seq_lens,
        )


_DEFAULT_ADAPTER = FSRS7JaxAdapter()


def forward(
    parameters_bp: torch.Tensor,
    feature_elapsed_days_real_bl: torch.Tensor,
    feature_rating_bl: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _DEFAULT_ADAPTER.forward(
        parameters_bp,
        feature_elapsed_days_real_bl,
        feature_rating_bl,
        seq_lens,
    )
