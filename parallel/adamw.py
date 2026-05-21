from __future__ import annotations

from typing import NamedTuple

import torch


class AdamWState(NamedTuple):
    step: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor

def init_adamw_state(params: torch.Tensor) -> AdamWState:
    return AdamWState(
        step=torch.zeros(params.shape[:-1], dtype=torch.int64, device=params.device),
        exp_avg=torch.zeros_like(params),
        exp_avg_sq=torch.zeros_like(params),
    )


def _reshape_for_params(value: float | torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.to(device=params.device, dtype=params.dtype)
    else:
        out = torch.tensor(value, device=params.device, dtype=params.dtype)

    while out.ndim < params.ndim:
        out = out.unsqueeze(-1)
    return out


def _masked_lr(lr: float | torch.Tensor, mask: torch.Tensor) -> float | torch.Tensor:
    if torch.is_tensor(lr) and lr.ndim > 0:
        return lr[mask]
    return lr


def adamw_step(
    params: torch.Tensor,
    grad: torch.Tensor,
    state: AdamWState,
    *,
    lr: float | torch.Tensor,
    mask: torch.Tensor | None = None,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, AdamWState]:
    if mask is not None:
        if mask.shape != state.step.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match state.step shape "
                f"{tuple(state.step.shape)}."
            )
        mask = mask.to(device=params.device, dtype=torch.bool)
        if not bool(mask.any()):
            return params, state

        new_params = params.clone()
        new_step = state.step.clone()
        new_exp_avg = state.exp_avg.clone()
        new_exp_avg_sq = state.exp_avg_sq.clone()

        (
            new_params[mask],
            new_step[mask],
            new_exp_avg[mask],
            new_exp_avg_sq[mask],
        ) = adamw_update(
            params[mask],
            grad[mask],
            state.step[mask],
            state.exp_avg[mask],
            state.exp_avg_sq[mask],
            lr=_masked_lr(lr, mask),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        return new_params, AdamWState(
            step=new_step,
            exp_avg=new_exp_avg,
            exp_avg_sq=new_exp_avg_sq,
        )

    new_params, step, exp_avg, exp_avg_sq = adamw_update(
        params,
        grad,
        state.step,
        state.exp_avg,
        state.exp_avg_sq,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    return new_params, AdamWState(
        step=step,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
    )


def adamw_update(
    params: torch.Tensor,
    grad: torch.Tensor,
    step: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float | torch.Tensor,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    beta1, beta2 = betas
    new_step = step + 1
    lr = _reshape_for_params(lr, params)

    new_exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    new_exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad.square()

    beta1_t = torch.tensor(beta1, device=params.device, dtype=params.dtype)
    beta2_t = torch.tensor(beta2, device=params.device, dtype=params.dtype)
    bias_correction1 = _reshape_for_params(
        1 - torch.pow(beta1_t, new_step.to(dtype=params.dtype)),
        params,
    )
    bias_correction2 = _reshape_for_params(
        1 - torch.pow(beta2_t, new_step.to(dtype=params.dtype)),
        params,
    )

    decayed_params = params * (1 - lr * weight_decay)
    denom = new_exp_avg_sq.sqrt() / bias_correction2.sqrt() + eps
    new_params = decayed_params - (lr / bias_correction1) * new_exp_avg / denom

    return new_params, new_step, new_exp_avg, new_exp_avg_sq
