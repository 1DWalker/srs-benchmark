from __future__ import annotations

from typing import NamedTuple

import torch


class AdamWState(NamedTuple):
    step: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor

    def __getitem__(self, index):
        step_index = index[0] if isinstance(index, tuple) else index
        return AdamWState(
            step=self.step[step_index] if self.step.dim() > 0 else self.step,
            exp_avg=self.exp_avg[index],
            exp_avg_sq=self.exp_avg_sq[index],
        )


def init_adamw_state(params: torch.Tensor) -> AdamWState:
    return AdamWState(
        step=torch.zeros((), dtype=torch.int64, device=params.device),
        exp_avg=torch.zeros_like(params),
        exp_avg_sq=torch.zeros_like(params),
    )


def adamw_step(
    params: torch.Tensor,
    grad: torch.Tensor,
    state: AdamWState,
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, AdamWState]:
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
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    beta1, beta2 = betas
    new_step = step + 1

    new_exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    new_exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad.square()

    bias_correction1 = 1 - beta1**new_step
    bias_correction2 = 1 - beta2**new_step

    decayed_params = params * (1 - lr * weight_decay)
    denom = new_exp_avg_sq.sqrt() / bias_correction2.sqrt() + eps
    new_params = decayed_params - (lr / bias_correction1) * new_exp_avg / denom

    return new_params, new_step, new_exp_avg, new_exp_avg_sq
