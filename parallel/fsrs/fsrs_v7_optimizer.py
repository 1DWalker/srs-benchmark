from __future__ import annotations

from typing import NamedTuple

import torch

from parallel.fsrs import fsrs_v7_constants

class AdamWState(NamedTuple):
    step: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor

def init_adamw_state(params: torch.Tensor) -> AdamWState:
    return AdamWState(
        step=torch.zeros(params.shape[:-1], dtype=torch.int32, device=params.device),
        exp_avg=torch.zeros_like(params),
        exp_avg_sq=torch.zeros_like(params),
    )

# @torch.compile(fullgraph=True)
def adamw_step(
    params: torch.Tensor,
    grad: torch.Tensor,
    state: AdamWState,
    *,
    lr: torch.Tensor,
    mask: torch.Tensor,
    betas: tuple[float, float],
    eps: float = 1e-8,
    weight_decay: float = 0.00,
) -> tuple[torch.Tensor, AdamWState]:
    updated_params, updated_step, updated_exp_avg, updated_exp_avg_sq = adamw_update(
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

    return torch.where(mask.unsqueeze(-1), updated_params, params), AdamWState(
        step=torch.where(mask, updated_step, state.step),
        exp_avg=torch.where(mask.unsqueeze(-1), updated_exp_avg, state.exp_avg),
        exp_avg_sq=torch.where(mask.unsqueeze(-1), updated_exp_avg_sq, state.exp_avg_sq),
    )


def adamw_update(
    params: torch.Tensor,
    grad: torch.Tensor,
    step: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: torch.Tensor,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    beta1, beta2 = betas
    new_step = step + 1

    new_exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    new_exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad.square()

    beta1_t = torch.tensor(beta1, device=params.device, dtype=params.dtype)
    beta2_t = torch.tensor(beta2, device=params.device, dtype=params.dtype)
    bias_correction1 = 1 - torch.pow(beta1_t, new_step.unsqueeze(-1).to(dtype=params.dtype))
    bias_correction2 = 1 - torch.pow(beta2_t, new_step.unsqueeze(-1).to(dtype=params.dtype))

    decayed_params = params * (1 - lr * weight_decay)
    denom = new_exp_avg_sq.sqrt() / bias_correction2.sqrt() + eps
    new_params = decayed_params - (lr / bias_correction1) * new_exp_avg / denom

    return new_params, new_step, new_exp_avg, new_exp_avg_sq
