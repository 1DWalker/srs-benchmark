import torch
import math

def scheduler(
    t: torch.Tensor,
    total_steps: torch.Tensor,
    eta_min: float = 0.0,
) -> torch.Tensor:
    # cosine annealing lr
    progress = (t.float() / total_steps.float()).clamp(0.0, 1.0)
    cosine = 0.5 * (1.0 + torch.cos(torch.pi * progress))
    return eta_min + (1.0 - eta_min) * cosine