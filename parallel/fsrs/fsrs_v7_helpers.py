import torch

from parallel.fsrs import fsrs_v7_constants

PENALTY_W_L2 = 0.5

default_params = torch.tensor(fsrs_v7_constants.FSRS7_DEFAULT_35_VALUES, dtype=torch.float32)
sigma = torch.tensor(fsrs_v7_constants.FSRS7_L2_SIGMA_35_VALUES, dtype=torch.float32)

def get_initial_params_for_optimization():
    return torch.tensor(fsrs_v7_constants.FSRS7_DEFAULT_35_VALUES, dtype=torch.float32)

@torch.compile(fullgraph=True)
def apply_parameter_clipper(parameters_b):
    lo = torch.tensor(
        fsrs_v7_constants.FSRS_MIN_VALUES,
        device=parameters_b.device,
        dtype=parameters_b.dtype,
    )
    hi = torch.tensor(
        fsrs_v7_constants.FSRS_MAX_VALUES,
        device=parameters_b.device,
        dtype=parameters_b.dtype,
    )

    clipped = parameters_b.clamp(min=lo, max=hi).clone()
    clipped[..., 1] = torch.maximum(clipped[..., 1], clipped[..., 0])
    clipped[..., 2] = torch.maximum(clipped[..., 2], clipped[..., 1])
    clipped[..., 3] = torch.maximum(clipped[..., 3], clipped[..., 2])
    clipped[..., 28] = torch.maximum(clipped[..., 28], clipped[..., 27])
    clipped[..., 30] = torch.maximum(clipped[..., 30], clipped[..., 29])
    return clipped