import torch

from parallel.fsrs import fsrs_v7_constants

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

    with torch.no_grad():
        clipped = parameters_b.clamp(min=lo, max=hi).clone()
        clipped[..., 1] = torch.maximum(clipped[..., 1], clipped[..., 0])
        clipped[..., 2] = torch.maximum(clipped[..., 2], clipped[..., 1])
        clipped[..., 3] = torch.maximum(clipped[..., 3], clipped[..., 2])
        clipped[..., 28] = torch.maximum(clipped[..., 28], clipped[..., 27])
        clipped[..., 30] = torch.maximum(clipped[..., 30], clipped[..., 29])
    return clipped

@torch.compile(fullgraph=True)
def penalty_loss(parameters_kp, batch_size_k, training_set_size_k):
    default_params = torch.tensor(
        fsrs_v7_constants.FSRS7_DEFAULT_35_VALUES,
        device=parameters_kp.device,
        dtype=parameters_kp.dtype,
    )
    sigma = torch.tensor(
        fsrs_v7_constants.FSRS7_L2_SIGMA_35_VALUES,
        device=parameters_kp.device,
        dtype=parameters_kp.dtype,
    )
    l2_k = torch.sum(
        torch.square(parameters_kp - default_params.unsqueeze(0))
        / torch.square(sigma.unsqueeze(0)),
        dim=-1,
    )
    penalty_k = (
        fsrs_v7_constants.PENALTY_W_L2
        * batch_size_k / training_set_size_k
        * l2_k
    )
    return penalty_k

def gradient_weight(review_ord_kb, training_set_size_k):
    # more recent reviews get a higher weight
    review_ord_lin = review_ord_kb / training_set_size_k.unsqueeze(-1)
    return fsrs_v7_constants.RECENCY_C0 + fsrs_v7_constants.RECENCY_C1 * torch.pow(review_ord_lin, 3)