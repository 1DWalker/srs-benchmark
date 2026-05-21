import torch
from typing import NamedTuple


class FSRS7Buffer(NamedTuple):
    s: torch.Tensor
    # forgetting curve
    factor: torch.Tensor
    decay: torch.Tensor
    base: torch.Tensor
    base_weight: torch.Tensor
    swp: torch.Tensor
    s_exp: torch.Tensor

    # stability after review
    sinc_base: torch.Tensor
    sinc_s_exp: torch.Tensor
    sinc_r_mult: torch.Tensor
    fail_mult: torch.Tensor
    fail_d_exp: torch.Tensor
    fail_s_exp: torch.Tensor
    fail_r_mult: torch.Tensor
    hard_penalty: torch.Tensor
    easy_bonus: torch.Tensor

    # scalar params
    init_d0: torch.Tensor
    init_d1: torch.Tensor
    nextd_mult: torch.Tensor
    init_d_4_rating_weight: torch.Tensor
    transition_decay: torch.Tensor
    transition_scale: torch.Tensor


def build_params_buffer(parameters_bp):
    swp = parameters_bp[:, 33:35]
    s_exp = torch.stack([-swp[:, 0], swp[:, 1]], dim=1)  # PRECOMPUTED

    sinc_base_b2 = torch.stack((parameters_bp[:, 7], parameters_bp[:, 16]), dim=1)
    sinc_s_exp_b2 = torch.stack((parameters_bp[:, 8], parameters_bp[:, 17]), dim=1)
    sinc_r_mult_b2 = torch.stack((parameters_bp[:, 9], parameters_bp[:, 18]), dim=1)
    fail_mult_b2 = torch.stack((parameters_bp[:, 10], parameters_bp[:, 19]), dim=1)
    fail_d_exp_b2 = torch.stack((parameters_bp[:, 11], parameters_bp[:, 20]), dim=1)
    fail_s_exp_b2 = torch.stack((parameters_bp[:, 12], parameters_bp[:, 21]), dim=1)
    fail_r_mult_b2 = torch.stack((parameters_bp[:, 13], parameters_bp[:, 22]), dim=1)
    hard_penalty_b2 = torch.stack((parameters_bp[:, 14], parameters_bp[:, 23]), dim=1)
    easy_bonus_b2 = torch.stack((parameters_bp[:, 15], parameters_bp[:, 24]), dim=1)

    init_d_4_rating = 0.01 * init_d(
        parameters_bp[:, 4],
        parameters_bp[:, 5],
        torch.tensor(4.0, device=parameters_bp.device),
    )

    factor = parameters_bp[:, 29:31] ** (1 / -parameters_bp[:, 27:29]) - 1

    return FSRS7Buffer(
        s=parameters_bp[:, :4],
        factor=factor,
        decay=-parameters_bp[:, 27:29],
        base=parameters_bp[:, 29:31],
        base_weight=parameters_bp[:, 31:33],
        swp=swp,
        s_exp=s_exp,
        sinc_base=sinc_base_b2,
        sinc_s_exp=sinc_s_exp_b2,
        sinc_r_mult=sinc_r_mult_b2,
        fail_mult=fail_mult_b2,
        fail_d_exp=fail_d_exp_b2,
        fail_s_exp=fail_s_exp_b2,
        fail_r_mult=fail_r_mult_b2,
        hard_penalty=hard_penalty_b2,
        easy_bonus=easy_bonus_b2,
        init_d0=parameters_bp[:, 4],
        init_d1=parameters_bp[:, 5],
        nextd_mult=parameters_bp[:, 6],
        init_d_4_rating_weight=init_d_4_rating,
        transition_decay=parameters_bp[:, 25],
        transition_scale=parameters_bp[:, 26],
    )


def forgetting_curve(p: FSRS7Buffer, t, s):
    t_over_s = (t / s).unsqueeze(-1)
    if t.dim() == 2:
        factor = p.factor.unsqueeze(1)
        decay = p.decay.unsqueeze(1)
        base_weight = p.base_weight.unsqueeze(1)
        s_exp = p.s_exp.unsqueeze(1)
    else:
        factor = p.factor
        decay = p.decay
        base_weight = p.base_weight
        s_exp = p.s_exp
    R = (1 + factor * t_over_s) ** decay
    weight = base_weight * s.unsqueeze(-1) ** s_exp
    return 1e-5 + (1 - 2e-5) * ((weight * R).sum(-1) / weight.sum(-1))


def stability_after_review(p: FSRS7Buffer, stability, difficulty, r, rating):
    success = rating > 1
    hard_penalty = torch.where(
        rating.unsqueeze(1) == 2,
        p.hard_penalty,
        torch.ones_like(p.hard_penalty),
    )
    easy_bonus = torch.where(
        rating.unsqueeze(1) == 4,
        p.easy_bonus,
        torch.ones_like(p.easy_bonus),
    )
    sinc_coef = torch.exp(p.sinc_base - 1.5) * hard_penalty * easy_bonus

    new_s_fail = (
        p.fail_mult
        * difficulty.unsqueeze(1).pow(-p.fail_d_exp)
        * ((stability.unsqueeze(1) + 1).pow(p.fail_s_exp) - 1)
        * torch.exp((1 - r).unsqueeze(1) * p.fail_r_mult)
    )

    pls = torch.minimum(stability.unsqueeze(1), new_s_fail)

    SInc = (
        1
        + (11 - difficulty).unsqueeze(1)
        * stability.unsqueeze(1).pow(-p.sinc_s_exp)
        * (torch.exp((1 - r).unsqueeze(1) * p.sinc_r_mult) - 1)
        * sinc_coef
    )

    new_s_success = torch.maximum(pls, stability.unsqueeze(1) * SInc)
    new_s_both = torch.where(success.unsqueeze(1), new_s_success, pls)
    return new_s_both.unbind(dim=-1)


def init_d(d0, d1, rating):
    return d0 - torch.exp(d1 * (rating - 1)) + 1


def next_d(p: FSRS7Buffer, difficulty, rating):
    new_d = difficulty + ((-p.nextd_mult * (rating - 3)) / 9) * (10 - difficulty)
    new_d = p.init_d_4_rating_weight + 0.99 * new_d
    return new_d

def fsrs7_step(l: int, p: FSRS7Buffer, feature_elapsed, feature_rating, stability, difficulty):
    if l == 0:
        new_s = p.s.gather(dim=1, index=(feature_rating - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
        new_d = init_d(p.init_d0, p.init_d1, feature_rating).clamp(1, 10)
    else:
        r = forgetting_curve(p, feature_elapsed, stability)
        s_long, s_short = stability_after_review(p, stability, difficulty, r, feature_rating)
        coef_b = 1 - p.transition_scale * torch.exp(-p.transition_decay * feature_elapsed)
        new_s = torch.lerp(s_short, s_long, coef_b)
        new_d = next_d(p, difficulty, feature_rating).clamp(1, 10)

    new_s = new_s.clamp(1e-4, 36500)
    return new_s, new_d


# @torch.compile(fullgraph=True, dynamic=True)
def forward(parameters_bp, feature_elapsed_days_real_bl, feature_rating_bl, seq_lens):
    p: FSRS7Buffer = build_params_buffer(parameters_bp)

    B, L = feature_elapsed_days_real_bl.shape

    stability = torch.zeros((B,), device=parameters_bp.device)
    difficulty = torch.zeros((B,), device=parameters_bp.device)
    outputs = []

    for l, (feature_elapsed, feature_rating) in enumerate(zip(
        feature_elapsed_days_real_bl.transpose(0, 1),
        feature_rating_bl.long().transpose(0, 1),
    )):
        # if l == L:
        #     break
        stability, difficulty = fsrs7_step(l, p, feature_elapsed, feature_rating, stability, difficulty)
        outputs.append(stability)

    output_tensor = torch.stack(outputs, dim=-1)
    review_s = output_tensor[torch.arange(B), seq_lens - 2]
    review_elapsed = feature_elapsed_days_real_bl[torch.arange(B), seq_lens - 1]
    assert not review_elapsed.isnan().any()
    assert not review_s.isnan().any()
    return forgetting_curve(p, review_elapsed, review_s)

def get_initial_params_for_optimization():
    return FSRS7_DEFAULT_35.clone()

def apply_parameter_clipper(parameters_b):
    assert parameters_b.shape[-1] == 35
    lo = FSRS_MIN.to(device=parameters_b.device, dtype=parameters_b.dtype)
    hi = FSRS_MAX.to(device=parameters_b.device, dtype=parameters_b.dtype)

    clipped = parameters_b.clamp(min=lo, max=hi)
    clipped = clipped.clone()

    clipped[..., 1] = torch.maximum(clipped[..., 1], clipped[..., 0])
    clipped[..., 2] = torch.maximum(clipped[..., 2], clipped[..., 1])
    clipped[..., 3] = torch.maximum(clipped[..., 3], clipped[..., 2])
    clipped[..., 28] = torch.maximum(clipped[..., 28], clipped[..., 27])
    clipped[..., 30] = torch.maximum(clipped[..., 30], clipped[..., 29])
    return clipped


FSRS7_DEFAULT_35 = torch.tensor(
    [
        0.041,
        2.4175,
        4.1283,
        11.9709,  # Initial S
        5.6385,
        0.4468,
        3.262,  # Difficulty
        2.3054,
        0.1688,
        1.3325,
        0.3524,
        0.0049,
        0.7503,
        0.0896,
        0.6625,
        1.3,  # Stability (long-term)
        0.882,
        0.3072,
        3.5875,
        0.303,
        0.0107,
        0.2279,
        2.6413,
        0.5594,
        1.3,  # Stability (short-term)
        2.5,
        1.0,  # Long-short term transition function
        0.0723,
        0.1634,
        0.5,
        0.9555,
        0.2245,
        0.6232,
        0.1362,
        0.3862,
    ]
)

FSRS_MIN = torch.tensor(
    [
        0.0001,  # 0
        0.0001,  # 1 (depends on w0)
        0.0001,  # 2 (depends on w1)
        0.0001,  # 3 (depends on w2)
        1.0,  # 4
        0.001,  # 5
        0.1,  # 6
        0.0,  # 7
        0.0,  # 8
        0.3,  # 9
        0.01,  # 10
        0.001,  # 11
        0.1,  # 12
        0.0,  # 13
        0.0,  # 14
        1.0,  # 15
        0.0,  # 16
        0.0,  # 17
        0.5,  # 18
        0.001,  # 19
        0.001,  # 20
        0.001,  # 21
        0.0,  # 22
        0.0,  # 23
        1.0,  # 24
        2.5,  # 25
        0.0,  # 26
        0.01,  # 27
        0.01,  # 28 (depends on w27)
        0.5,  # 29
        0.5,  # 30 (depends on w29)
        0.01,  # 31
        0.1,  # 32
        0.0,  # 33
        0.1,  # 34
    ]
)

FSRS_MAX = torch.tensor(
    [
        50.0,  # 0
        100.0,  # 1
        100.0,  # 2
        100.0,  # 3
        10.0,  # 4
        4.0,  # 5
        4.0,  # 6
        4.0,  # 7
        1.2,  # 8
        3.0,  # 9
        1.5,  # 10
        0.9,  # 11
        1.0,  # 12
        3.5,  # 13
        1.0,  # 14
        7.0,  # 15
        4.0,  # 16
        2.0,  # 17
        6.0,  # 18
        1.5,  # 19
        2.0,  # 20
        1.0,  # 21
        5.0,  # 22
        1.0,  # 23
        7.0,  # 24
        15.0,  # 25
        1.0,  # 26
        0.25,  # 27
        0.95,  # 28
        0.85,  # 29
        0.99,  # 30
        1.0,  # 31
        1.0,  # 32
        0.9,  # 33
        1.1,  # 34
    ]
)

assert (FSRS_MIN <= FSRS7_DEFAULT_35).all()
assert (FSRS7_DEFAULT_35 <= FSRS_MAX).all(), FSRS7_DEFAULT_35 <= FSRS_MAX