import time
import torch
from dataclasses import dataclass
from typing import NamedTuple, Tuple

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
    sinc_coef_lb2: torch.Tensor
    delta_d_over_9_lb: torch.Tensor

    # scalar params
    init_d0: torch.Tensor
    init_d1: torch.Tensor
    nextd_mult: torch.Tensor
    init_d_4_rating_weight: torch.Tensor

    transition_coef_lb: torch.Tensor

def build_params(w, elapsed_bl, rating_bl):
    B, L = elapsed_bl.shape
    w_base = torch.tensor([7, 16], device=w.device)
    swp = w[33:35]
    s_exp = torch.stack([-swp[0], swp[1]])  # PRECOMPUTED

    hard_penalty_bl2 = torch.where(
        rating_bl.unsqueeze(-1).expand(B, L, 2) == 2,
        w[w_base + 7].view(1, 1, -1),
        torch.ones(1, 1, 2, device=elapsed_bl.device),
    )
    assert hard_penalty_bl2.shape == (B, L, 2), hard_penalty_bl2.shape

    easy_bonus_bl2 = torch.where(
        rating_bl.unsqueeze(-1).expand(B, L, 2) == 4,
        w[w_base + 8].view(1, 1, -1),
        torch.ones(1, 1, 2, device=elapsed_bl.device),
    )

    sinc_coef = (
        torch.exp(w[w_base] - 1.5).view(1, 1, -1)
        * hard_penalty_bl2
        * easy_bonus_bl2
    ).transpose(0, 1)
    # assert sinc_coef.shape == elapsed_bl.shape, sinc_coef.shape

    transition_coef_lb = (1 - w[26] * torch.exp(-w[25] * elapsed_bl)).transpose(0, 1)

    init_d_4_rating = 0.01 * init_d(w[4], w[5], torch.tensor(4.0, device=elapsed_bl.device))
    delta_d_over_9 = ((-w[6] * (rating_bl - 3)) / 9).transpose(0, 1)

    factor = w[29:31] ** (1 / -w[27:29]) - 1

    return FSRS7Buffer(
        s=w[:4],
        factor=factor,
        decay=-w[27:29],
        base=w[29:31],
        base_weight=w[31:33],
        swp=swp,
        s_exp=s_exp,

        sinc_base=w[w_base],
        sinc_s_exp=w[w_base + 1],
        sinc_r_mult=w[w_base + 2],
        fail_mult=w[w_base + 3],
        fail_d_exp=w[w_base + 4],
        fail_s_exp=w[w_base + 5],
        fail_r_mult=w[w_base + 6],
        sinc_coef_lb2=sinc_coef,

        init_d0=w[4],
        init_d1=w[5],
        nextd_mult=w[6],
        delta_d_over_9_lb=delta_d_over_9,
        init_d_4_rating_weight=init_d_4_rating,

        transition_coef_lb=transition_coef_lb,
    )

def forgetting_curve(p: FSRS7Buffer, t, s):
    t_over_s = (t / s).unsqueeze(-1)
    R = (1 + p.factor * t_over_s) ** p.decay
    weight = p.base_weight * s.unsqueeze(-1) ** p.s_exp
    return 1e-5 + (1 - 2e-5) * ((weight * R).sum(-1) / weight.sum(-1))

def stability_after_review(l: int, p: FSRS7Buffer, stability, difficulty, r, rating):
    success = rating > 1
    B = stability.shape[0]

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
        * p.sinc_coef_lb2[l]
    )

    new_s_success = torch.maximum(pls, stability.unsqueeze(1) * SInc)
    new_s_both = torch.where(success.unsqueeze(1), new_s_success, pls)
    return new_s_both.unbind(dim=-1)

def init_d(d0, d1, rating):
    return d0 - torch.exp(d1 * (rating - 1)) + 1

def next_d(l: int, p: FSRS7Buffer, difficulty):
    new_d = difficulty + p.delta_d_over_9_lb[l] * (10 - difficulty)
    new_d = p.init_d_4_rating_weight + 0.99 * new_d
    return new_d

def fsrs7_step(l: int, p: FSRS7Buffer, feature_elapsed, feature_rating, stability, difficulty):
    if l == 0:
        new_s = p.s.gather(dim=0, index=(feature_rating - 1).clamp_min(0))
        new_d = init_d(p.init_d0, p.init_d1, feature_rating).clamp(1, 10)
    else:
        r = forgetting_curve(p, feature_elapsed, stability)
        s_long, s_short = stability_after_review(l, p, stability, difficulty, r, feature_rating)
        coef_b = p.transition_coef_lb[l]
        new_s = torch.lerp(s_short, s_long, coef_b)
        new_d = next_d(l, p, difficulty).clamp(1, 10)

    new_s = new_s.clamp(1e-4, 36500)
    return new_s, new_d

@torch.jit.script
def forward(parameters_p, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_real_bl):
    p: FSRS7Buffer = build_params(parameters_p, feature_elapsed_days_real_bl, feature_rating_bl)

    B, L = feature_elapsed_days_real_bl.shape

    stability = torch.zeros((B,), device=parameters_p.device)
    difficulty = torch.zeros((B,), device=parameters_p.device)
    outputs = []

    l = 0
    for feature_elapsed, feature_rating in zip(feature_elapsed_days_real_bl.transpose(0, 1), feature_rating_bl.long().transpose(0, 1)):
        stability, difficulty = fsrs7_step(l, p, feature_elapsed, feature_rating, stability, difficulty)
        outputs.append(stability)
        l += 1

    output_tensor = torch.stack(outputs, dim=-1)

    return forgetting_curve(p, label_elapsed_days_real_bl, output_tensor), output_tensor

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
        1.15,  # Stability (long-term)
        0.882,
        0.3072,
        3.5875,
        0.303,
        0.0107,
        0.2279,
        2.6413,
        0.5594,
        1.15,  # Stability (short-term)
        3.5,
        0.5,  # Long-short term transition function
        0.0723,
        0.1634,
        0.6,
        0.9555,
        0.2245,
        0.6232,
        0.1362,
        0.3862,
    ]
)

FSRS_MIN = torch.tensor([
    0.0001,  # 0
    0.0001,  # 1 (depends on w0)
    0.0001,  # 2 (depends on w1)
    0.0001,  # 3 (depends on w2)
    1.0,     # 4
    0.001,   # 5
    0.1,     # 6
    0.0,     # 7
    0.0,     # 8
    0.3,     # 9
    0.01,    # 10
    0.001,   # 11
    0.1,     # 12
    0.0,     # 13
    0.0,     # 14
    1.0,     # 15
    0.0,     # 16
    0.0,     # 17
    0.5,     # 18
    0.001,   # 19
    0.001,   # 20
    0.001,   # 21
    0.0,     # 22
    0.0,     # 23
    1.0,     # 24
    2.5,     # 25
    0.0,     # 26
    0.01,    # 27
    0.01,    # 28 (depends on w27)
    0.5,     # 29
    0.5,     # 30 (depends on w29)
    0.01,    # 31
    0.1,     # 32
    0.0,     # 33
    0.1      # 34
])

FSRS_MAX = torch.tensor([
    50.0,   # 0
    100.0,  # 1
    100.0,  # 2
    100.0,  # 3
    10.0,   # 4
    4.0,    # 5
    4.0,    # 6
    4.0,    # 7
    1.2,    # 8
    3.0,    # 9
    1.5,    # 10
    0.9,    # 11
    1.0,    # 12
    3.5,    # 13
    1.0,    # 14
    7.0,    # 15
    4.0,    # 16
    2.0,    # 17
    6.0,    # 18
    1.5,    # 19
    2.0,    # 20
    1.0,    # 21
    5.0,    # 22
    1.0,    # 23
    7.0,    # 24
    15.0,   # 25
    1.0,    # 26
    0.25,   # 27
    0.95,   # 28
    0.85,   # 29
    0.99,   # 30
    1.0,    # 31
    1.0,    # 32
    0.9,    # 33
    1.1     # 34
])

assert (FSRS_MIN < FSRS7_DEFAULT_35).all()
assert (FSRS7_DEFAULT_35 < FSRS_MAX).all(), FSRS7_DEFAULT_35 < FSRS_MAX

@torch.jit.script
def _bounded(x, lo, hi, default):
    mid = torch.log((default - lo) / (hi - default))
    return lo + (hi - lo) * torch.sigmoid(x + mid)

@torch.jit.script
def _bounded_exp(x, lo, hi, default):
    lo_log = torch.log(lo)
    hi_log = torch.log(hi)
    def_log = torch.log(default)
    mid = torch.log((def_log - lo_log) / (hi_log - def_log))
    out_log = lo_log + (hi_log - lo_log) * torch.sigmoid(x + mid)
    return torch.exp(out_log)

def nn_vec_to_fsrs7_params(x):
    lo = FSRS_MIN.to(x.device)
    hi = FSRS_MAX.to(x.device)
    default = FSRS7_DEFAULT_35.to(x.device)

    exp_mask = torch.zeros(35, dtype=torch.bool, device=x.device)
    exp_mask[:4] = True

    out = torch.empty_like(x)

    out[exp_mask] = _bounded_exp(
        x[exp_mask],
        lo[exp_mask],
        hi[exp_mask],
        default[exp_mask],
    )

    out[~exp_mask] = _bounded(
        x[~exp_mask],
        lo[~exp_mask],
        hi[~exp_mask],
        default[~exp_mask],
    )

    sorted_S1_to_S3, _ = torch.sort(x[1:4])
    low_S = out[0].expand(3)
    hi_S = hi[1:4]
    default_S = low_S * 0.95 + hi_S * 0.05
    out = out.clone()
    out[1:4] = _bounded_exp(
        sorted_S1_to_S3,
        low_S,
        hi_S,
        default_S,
    )

    out[28] = _bounded(
        x[28],
        out[27],
        hi[28],
        (out[27] + hi[28]) / 2,
    )

    out[30] = _bounded(
        x[30],
        out[29],
        hi[30],
        (out[29] + hi[30]) / 2,
    )

    return out

if __name__ == '__main__':
    a = nn_vec_to_fsrs7_params(torch.full((35,), -100.0))
    b = nn_vec_to_fsrs7_params(torch.full((35,), 0.0))
    c = nn_vec_to_fsrs7_params(torch.full((35,), 100.0))
    print(a)
    print(b)
    print(c)

    sampler = torch.distributions.studentT.StudentT(torch.full((35,), 1.0))
    for i in range(1000):
        x = sampler.sample()
        w = nn_vec_to_fsrs7_params(x)

        # check dynamic constraints
        for i in range(1, 4):
            assert w[i - 1] <= w[i], w

        assert w[27] <= w[28]
        assert w[29] <= w[30]

    print("Check complete.")