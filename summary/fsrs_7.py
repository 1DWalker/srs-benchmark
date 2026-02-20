import time
import torch
from dataclasses import dataclass

@dataclass
class FSRS7Params:
    s: torch.Tensor
    # forgetting curve
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
    hard: torch.Tensor
    easy: torch.Tensor

    # scalar params
    init_d0: torch.Tensor
    init_d1: torch.Tensor
    nextd_mult: torch.Tensor

    trans_a: torch.Tensor
    trans_b: torch.Tensor

def build_params(w):
    w_base = torch.tensor([7, 16], device=w.device)
    swp = w[33:35]
    s_exp = torch.stack([-swp[0], swp[1]])  # PRECOMPUTED

    return FSRS7Params(
        s=w[:4],
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
        hard=w[w_base + 7],
        easy=w[w_base + 8],

        init_d0=w[4],
        init_d1=w[5],
        nextd_mult=w[6],

        trans_a=w[26],
        trans_b=w[25],
    )

def forgetting_curve(p, t, s):
    t_over_s = (t / s).unsqueeze(-1)
    factor = p.base ** (1 / p.decay) - 1
    R = (1 + factor * t_over_s) ** p.decay
    weight = p.base_weight * s.unsqueeze(-1) ** p.s_exp
    return (weight * R).sum(-1) / weight.sum(-1)

def stability_after_review(p, state, r, rating):
    old_s = state[:, 0]
    old_d = state[:, 1]
    success = rating > 1
    B = state.shape[0]

    hard_penalty = torch.where(
        rating.unsqueeze(1) == 2,
        p.hard.unsqueeze(0),
        torch.ones(B, 2, device=state.device),
    )

    easy_bonus = torch.where(
        rating.unsqueeze(1) == 4,
        p.easy.unsqueeze(0),
        torch.ones(B, 2, device=state.device),
    )

    new_s_fail = (
        p.fail_mult
        * old_d.unsqueeze(1).pow(-p.fail_d_exp)
        * ((old_s.unsqueeze(1) + 1).pow(p.fail_s_exp) - 1)
        * torch.exp((1 - r).unsqueeze(1) * p.fail_r_mult)
    )

    pls = torch.minimum(old_s.unsqueeze(1), new_s_fail)

    SInc = (
        1
        + torch.exp(p.sinc_base - 1.5)
        * (11 - old_d).unsqueeze(1)
        * old_s.unsqueeze(1).pow(-p.sinc_s_exp)
        * (torch.exp((1 - r).unsqueeze(1) * p.sinc_r_mult) - 1)
        * hard_penalty
        * easy_bonus
    )

    new_s_success = torch.maximum(pls, old_s.unsqueeze(1) * SInc)
    new_s_both = torch.where(success.unsqueeze(1), new_s_success, pls)

    return new_s_both[:, 0], new_s_both[:, 1]

def init_d(p, rating):
    return p.init_d0 - torch.exp(p.init_d1 * (rating - 1)) + 1

def next_d(p, state, rating):
    delta_d = -p.nextd_mult * (rating - 3)
    new_d = state[:, 1] + delta_d * (10 - state[:, 1]) / 9
    new_d = 0.01 * init_d(p, torch.tensor(4.0, device=state.device)) + 0.99 * new_d
    return new_d

def transition_function(p, delta_t):
    return 1 - p.trans_a * torch.exp(-p.trans_b * delta_t)

def linear_damping(delta_d, old_d):
    return delta_d * (10 - old_d) / 9

def fsrs7_step(p, X, state):
    if torch.equal(state, torch.zeros_like(state)):
        keys = torch.tensor([1, 2, 3, 4], device=X.device)
        index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)

        new_s = torch.ones_like(state[:, 0])
        new_s[index[0]] = p.s[index[1]]

        new_d = init_d(p, X[:, 1]).clamp(1, 10)
    else:
        r = forgetting_curve(p, X[:, 0], state[:, 0])
        s_long, s_short = stability_after_review(p, state, r, X[:, 1])
        coef = transition_function(p, X[:, 0])

        new_s = coef * s_long + (1 - coef) * s_short
        new_d = next_d(p, state, X[:, 1]).clamp(1, 10)

    new_s = new_s.clamp(1e-4, 36500)
    return torch.stack([new_s, new_d], dim=1)

def forward(parameters_p, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_real_bl):
    p = build_params(parameters_p)

    B, L = feature_elapsed_days_real_bl.shape
    inputs_bl2 = torch.stack((feature_elapsed_days_real_bl, feature_rating_bl), dim=-1)

    state = torch.zeros((B, 2), device=inputs_bl2.device)
    outputs = []

    for X in inputs_bl2.transpose(0, 1):
        state = fsrs7_step(p, X, state)
        outputs.append(state)

    output_tensor = torch.stack(outputs).permute(1, 0, 2)
    output_s = output_tensor[..., 0]

    return forgetting_curve(p, label_elapsed_days_real_bl, output_s), output_tensor


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

def _bounded(x, lo, hi, default):
    mid = torch.log((default - lo) / (hi - default))
    return lo + (hi - lo) * torch.sigmoid(x + mid)

def _bounded_exp(x, lo, hi, default):
    lo_log = torch.log(lo)
    hi_log = torch.log(hi)
    def_log = torch.log(default)
    mid = torch.log((def_log - lo_log) / (hi_log - def_log))
    out_log = lo_log + (hi_log - lo_log) * torch.sigmoid(x + mid)
    return torch.exp(out_log)

def nn_vec_to_fsrs7_params(nn_vec_35):
    lo = FSRS_MIN.to(nn_vec_35.device)
    hi = FSRS_MAX.to(nn_vec_35.device)
    default = FSRS7_DEFAULT_35.to(nn_vec_35.device)

    exp_mask = torch.zeros(35, dtype=torch.bool, device=nn_vec_35.device)
    exp_mask[:4] = True

    out = torch.empty_like(nn_vec_35)

    out[exp_mask] = _bounded_exp(
        nn_vec_35[exp_mask],
        lo[exp_mask],
        hi[exp_mask],
        default[exp_mask],
    )

    out[~exp_mask] = _bounded(
        nn_vec_35[~exp_mask],
        lo[~exp_mask],
        hi[~exp_mask],
        default[~exp_mask],
    )

    return out

if __name__ == '__main__':
    a = nn_vec_to_fsrs7_params(torch.full((35,), -10.0))
    b = nn_vec_to_fsrs7_params(torch.full((35,), 0.0))
    c = nn_vec_to_fsrs7_params(torch.full((35,), 10.0))
    print(a)
    print(b)
    print(c)