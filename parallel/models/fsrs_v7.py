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


def forward(parameters_bp, feature_elapsed_days_real_bl, feature_rating_bl, seq_lens):
    p: FSRS7Buffer = build_params_buffer(parameters_bp)

    B, L = feature_elapsed_days_real_bl.shape
    assert parameters_bp.shape[0] == B

    stability = torch.zeros((B,), device=parameters_bp.device)
    difficulty = torch.zeros((B,), device=parameters_bp.device)
    outputs = []

    l = 0
    for feature_elapsed, feature_rating in zip(
        feature_elapsed_days_real_bl.transpose(0, 1),
        feature_rating_bl.long().transpose(0, 1),
    ):
        if l == L:
            break
        stability, difficulty = fsrs7_step(l, p, feature_elapsed, feature_rating, stability, difficulty)
        outputs.append(stability)
        l += 1

    output_tensor = torch.stack(outputs, dim=-1)
    review_s = output_tensor[torch.arange(B), seq_lens - 2]
    review_elapsed = feature_elapsed_days_real_bl[torch.arange(B), seq_lens - 1]
    return forgetting_curve(p, review_elapsed, review_s)

def get_initial_params_for_optimization():
    return torch.zeros(35)

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
    # [
    #     0.041,
    #     2.4175,
    #     4.1283,
    #     11.9709,  # Initial S
    #     5.6385,
    #     0.4468,
    #     3.262,  # Difficulty
    #     2.3054,
    #     0.1688,
    #     1.3325,
    #     0.3524,
    #     0.0049,
    #     0.7503,
    #     0.0896,
    #     0.6625,
    #     1.3,  # Stability (long-term)
    #     0.882,
    #     0.3072,
    #     3.5875,
    #     0.303,
    #     0.0107,
    #     0.2279,
    #     2.6413,
    #     0.5594,
    #     1.3,  # Stability (short-term)
    #     2.5,
    #     1.0,  # Long-short term transition function
    #     0.0723,
    #     0.1634,
    #     0.5,
    #     0.9555,
    #     0.2245,
    #     0.6232,
    #     0.1362,
    #     0.3862,
    # ]
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


def make_slices_increasing(x):
    sort_slices = [(1, 4)]
    for i in range(1, len(sort_slices)):
        assert sort_slices[i][0] >= sort_slices[i - 1][1]
    x_out = []
    start = 0
    dim = x.dim() - 1
    for l, r in sort_slices:
        if l > start:
            x_out.append(x[..., start:l])
        slice_len = r - l
        if slice_len == 1:
            x_out.append(x[..., l:r])
        else:
            first = x[..., l : l + 1]
            increments = torch.nn.functional.softplus(x[..., (l + 1) : r])
            increasing_slice = torch.cat(
                [first, first + torch.cumsum(increments, dim=dim)],
                dim=dim,
            )
            x_out.append(increasing_slice)
        start = r
    if start < x.shape[-1]:
        x_out.append(x[..., start:])
    return torch.cat(x_out, dim=dim)


def nn_vec_to_fsrs7_params(x):
    assert x.shape[-1] == 35
    lo = FSRS_MIN.to(x.device)
    hi = FSRS_MAX.to(x.device)
    default = FSRS7_DEFAULT_35.to(x.device)

    x = make_slices_increasing(x)

    exp_mask = torch.zeros(35, dtype=torch.bool, device=x.device)
    exp_mask[:4] = True

    out = torch.empty_like(x)

    out[..., exp_mask] = _bounded_exp(
        x[..., exp_mask],
        lo[exp_mask],
        hi[exp_mask],
        default[exp_mask],
    )

    out[..., ~exp_mask] = _bounded(
        x[..., ~exp_mask],
        lo[~exp_mask],
        hi[~exp_mask],
        default[~exp_mask],
    )

    out_copy = out.clone()
    out_copy[..., 1:4] = _bounded_exp(
        x[..., 1:4],
        out[..., 0:1].expand_as(x[..., 1:4]),
        hi[1:4],
        (out[..., 0:1] + hi[1:4]) / 2,
    )
    out_copy[..., 28] = _bounded(
        x[..., 28],
        out[..., 27],
        hi[28],
        (out[..., 27] + hi[28]) / 2,
    )
    out_copy[..., 30] = _bounded(
        x[..., 30],
        out[..., 29],
        hi[30],
        (out[..., 29] + hi[30]) / 2,
    )
    return out_copy


def _assert_param_constraints(w):
    assert (FSRS_MIN.to(w.device) <= w + 1e-4).all(), FSRS_MIN.to(w.device) <= w + 1e-4
    assert (w <= FSRS_MAX.to(w.device) + 1e-4).all(), w <= FSRS_MAX.to(w.device) + 1e-4
    assert (w[..., 0] <= w[..., 1]).all(), w
    assert (w[..., 1] <= w[..., 2]).all(), w
    assert (w[..., 2] <= w[..., 3]).all(), w
    assert (w[..., 27] <= w[..., 28]).all(), w
    assert (w[..., 29] <= w[..., 30]).all(), w


def _check_batched_forward():
    raw_bp = torch.stack(
        [
            torch.zeros(35),
            torch.linspace(-1.0, 1.0, 35),
            torch.linspace(1.0, -1.0, 35),
        ],
        dim=0,
    )
    parameters_bp = nn_vec_to_fsrs7_params(raw_bp)

    feature_elapsed_days_real_bl = torch.tensor(
        [
            [0.0, 1.0, 3.0, 0.25, 10.0],
            [0.0, 0.5, 2.0, 4.0, 8.0],
            [0.0, 2.0, 1.0, 5.0, 13.0],
        ],
        dtype=torch.float32,
    )
    feature_rating_bl = torch.tensor(
        [
            [1, 2, 3, 4, 3],
            [4, 3, 2, 1, 4],
            [2, 2, 3, 3, 4],
        ],
        dtype=torch.long,
    )
    label_elapsed_days_real_bl = torch.tensor(
        [
            [0.25, 1.5, 4.0, 1.0, 12.0],
            [0.1, 1.0, 3.0, 6.0, 10.0],
            [0.5, 2.5, 2.0, 7.0, 15.0],
        ],
        dtype=torch.float32,
    )

    batched_retention_bl, batched_stability_bl = forward(
        parameters_bp,
        feature_elapsed_days_real_bl,
        feature_rating_bl,
        label_elapsed_days_real_bl,
    )

    for b in range(parameters_bp.shape[0]):
        single_retention_bl, single_stability_bl = forward(
            parameters_bp[b : b + 1],
            feature_elapsed_days_real_bl[b : b + 1],
            feature_rating_bl[b : b + 1],
            label_elapsed_days_real_bl[b : b + 1],
        )
        assert torch.allclose(
            batched_retention_bl[b : b + 1],
            single_retention_bl,
            rtol=1e-6,
            atol=1e-6,
        )
        assert torch.allclose(
            batched_stability_bl[b : b + 1],
            single_stability_bl,
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    a = nn_vec_to_fsrs7_params(torch.full((35,), -1000.0))
    b = nn_vec_to_fsrs7_params(torch.full((35,), 0.0))
    c = nn_vec_to_fsrs7_params(torch.full((35,), 1000.0))
    print(a)
    print(b)
    print(c)
    assert ((a - FSRS_MIN).abs() < 1e-4).all()
    assert ((c - FSRS_MAX).abs() < 1e-4).all()
    _assert_param_constraints(a)
    _assert_param_constraints(b)
    _assert_param_constraints(c)
    h = [0.0 for _ in range(35)]
    h[1] = -2
    print(nn_vec_to_fsrs7_params(torch.tensor(h)))

    batch_abc = nn_vec_to_fsrs7_params(
        torch.stack(
            [
                torch.full((35,), -1000.0),
                torch.full((35,), 0.0),
                torch.full((35,), 1000.0),
            ],
            dim=0,
        )
    )
    assert torch.allclose(batch_abc[0], a)
    assert torch.allclose(batch_abc[1], b)
    assert torch.allclose(batch_abc[2], c)
    _assert_param_constraints(batch_abc)

    sampler = torch.distributions.studentT.StudentT(torch.full((35,), 1.0))
    for _ in range(1000):
        x = sampler.sample()
        w = nn_vec_to_fsrs7_params(x)
        _assert_param_constraints(w)

    for _ in range(100):
        x = sampler.sample((8,))
        w = nn_vec_to_fsrs7_params(x)
        _assert_param_constraints(w)

    _check_batched_forward()

    print("Check complete.")
