from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import torch


def tensor_to_jax_array(tensor: torch.Tensor) -> jax.Array:
    tensor = tensor.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    try:
        return jax.dlpack.from_dlpack(tensor, copy=False)
    except ValueError as exc:
        if "requires a copy" not in str(exc):
            raise
        return jax.dlpack.from_dlpack(tensor, copy=None)

def tensors_to_jax_arrays(*tensors: Any) -> tuple[jax.Array, ...]:
    return tuple(tensor_to_jax_array(tensor) for tensor in tensors)


class FSRS7Buffer(NamedTuple):
    s: jax.Array
    # forgetting curve
    factor: jax.Array
    decay: jax.Array
    base: jax.Array
    base_weight: jax.Array
    swp: jax.Array
    s_exp: jax.Array

    # stability after review
    sinc_base: jax.Array
    sinc_s_exp: jax.Array
    sinc_r_mult: jax.Array
    fail_mult: jax.Array
    fail_d_exp: jax.Array
    fail_s_exp: jax.Array
    fail_r_mult: jax.Array
    hard_penalty: jax.Array
    easy_bonus: jax.Array

    # scalar params
    init_d0: jax.Array
    init_d1: jax.Array
    nextd_mult: jax.Array
    init_d_4_rating_weight: jax.Array
    transition_decay: jax.Array
    transition_scale: jax.Array


def build_params_buffer(parameters_bp: jax.Array) -> FSRS7Buffer:
    swp = parameters_bp[:, 33:35]
    s_exp = jnp.stack((-swp[:, 0], swp[:, 1]), axis=1)

    sinc_base_b2 = jnp.stack((parameters_bp[:, 7], parameters_bp[:, 16]), axis=1)
    sinc_s_exp_b2 = jnp.stack((parameters_bp[:, 8], parameters_bp[:, 17]), axis=1)
    sinc_r_mult_b2 = jnp.stack((parameters_bp[:, 9], parameters_bp[:, 18]), axis=1)
    fail_mult_b2 = jnp.stack((parameters_bp[:, 10], parameters_bp[:, 19]), axis=1)
    fail_d_exp_b2 = jnp.stack((parameters_bp[:, 11], parameters_bp[:, 20]), axis=1)
    fail_s_exp_b2 = jnp.stack((parameters_bp[:, 12], parameters_bp[:, 21]), axis=1)
    fail_r_mult_b2 = jnp.stack((parameters_bp[:, 13], parameters_bp[:, 22]), axis=1)
    hard_penalty_b2 = jnp.stack((parameters_bp[:, 14], parameters_bp[:, 23]), axis=1)
    easy_bonus_b2 = jnp.stack((parameters_bp[:, 15], parameters_bp[:, 24]), axis=1)

    init_d_4_rating = 0.01 * init_d(
        parameters_bp[:, 4],
        parameters_bp[:, 5],
        jnp.asarray(4.0, dtype=parameters_bp.dtype),
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


def forgetting_curve(p: FSRS7Buffer, t: jax.Array, s: jax.Array) -> jax.Array:
    t_over_s = (t / s)[..., None]
    if t.ndim == 2:
        factor = p.factor[:, None, :]
        decay = p.decay[:, None, :]
        base_weight = p.base_weight[:, None, :]
        s_exp = p.s_exp[:, None, :]
    else:
        factor = p.factor
        decay = p.decay
        base_weight = p.base_weight
        s_exp = p.s_exp
    r = (1 + factor * t_over_s) ** decay
    weight = base_weight * s[..., None] ** s_exp
    return 1e-5 + (1 - 2e-5) * ((weight * r).sum(axis=-1) / weight.sum(axis=-1))


def stability_after_review(
    p: FSRS7Buffer,
    stability: jax.Array,
    difficulty: jax.Array,
    r: jax.Array,
    rating: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    success = rating > 1
    hard_penalty = jnp.where(
        rating[:, None] == 2,
        p.hard_penalty,
        jnp.ones_like(p.hard_penalty),
    )
    easy_bonus = jnp.where(
        rating[:, None] == 4,
        p.easy_bonus,
        jnp.ones_like(p.easy_bonus),
    )
    sinc_coef = jnp.exp(p.sinc_base - 1.5) * hard_penalty * easy_bonus

    new_s_fail = (
        p.fail_mult
        * difficulty[:, None] ** (-p.fail_d_exp)
        * ((stability[:, None] + 1) ** p.fail_s_exp - 1)
        * jnp.exp((1 - r)[:, None] * p.fail_r_mult)
    )

    pls = jnp.minimum(stability[:, None], new_s_fail)

    s_inc = (
        1
        + (11 - difficulty)[:, None]
        * stability[:, None] ** (-p.sinc_s_exp)
        * jnp.expm1((1 - r)[:, None] * p.sinc_r_mult)
        * sinc_coef
    )

    new_s_success = jnp.maximum(pls, stability[:, None] * s_inc)
    new_s_both = jnp.where(success[:, None], new_s_success, pls)
    return new_s_both[:, 0], new_s_both[:, 1]


def init_d(d0: jax.Array, d1: jax.Array, rating: jax.Array) -> jax.Array:
    return d0 - jnp.exp(d1 * (rating - 1)) + 1


def next_d(p: FSRS7Buffer, difficulty: jax.Array, rating: jax.Array) -> jax.Array:
    new_d = difficulty + ((-p.nextd_mult * (rating - 3)) / 9) * (10 - difficulty)
    new_d = p.init_d_4_rating_weight + 0.99 * new_d
    return new_d


def first_review(
    p: FSRS7Buffer,
    feature_rating: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    rating_idx = jnp.maximum(feature_rating.astype(jnp.int32) - 1, 0)[:, None]
    new_s = jnp.take_along_axis(p.s, rating_idx, axis=1).squeeze(axis=1)
    new_d = jnp.clip(init_d(p.init_d0, p.init_d1, feature_rating), 1, 10)
    return jnp.clip(new_s, 1e-4, 36500), new_d


def fsrs7_step(
    p: FSRS7Buffer,
    feature_elapsed: jax.Array,
    feature_rating: jax.Array,
    stability: jax.Array,
    difficulty: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r = forgetting_curve(p, feature_elapsed, stability)
    s_long, s_short = stability_after_review(p, stability, difficulty, r, feature_rating)
    coef_b = 1 - p.transition_scale * jnp.exp(-p.transition_decay * feature_elapsed)
    new_s = s_short + coef_b * (s_long - s_short)
    new_d = jnp.clip(next_d(p, difficulty, feature_rating), 1, 10)
    return jnp.clip(new_s, 1e-4, 36500), new_d


def _forward(
    parameters_bp: jax.Array,
    feature_elapsed_days_real_bl: jax.Array,
    feature_rating_bl: jax.Array,
    seq_lens: jax.Array,
) -> jax.Array:
    p = build_params_buffer(parameters_bp)

    b = feature_elapsed_days_real_bl.shape[0]
    l = feature_elapsed_days_real_bl.shape[1]
    feature_elapsed_days_real_lb = feature_elapsed_days_real_bl.T
    feature_rating_lb = feature_rating_bl.astype(jnp.int32).T
    seq_lens = seq_lens.astype(jnp.int32)
    target_step = seq_lens - 2
    init_s, init_d = first_review(p, feature_rating_lb[0])

    def scan_step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        xs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        stability, difficulty, review_s = carry
        step, feature_elapsed, feature_rating = xs
        new_s, new_d = fsrs7_step(
            p,
            feature_elapsed,
            feature_rating,
            stability,
            difficulty,
        )
        review_s = jnp.where(step == target_step, new_s, review_s)
        return (new_s, new_d, review_s), None

    (_, _, review_s), _ = jax.lax.scan(
        scan_step,
        (init_s, init_d, init_s),
        (
            jnp.arange(1, l - 1, dtype=jnp.int32),
            feature_elapsed_days_real_lb[1:-1],
            feature_rating_lb[1:-1],
        ),
    )

    batch_idx = jnp.arange(b)
    review_elapsed = feature_elapsed_days_real_bl[batch_idx, seq_lens - 1]
    return forgetting_curve(p, review_elapsed, review_s)


@jax.jit
def forward(
    parameters_bp: jax.Array,
    feature_elapsed_days_real_bl: jax.Array,
    feature_rating_bl: jax.Array,
    seq_lens: jax.Array,
) -> jax.Array:
    return _forward(
        parameters_bp,
        feature_elapsed_days_real_bl,
        feature_rating_bl,
        seq_lens,
    )


def binary_cross_entropy_sum(prediction: jax.Array, label: jax.Array) -> jax.Array:
    label = label.astype(prediction.dtype)
    return -jnp.sum(label * jnp.log(prediction) + (1 - label) * jnp.log1p(-prediction))


def binary_cross_entropy_masked_sum(
    prediction: jax.Array,
    label: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    label = label.astype(prediction.dtype)
    mask = mask.astype(prediction.dtype)
    loss_b = -(label * jnp.log(prediction) + (1 - label) * jnp.log1p(-prediction))
    return jnp.sum(loss_b * mask)


def label_from_feature_rating(
    feature_rating_bl: jax.Array,
    seq_lens: jax.Array,
) -> jax.Array:
    batch_idx = jnp.arange(feature_rating_bl.shape[0])
    target_rating = feature_rating_bl[batch_idx, seq_lens.astype(jnp.int32) - 1]
    return (target_rating > 1).astype(jnp.float32)


def loss(
    parameters_bp: jax.Array,
    feature_elapsed_days_real_bl: jax.Array,
    feature_rating_bl: jax.Array,
    seq_lens: jax.Array,
) -> jax.Array:
    prediction_b = _forward(
        parameters_bp,
        feature_elapsed_days_real_bl,
        feature_rating_bl,
        seq_lens,
    )
    label_b = label_from_feature_rating(feature_rating_bl, seq_lens)
    return binary_cross_entropy_sum(prediction_b, label_b)


def loss_with_prediction(
    parameters_bp: jax.Array,
    feature_elapsed_days_real_bl: jax.Array,
    feature_rating_bl: jax.Array,
    seq_lens: jax.Array,
    mask_b: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    prediction_b = _forward(
        parameters_bp,
        feature_elapsed_days_real_bl,
        feature_rating_bl,
        seq_lens,
    )
    label_b = label_from_feature_rating(feature_rating_bl, seq_lens)
    return binary_cross_entropy_masked_sum(prediction_b, label_b, mask_b), prediction_b


loss_and_grad = jax.jit(jax.value_and_grad(loss, argnums=0))
loss_and_prediction_and_grad = jax.jit(
    jax.value_and_grad(loss_with_prediction, argnums=0, has_aux=True)
)


def get_initial_params_for_optimization() -> jax.Array:
    return jnp.zeros(35, dtype=jnp.float32)


FSRS7_DEFAULT_35 = jnp.array(
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
    ],
    dtype=jnp.float32,
)

FSRS_MIN = jnp.array(
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
    ],
    dtype=jnp.float32,
)

FSRS_MAX = jnp.array(
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
    ],
    dtype=jnp.float32,
)

assert bool(jnp.all(FSRS_MIN < FSRS7_DEFAULT_35))
assert bool(jnp.all(FSRS7_DEFAULT_35 < FSRS_MAX))
