from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from parallel.models.fsrs_v7_constants import (
    FSRS7_DEFAULT_35_VALUES,
    FSRS7_L2_SIGMA_35_VALUES,
    FSRS_MAX_VALUES,
    FSRS_MIN_VALUES,
)


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
    return ((weight * r).sum(axis=-1) / weight.sum(axis=-1))


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


def binary_cross_entropy_masked_sum(
    prediction: jax.Array,
    label: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    label = label.astype(prediction.dtype)
    mask = mask.astype(prediction.dtype)
    loss_b = -(label * jnp.log(prediction) + (1 - label) * jnp.log1p(-prediction))
    return jnp.sum(loss_b * mask)


FSRS7_DEFAULT_35 = jnp.array(FSRS7_DEFAULT_35_VALUES, dtype=jnp.float32)
FSRS_MIN = jnp.array(FSRS_MIN_VALUES, dtype=jnp.float32)
FSRS_MAX = jnp.array(FSRS_MAX_VALUES, dtype=jnp.float32)
FSRS7_L2_SIGMA_35 = jnp.array(FSRS7_L2_SIGMA_35_VALUES, dtype=jnp.float32)

def fsrs7_l2_loss_term(
    parameters_bp: jax.Array,
    epoch_lens_b: jax.Array,
    mask_b: jax.Array | None = None,
) -> jax.Array:
    default = FSRS7_DEFAULT_35.astype(parameters_bp.dtype)
    sigma = FSRS7_L2_SIGMA_35.astype(parameters_bp.dtype)
    epoch_lens_b = epoch_lens_b.astype(parameters_bp.dtype)
    penalty_b = (
        jnp.sum(jnp.square(parameters_bp - default) / jnp.square(sigma), axis=-1)
        / epoch_lens_b
    )
    if mask_b is not None:
        penalty_b = penalty_b * mask_b.astype(parameters_bp.dtype)
    return jnp.sum(penalty_b)


def label_from_feature_rating(
    feature_rating_bl: jax.Array,
    seq_lens: jax.Array,
) -> jax.Array:
    batch_idx = jnp.arange(feature_rating_bl.shape[0])
    target_rating = feature_rating_bl[batch_idx, seq_lens.astype(jnp.int32) - 1]
    return (target_rating > 1).astype(jnp.float32)


def loss_with_prediction(
    parameters_bp: jax.Array,
    feature_elapsed_days_real_bl: jax.Array,
    feature_rating_bl: jax.Array,
    seq_lens: jax.Array,
    mask_b: jax.Array,
    epoch_lens_b: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    prediction_b = _forward(
        parameters_bp,
        feature_elapsed_days_real_bl,
        feature_rating_bl,
        seq_lens,
    )
    label_b = label_from_feature_rating(feature_rating_bl, seq_lens)
    return (
        binary_cross_entropy_masked_sum(prediction_b, label_b, mask_b)
        + fsrs7_l2_loss_term(parameters_bp, epoch_lens_b, mask_b),
        prediction_b,
    )


loss_and_prediction_and_grad = jax.jit(
    jax.value_and_grad(loss_with_prediction, argnums=0, has_aux=True)
)
