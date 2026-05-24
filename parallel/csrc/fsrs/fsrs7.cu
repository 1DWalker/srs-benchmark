#include <cuda_runtime.h>
#include <stdint.h>

#include "fsrs7_constants.cuh"

struct fsrs_state_t {
    float s; 
    float d;
};

__device__ __forceinline__
float fsrs7_clamp(const float x, const float lo, const float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ __forceinline__
fsrs_state_t fsrs7_clamp_state(const float stability, const float difficulty) {
    return fsrs_state_t{
        fsrs7_clamp(stability, 1e-4f, 36500.0f),
        fsrs7_clamp(difficulty, 1.0f, 10.0f),
    };
}

__device__ __forceinline__
float fsrs7_initial_difficulty(
    const fsrs_params_t &fsrs_params,
    const float rating
) {
    return fsrs_params.init_d0 - expf(fsrs_params.init_d1 * (rating - 1.0f)) + 1.0f;
}

__device__ __forceinline__
float fsrs7_linear_damping(const float delta_d, const float old_d) {
    return delta_d * (10.0f - old_d) / 9.0f;
}

__device__ __forceinline__
float fsrs7_mean_reversion(const float init, const float current) {
    return 0.01f * init + 0.99f * current;
}

__device__ __forceinline__
float fsrs7_next_d(
    const fsrs_params_t &fsrs_params,
    const fsrs_state_t fsrs_state,
    const int8_t rating
) {
    const float delta_d = -fsrs_params.next_d_mult * (static_cast<float>(rating) - 3.0f);
    const float new_d = fsrs_state.d + fsrs7_linear_damping(delta_d, fsrs_state.d);
    return fsrs7_mean_reversion(fsrs7_initial_difficulty(fsrs_params, 4.0f), new_d);
}

__device__ __forceinline__
float fsrs7_forgetting_curve(
    const fsrs_params_t &fsrs_params,
    const float elapsed_time,
    const float stability
) {
    const float t_over_s = elapsed_time / stability;

    const float decay1 = -fsrs_params.decay1;
    const float factor1 = powf(fsrs_params.base1, 1.0f / decay1) - 1.0f;
    const float r1 = powf(1.0f + factor1 * t_over_s, decay1);

    const float decay2 = -fsrs_params.decay2;
    const float factor2 = powf(fsrs_params.base2, 1.0f / decay2) - 1.0f;
    const float r2 = powf(1.0f + factor2 * t_over_s, decay2);

    const float weight1 = fsrs_params.base_weight1 * powf(stability, -fsrs_params.s_weight_power1);
    const float weight2 = fsrs_params.base_weight2 * powf(stability, fsrs_params.s_weight_power2);
    const float retention = (weight1 * r1 + weight2 * r2) / (weight1 + weight2);

    return 1e-5f + (1.0f - 2e-5f) * retention;
}

__device__ __forceinline__
float fsrs7_stability_after_review_one_term(
    const float old_s,
    const float old_d,
    const float retention,
    const int8_t rating,
    const float sinc_base,
    const float sinc_s_exp,
    const float sinc_r_mult,
    const float fail_mult,
    const float fail_d_exp,
    const float fail_s_exp,
    const float fail_r_mult,
    const float hard_penalty_param,
    const float easy_bonus_param
) {
    const float hard_penalty = rating == 2 ? hard_penalty_param : 1.0f;
    const float easy_bonus = rating == 4 ? easy_bonus_param : 1.0f;

    const float new_s_fail =
        fail_mult
        * powf(old_d, -fail_d_exp)
        * (powf(old_s + 1.0f, fail_s_exp) - 1.0f)
        * expf((1.0f - retention) * fail_r_mult);
    const float pls = fminf(old_s, new_s_fail);

    const float s_inc =
        1.0f
        + expf(sinc_base - 1.5f)
        * (11.0f - old_d)
        * powf(old_s, -sinc_s_exp)
        * (expf((1.0f - retention) * sinc_r_mult) - 1.0f)
        * hard_penalty
        * easy_bonus;
    const float new_s_success = fmaxf(pls, old_s * s_inc);

    return rating > 1 ? new_s_success : pls;
}

__device__
fsrs_state_t fsrs7_init(
    const fsrs_params_t &fsrs_params,
    const int8_t first_rating
) {
    float initial_stability;
    switch (first_rating) {
        case 2:
            initial_stability = fsrs_params.s0_hard;
            break;
        case 3:
            initial_stability = fsrs_params.s0_good;
            break;
        case 4:
            initial_stability = fsrs_params.s0_easy;
            break;
        case 1:
        default:
            initial_stability = fsrs_params.s0_again;
            break;
    }

    const float initial_difficulty = fsrs7_initial_difficulty(
        fsrs_params,
        static_cast<float>(first_rating)
    );

    return fsrs7_clamp_state(initial_stability, initial_difficulty);
}

__device__
fsrs_state_t fsrs7_step(
    const fsrs_params_t &fsrs_params,
    const fsrs_state_t fsrs_state,
    const float elapsed_time,
    const int8_t rating
) {
    const float retention = fsrs7_forgetting_curve(
        fsrs_params,
        elapsed_time,
        fsrs_state.s
    );

    const float long_stability = fsrs7_stability_after_review_one_term(
        fsrs_state.s,
        fsrs_state.d,
        retention,
        rating,
        fsrs_params.long_sinc_base,
        fsrs_params.long_sinc_s_exp,
        fsrs_params.long_sinc_r_mult,
        fsrs_params.long_fail_mult,
        fsrs_params.long_fail_d_exp,
        fsrs_params.long_fail_s_exp,
        fsrs_params.long_fail_r_mult,
        fsrs_params.long_hard_penalty,
        fsrs_params.long_easy_bonus
    );

    const float short_stability = fsrs7_stability_after_review_one_term(
        fsrs_state.s,
        fsrs_state.d,
        retention,
        rating,
        fsrs_params.short_sinc_base,
        fsrs_params.short_sinc_s_exp,
        fsrs_params.short_sinc_r_mult,
        fsrs_params.short_fail_mult,
        fsrs_params.short_fail_d_exp,
        fsrs_params.short_fail_s_exp,
        fsrs_params.short_fail_r_mult,
        fsrs_params.short_hard_penalty,
        fsrs_params.short_easy_bonus
    );

    const float coefficient =
        1.0f - fsrs_params.transition_scale * expf(-fsrs_params.transition_decay * elapsed_time);
    const float new_s = short_stability + coefficient * (long_stability - short_stability);
    const float new_d = fsrs7_next_d(fsrs_params, fsrs_state, rating);

    return fsrs7_clamp_state(new_s, new_d);
}
