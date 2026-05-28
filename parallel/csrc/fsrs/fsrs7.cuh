#pragma once

struct fsrs_state_t {
    float s;
    float d;
};

struct fsrs_stability_after_review_params_t {
    float sinc_base;
    float sinc_s_exp;
    float sinc_r_mult;
    float fail_mult;
    float fail_d_exp;
    float fail_s_exp;
    float fail_r_mult;
    float hard_penalty;
    float easy_bonus;
};

struct fsrs_params_t {
    // 0..3: Initial stability by first rating.
    float s0_again;
    float s0_hard;
    float s0_good;
    float s0_easy;

    // 4..6: Difficulty.
    float init_d0;
    float init_d1;
    float next_d_mult;

    // 7..15: Long-term stability after review.
    fsrs_stability_after_review_params_t long_stability;

    // 16..24: Short-term stability after review.
    fsrs_stability_after_review_params_t short_stability;

    // 25..26: Long-short term transition function.
    float transition_decay;
    float transition_scale;

    // 27..34: Forgetting curve.
    float decay1;
    float decay2;
    float base1;
    float base2;
    float base_weight1;
    float base_weight2;
    float s_weight_power1;
    float s_weight_power2;
};