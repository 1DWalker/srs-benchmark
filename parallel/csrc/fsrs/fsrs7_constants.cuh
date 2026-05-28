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

static constexpr fsrs_params_t fsrs7_init_w = {
    .s0_again = 0.041f,
    .s0_hard = 2.4175f,
    .s0_good = 4.1283f,
    .s0_easy = 11.9709f,
    .init_d0 = 5.6385f,
    .init_d1 = 0.4468f,
    .next_d_mult = 3.262f,
    .long_stability = {
        .sinc_base = 2.3054f,
        .sinc_s_exp = 0.1688f,
        .sinc_r_mult = 1.3325f,
        .fail_mult = 0.3524f,
        .fail_d_exp = 0.0049f,
        .fail_s_exp = 0.7503f,
        .fail_r_mult = 0.0896f,
        .hard_penalty = 0.6625f,
        .easy_bonus = 1.3f,
    },
    .short_stability = {
        .sinc_base = 0.882f,
        .sinc_s_exp = 0.3072f,
        .sinc_r_mult = 3.5875f,
        .fail_mult = 0.303f,
        .fail_d_exp = 0.0107f,
        .fail_s_exp = 0.2279f,
        .fail_r_mult = 2.6413f,
        .hard_penalty = 0.5594f,
        .easy_bonus = 1.3f,
    },
    .transition_decay = 2.5f,
    .transition_scale = 1.0f,
    .decay1 = 0.0723f,
    .decay2 = 0.1634f,
    .base1 = 0.5f,
    .base2 = 0.9555f,
    .base_weight1 = 0.2245f,
    .base_weight2 = 0.6232f,
    .s_weight_power1 = 0.1362f,
    .s_weight_power2 = 0.3862f,
};

static constexpr fsrs_params_t fsrs7_sigma = {
    .s0_again = 9999.0f,
    .s0_hard = 9999.0f,
    .s0_good = 9999.0f,
    .s0_easy = 9999.0f,
    .init_d0 = 0.523f,
    .init_d1 = 0.2528f,
    .next_d_mult = 0.4329f,
    .long_stability = {
        .sinc_base = 0.2966f,
        .sinc_s_exp = 0.2139f,
        .sinc_r_mult = 0.2889f,
        .fail_mult = 0.1862f,
        .fail_d_exp = 0.0829f,
        .fail_s_exp = 0.175f,
        .fail_r_mult = 0.3812f,
        .hard_penalty = 0.3013f,
        .easy_bonus = 0.9104f,
    },
    .short_stability = {
        .sinc_base = 0.3234f,
        .sinc_s_exp = 0.2448f,
        .sinc_r_mult = 0.3273f,
        .fail_mult = 0.1842f,
        .fail_d_exp = 0.1542f,
        .fail_s_exp = 0.1735f,
        .fail_r_mult = 0.4608f,
        .hard_penalty = 0.311f,
        .easy_bonus = 0.864f,
    },
    .transition_decay = 0.4053f,
    .transition_scale = 0.162f,
    .decay1 = 0.0418f,
    .decay2 = 0.2596f,
    .base1 = 0.0798f,
    .base2 = 0.0682f,
    .base_weight1 = 0.1282f,
    .base_weight2 = 0.1397f,
    .s_weight_power1 = 0.1407f,
    .s_weight_power2 = 0.1489f,
};
