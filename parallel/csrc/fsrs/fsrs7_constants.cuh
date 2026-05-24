#pragma once

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
    float long_sinc_base;
    float long_sinc_s_exp;
    float long_sinc_r_mult;
    float long_fail_mult;
    float long_fail_d_exp;
    float long_fail_s_exp;
    float long_fail_r_mult;
    float long_hard_penalty;
    float long_easy_bonus;

    // 16..24: Short-term stability after review.
    float short_sinc_base;
    float short_sinc_s_exp;
    float short_sinc_r_mult;
    float short_fail_mult;
    float short_fail_d_exp;
    float short_fail_s_exp;
    float short_fail_r_mult;
    float short_hard_penalty;
    float short_easy_bonus;

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
    .long_sinc_base = 2.3054f,
    .long_sinc_s_exp = 0.1688f,
    .long_sinc_r_mult = 1.3325f,
    .long_fail_mult = 0.3524f,
    .long_fail_d_exp = 0.0049f,
    .long_fail_s_exp = 0.7503f,
    .long_fail_r_mult = 0.0896f,
    .long_hard_penalty = 0.6625f,
    .long_easy_bonus = 1.3f,
    .short_sinc_base = 0.882f,
    .short_sinc_s_exp = 0.3072f,
    .short_sinc_r_mult = 3.5875f,
    .short_fail_mult = 0.303f,
    .short_fail_d_exp = 0.0107f,
    .short_fail_s_exp = 0.2279f,
    .short_fail_r_mult = 2.6413f,
    .short_hard_penalty = 0.5594f,
    .short_easy_bonus = 1.3f,
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
    .long_sinc_base = 0.2966f,
    .long_sinc_s_exp = 0.2139f,
    .long_sinc_r_mult = 0.2889f,
    .long_fail_mult = 0.1862f,
    .long_fail_d_exp = 0.0829f,
    .long_fail_s_exp = 0.175f,
    .long_fail_r_mult = 0.3812f,
    .long_hard_penalty = 0.3013f,
    .long_easy_bonus = 0.9104f,
    .short_sinc_base = 0.3234f,
    .short_sinc_s_exp = 0.2448f,
    .short_sinc_r_mult = 0.3273f,
    .short_fail_mult = 0.1842f,
    .short_fail_d_exp = 0.1542f,
    .short_fail_s_exp = 0.1735f,
    .short_fail_r_mult = 0.4608f,
    .short_hard_penalty = 0.311f,
    .short_easy_bonus = 0.864f,
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
