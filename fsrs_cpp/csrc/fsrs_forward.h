#pragma once

template <typename F>
struct fsrs_state {
    // All fields should be floats
    F s;
    F d;
};

// template <typename F>
// F forgetting_curve(const float t, const float s, const float decay);
// template <typename F>
// F stability_short_term(const float* params, const fsrs_state state, const int rating);
// template <typename F>
// F stability_after_success(const float* params, const fsrs_state state, const float r, const int rating);
// template <typename F>
// F stability_after_failure(const float* params, const fsrs_state state, const float r);
// template <typename F>
// F init_d(const float* params, const int rating);
// template <typename F>
// F linear_dampening(const float delta_d, const float old_d);
// template <typename F>
// F mean_reversion(const float* params, const float init, const float current);
// template <typename F>
// F next_d(const float* params, const fsrs_state state, const int rating);