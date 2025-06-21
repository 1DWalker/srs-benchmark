#pragma once

struct fsrs_state {
    // All fields should be floats, otherwise there is potential UB
    float s;
    float d;
};

float forgetting_curve(const float t, const float s, const float decay);
float stability_short_term(const float* params, const fsrs_state state, const int rating);
float stability_after_success(const float* params, const fsrs_state state, const float r, const int rating);
float stability_after_failure(const float* params, const fsrs_state state, const float r);
float init_d(const float* params, const int rating);
float linear_dampening(const float delta_d, const float old_d);
float mean_reversion(const float* params, const float init, const float current);
float next_d(const float* params, const fsrs_state state, const int rating);