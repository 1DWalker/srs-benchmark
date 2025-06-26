#pragma once

template <typename F>
struct fsrs_state {
    F s;
    F d;
};

template <typename F>
struct checkpoint_t {
    fsrs_state<F> start_state;
    F r;
    // new_s and new_d are prior to clamping
    F new_s;
    F new_d;
    fsrs_state<F> new_state;
};