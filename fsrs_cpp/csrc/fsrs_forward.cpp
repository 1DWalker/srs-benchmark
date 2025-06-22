#pragma once

#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <math.h>

template <typename F>
struct fsrs_state {
    F s;
    F d;
};

template <typename F>
F forgetting_curve(const F t, const F s, const F decay) {
    F factor = pow(0.9, 1 / -decay) - 1;
    return pow(1 + factor * t / s, -decay);
}

template <typename F>
F stability_short_term(const F* params, const fsrs_state<F> state, const int rating) {
    F sinc = exp(params[17] * (rating - 3 + params[18])) * pow(state.s, -params[19]);
    return state.s * (rating >= 3 ? std::max((F)1.0, sinc) : sinc);
}

template <typename F>
F stability_after_success(const F* params, const fsrs_state<F> state, const F r, const int rating) {
    F hard_penalty = rating == 2 ? params[15] : 1.0;
    F easy_bonus = rating == 4 ? params[16] : 1.0;
    F new_s = state.s * (
        1
        + exp(params[8])
        * (11 - state.d)
        * pow(state.s, -params[9])
        * (exp((1 - r) * params[10]) - 1)
        * hard_penalty
        * easy_bonus
    );
    return new_s;
}

template <typename F>
F stability_after_failure(const F* params, const fsrs_state<F> state, const F r) {
    F new_s = (
        params[11]
        * pow(state.d, -params[12])
        * (pow(state.s + 1, params[13]) - 1)
        * exp((1 - r) * params[14])
    );
    F new_minimum_s = state.s / exp(params[17] * params[18]);
    return std::min(new_s, new_minimum_s);
}

template <typename F>
F init_d(const F* params, const int rating) {
    return params[4] - exp(params[5] * (rating - 1)) + 1.0;
}

template <typename F>
F linear_dampening(const F delta_d, const F old_d) {
    return delta_d * (10 - old_d) / 9;
}

template <typename F>
F mean_reversion(const F* params, const F init, const F current) {
    return params[7] * init + (1 - params[7]) * current;
}

template <typename F>
F next_d(const F* params, const fsrs_state<F> state, const int rating) {
    F delta_d = -params[6] * (rating - 3);
    F new_d = state.d + linear_dampening(delta_d, state.d);
    return mean_reversion(params, init_d(params, 4), new_d);
}

template <typename F>
void fsrs6_forward(
    const int L,
    const F* params, 
    F* out_L,
    fsrs_state<F>* checkpoints_L,
    const int* rating,
    const F* elapsed_days_real,
    const F* elapsed_days_int,
    const F* label_elapsed_days_real,
    const F* label_elapsed_days_int
) {
    fsrs_state<F> state = {};
    for (int l = 0; l < L; l++) {
        checkpoints_L[l] = state;
        F new_s, new_d;
        if (l == 0) {
            new_s = params[rating[0] - 1]; 
            new_d = init_d(params, rating[0]);
        } else {
            F r = forgetting_curve(elapsed_days_int[l], state.s, params[20]);
            bool short_term = elapsed_days_int[l] < 1;
            bool success = rating[l] > 1;
            if (short_term) {  // Short term
                new_s = stability_short_term(params, state, rating[l]);
            } else {
                if (success) {
                    new_s = stability_after_success(params, state, r, rating[l]);
                } else {
                    new_s = stability_after_failure(params, state, r);
                }
            }
            new_d = next_d(params, state, rating[l]);
        }
        new_s = std::clamp(new_s, (F)0.001, (F)36500.0);
        new_d = std::clamp(new_d, (F)1.0, (F)10.0);
        state = {new_s, new_d};
        out_L[l] = forgetting_curve(label_elapsed_days_int[l], state.s, params[20]);
    }
}

template <typename F>
std::tuple<torch::Tensor, torch::Tensor> fsrs6_forward_verify(
    const at::Tensor& params_P,
    const at::Tensor& rating_L,
    const at::Tensor& elapsed_days_real_L,
    const at::Tensor& elapsed_days_int_L,
    const at::Tensor& label_elapsed_days_real_L,
    const at::Tensor& label_elapsed_days_int_L
) {
    const int L = rating_L.size(0);
    const F* params_P_ptr = params_P.data_ptr<F>();
    const int* rating_L_ptr = rating_L.data_ptr<int>();
    const F* elapsed_days_real_L_ptr = elapsed_days_real_L.data_ptr<F>();
    const F* elapsed_days_int_L_ptr = elapsed_days_int_L.data_ptr<F>();
    const F* label_elapsed_days_real_L_ptr = label_elapsed_days_real_L.data_ptr<F>();
    const F* label_elapsed_days_int_L_ptr = label_elapsed_days_int_L.data_ptr<F>();
    at::Tensor out_L = torch::empty(elapsed_days_real_L.sizes(), elapsed_days_real_L.options());
    F* out_L_ptr = out_L.data_ptr<F>();
    size_t num_chunks = (sizeof(fsrs_state<F>) + 4 - 1) / 4;
    at::Tensor checkpoints_L = torch::empty({L, (int)num_chunks}, elapsed_days_real_L.options().requires_grad(false));
    fsrs_state<F>* checkpoints_L_ptr = (fsrs_state<F>*)checkpoints_L.data_ptr();
    fsrs6_forward(L, params_P_ptr, out_L_ptr, checkpoints_L_ptr, rating_L_ptr, elapsed_days_real_L_ptr, elapsed_days_int_L_ptr, label_elapsed_days_real_L_ptr, label_elapsed_days_int_L_ptr);
    return {out_L, checkpoints_L};
}

namespace fsrs {
    TORCH_LIBRARY_IMPL(fsrs, CPU, m) {
        m.impl("fsrs6_forward_verify_float", &fsrs6_forward_verify<float>);
        m.impl("fsrs6_forward_verify_double", &fsrs6_forward_verify<double>);
    }
}