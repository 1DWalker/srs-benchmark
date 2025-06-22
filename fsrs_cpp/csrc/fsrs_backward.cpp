#pragma once

#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <math.h>
#include "fsrs_forward.h"

using fsrs_state_grad = fsrs_state;

void min_backward(float &grad_x, float &grad_y, const float x, const float y) {
    if (x <= y) {
        grad_y = 0;
    } else {
        grad_x = 0;
    }
}

void clamp_backward(float &grad_x, const float x, const float low, const float high) {
    if (x < low || high < x) {
        grad_x = 0;
    }
}

void forgetting_curve_backward(float* out_grad_params, fsrs_state_grad &grad_state, const float grad_r, const float t, const float s, const float decay) {
    const float factor = pow(0.9, 1 / -decay) - 1;
    const float inside = 1 + factor * t / s;
    const float base = pow(inside, -decay);
    out_grad_params[20] += -grad_r * base * log(inside);
    const float grad_inside = grad_r * -decay * base / inside;
    const float grad_factor = grad_inside * t / s;
    out_grad_params[20] += grad_factor * pow(0.9, 1 / -decay) * log(0.9) / pow(decay, 2);
    grad_state.s += grad_inside * factor * t / -pow(s, 2);
}

void stability_short_term_backward(float* out_grad_params, fsrs_state_grad &grad_state, const float grad_s, const float* params, const fsrs_state state, const int rating) {
    const float rating_mul = (rating - 3 + params[18]);
    const float t1 = exp(params[17] * rating_mul);
    const float t2 = pow(state.s, -params[19]);
    const float sinc = t1 * t2;
    // return state.s * (rating >= 3 ? std::max(1.0f, sinc) : sinc);
    grad_state.s += grad_s * (rating >= 3 ? std::max(1.0f, sinc) : sinc);
    const float grad_mul = grad_s * state.s;
    float grad_sinc;
    if (rating >= 3) {
        if (1.0f >= sinc) {
            grad_sinc = 0.0;
        } else {
            grad_sinc = grad_mul;
        }
    } else {
        grad_sinc = grad_mul;
    }
    // sinc = t1 * t2
    const float grad_t1 = grad_sinc * t2;
    const float grad_t1_inside = grad_t1 * t1;
    out_grad_params[17] += grad_t1_inside * rating_mul;
    out_grad_params[18] += grad_t1_inside * params[17];

    const float grad_t2 = grad_sinc * t1;
    grad_state.s += grad_t2 * -params[19] * t2 / state.s;
    out_grad_params[19] += -grad_t2 * t2 * log(state.s);
}

void stability_after_success_backward(float* out_grad_params, fsrs_state_grad &grad_state, float &grad_r, const float grad_s, const float* params, const fsrs_state state, const float r, const int rating) {
    const float hard_penalty = rating == 2 ? params[15] : 1.0;
    const float easy_bonus = rating == 4 ? params[16] : 1.0;
    const int BINS = 6;
    float vals[BINS];
    vals[0] = exp(params[8]);
    vals[1] = (11 - state.d);
    vals[2] = pow(state.s, -params[9]);
    vals[3] = exp((1 - r) * params[10]) - 1;
    vals[4] = hard_penalty;
    vals[5] = easy_bonus;

    float preprod[BINS], sufprod[BINS];
    for (int i = 0; i < BINS; i++) {
        preprod[i] = vals[i] * (i == 0 ? 1.0 : preprod[i - 1]);
    }
    for (int i = BINS - 1; i >= 0; i--) {
        sufprod[i] = vals[i] * (i == BINS - 1 ? 1.0 : sufprod[i + 1]);
    }

    grad_state.s += grad_s * (1 + sufprod[0]);
    const float grad_sinc = grad_s * state.s;
    out_grad_params[8]  += grad_sinc * sufprod[1] * vals[0];
    grad_state.d        += -grad_sinc * preprod[0] * sufprod[2];
    grad_state.s        += -grad_sinc * preprod[1] * sufprod[3] * params[9] * vals[2] / state.s;
    out_grad_params[9]  += -grad_sinc * preprod[1] * sufprod[3] * vals[2] * log(state.s);
    const float grad_1_minus_r_times_params10 = grad_sinc * preprod[2] * sufprod[4] * (vals[3] + 1);
    grad_r              += -grad_1_minus_r_times_params10 * params[10];
    out_grad_params[10] += grad_1_minus_r_times_params10 * (1 - r);
    if (rating == 2) {
        out_grad_params[15] += grad_sinc * preprod[3] * sufprod[5];
    }
    if (rating == 4) {
        out_grad_params[16] += grad_sinc * preprod[4];
    }
}

void stability_after_failure_backward(float* out_grad_params, fsrs_state_grad &grad_state, float &grad_r, const float grad_s, const float* params, const fsrs_state state, const float r) {
    const int BINS = 4;
    float vals[BINS];
    vals[0] = params[11];
    vals[1] = pow(state.d, -params[12]);
    vals[2] = (pow(state.s + 1, params[13]) - 1);
    vals[3] = exp((1 - r) * params[14]);

    float preprod[BINS], sufprod[BINS];
    for (int i = 0; i < BINS; i++) {
        preprod[i] = vals[i] * (i == 0 ? 1.0 : preprod[i - 1]);
    }
    for (int i = BINS - 1; i >= 0; i--) {
        sufprod[i] = vals[i] * (i == BINS - 1 ? 1.0 : sufprod[i + 1]);
    }

    const float new_s = sufprod[0];
    const float exp_17_times_18 = exp(params[17] * params[18]);
    const float new_minimum_s = state.s / exp_17_times_18;

    // Backward
    if (new_s <= new_minimum_s) {
        out_grad_params[11] += grad_s * sufprod[1];
        out_grad_params[12] -= grad_s * preprod[0] * sufprod[2] * vals[1] * log(state.d);
        grad_state.d        += grad_s * preprod[0] * sufprod[2] * -params[12] * pow(state.d, -params[12] - 1);
        out_grad_params[13] += grad_s * preprod[1] * sufprod[3] * (vals[2] + 1) * log(state.s + 1);
        grad_state.s        += grad_s * preprod[1] * sufprod[3] * params[13] * pow(state.s + 1, params[13] - 1);
        out_grad_params[14] += grad_s * preprod[2] * vals[3] * (1 - r);
        grad_r              -= grad_s * preprod[2] * vals[3] * params[14];
    } else {
        float grad_params17_times_params18 = -grad_s * state.s / exp_17_times_18;
        out_grad_params[17] += grad_params17_times_params18 * params[18];
        out_grad_params[18] += grad_params17_times_params18 * params[17];
        grad_state.s += grad_s / exp_17_times_18;
    }
}

void init_d_backward(float* out_grad_params, const float grad, const float* params, int rating) {
    out_grad_params[4] += grad;
    out_grad_params[5] += -grad * exp(params[5] * (rating - 1)) * (rating - 1);
}

void linear_dampening_backward(float &grad_delta_d, float &grad_old_d, const float grad, const float delta_d, const float old_d) {
    grad_delta_d += grad * (10 - old_d) / 9;
    grad_old_d += -grad * delta_d / 9;
}

void mean_reversion_backward(float* out_grad_params, float &grad_init, float &grad_current, const float grad, const float* params, const float init, const float current) {
    out_grad_params[7] += grad * (init - current);
    grad_init += grad * params[7];
    grad_current += grad * (1 - params[7]);
}

void next_d_backward(float* out_grad_params, fsrs_state_grad &grad_state, const float grad_d, const float* params, const fsrs_state state, const int rating) {
    const float delta_d = -params[6] * (rating - 3);
    const float new_d = state.d + linear_dampening(delta_d, state.d);

    float grad_init = 0.0, grad_new_d = 0.0;
    mean_reversion_backward(out_grad_params, grad_init, grad_new_d, grad_d, params, init_d(params, 4), new_d);
    init_d_backward(out_grad_params, grad_init, params, 4);
    float grad_delta_d = 0.0;
    grad_state.d += grad_new_d;
    linear_dampening_backward(grad_delta_d, grad_state.d, grad_new_d, delta_d, state.d);
    out_grad_params[6] += -grad_delta_d * (rating - 3);
}

void fsrs6_backward(
    const int L,
    const float* grad_r_L,
    const fsrs_state* checkpoints_L,
    const float* params, 
    float* out_grad_params,
    const int* rating,
    const float* elapsed_days_real,
    const float* elapsed_days_int,
    const float* label_elapsed_days_real,
    const float* label_elapsed_days_int
) {
    fsrs_state_grad grad_state = {};
    for (int l = L - 1; l >= 0; l--) {
        // Starting from the checkpoint, redo a forward pass to recompute intermediate values
        fsrs_state state = checkpoints_L[l];
        float new_s, new_d, r;
        bool short_term, success;
        if (l == 0) {
            new_s = params[rating[0] - 1]; 
            new_d = init_d(params, rating[0]);
        } else {
            r = forgetting_curve(elapsed_days_int[l], state.s, params[20]);
            short_term = elapsed_days_int[l] < 1;
            success = rating[l] > 1;
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
        float new_s_clamped = std::clamp(new_s, 0.001f, 36500.0f);
        float new_d_clamped = std::clamp(new_d, 1.0f, 10.0f);
        fsrs_state new_state = {new_s_clamped, new_d_clamped};

        // Backward pass begins
        forgetting_curve_backward(out_grad_params, grad_state, grad_r_L[l], label_elapsed_days_int[l], new_state.s, params[20]);
        clamp_backward(grad_state.s, new_s, 0.001f, 36500.0f);
        clamp_backward(grad_state.d, new_d, 1.0f, 10.0f);

        if (l == 0) {
            out_grad_params[rating[0] - 1] = grad_state.s;
            init_d_backward(out_grad_params, grad_state.d, params, rating[0]);
        } else {
            fsrs_state_grad new_grad_state = {};
            float grad_r = 0.0;
            if (short_term) {
                stability_short_term_backward(out_grad_params, new_grad_state, grad_state.s, params, state, rating[l]);
            } else {
                if (success) {
                    stability_after_success_backward(out_grad_params, new_grad_state, grad_r, grad_state.s, params, state, r, rating[l]);
                } else {
                    stability_after_failure_backward(out_grad_params, new_grad_state, grad_r, grad_state.s, params, state, r);
                }
            }
            forgetting_curve_backward(out_grad_params, new_grad_state, grad_r, elapsed_days_int[l], state.s, params[20]);
            next_d_backward(out_grad_params, new_grad_state, grad_state.d, params, state, rating[l]);

            grad_state = new_grad_state;
        }
    }
}

torch::Tensor fsrs6_backward_verify(
    const at::Tensor& grad_r_L,
    const at::Tensor& checkpoints_L,
    const at::Tensor& params_P,
    const at::Tensor& rating_L,
    const at::Tensor& elapsed_days_real_L,
    const at::Tensor& elapsed_days_int_L,
    const at::Tensor& label_elapsed_days_real_L,
    const at::Tensor& label_elapsed_days_int_L
) {
    const int L = rating_L.size(0);
    const float* grad_r_L_ptr = grad_r_L.data_ptr<float>();
    const fsrs_state* checkpoints_L_ptr = (fsrs_state*)checkpoints_L.data_ptr();
    const float* params_P_ptr = params_P.data_ptr<float>();
    const int* rating_L_ptr = rating_L.data_ptr<int>();
    const float* elapsed_days_real_L_ptr = elapsed_days_real_L.data_ptr<float>();
    const float* elapsed_days_int_L_ptr = elapsed_days_int_L.data_ptr<float>();
    const float* label_elapsed_days_real_L_ptr = label_elapsed_days_real_L.data_ptr<float>();
    const float* label_elapsed_days_int_L_ptr = label_elapsed_days_int_L.data_ptr<float>();
    at::Tensor out_grad_params = torch::zeros(params_P.sizes(), params_P.options());
    float* out_grad_params_ptr = out_grad_params.data_ptr<float>();
    fsrs6_backward(L, grad_r_L_ptr, checkpoints_L_ptr, params_P_ptr, out_grad_params_ptr, rating_L_ptr, elapsed_days_real_L_ptr, elapsed_days_int_L_ptr, label_elapsed_days_real_L_ptr, label_elapsed_days_int_L_ptr);
    return out_grad_params;
}

namespace fsrs {
    TORCH_LIBRARY_IMPL(fsrs, CPU, m) {
        m.impl("fsrs6_backward_verify", &fsrs6_backward_verify);
    }
}