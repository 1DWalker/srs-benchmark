#pragma once

#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <math.h>
#include "fsrs_forward.h"

using fsrs_state_grad = fsrs_state;

void forgetting_curve_backward(float* out_grad_params, fsrs_state_grad &grad_state, const float grad_r, const float t, const float s, const float decay) {
    float factor = pow(0.9, 1 / -decay) - 1;
    double inside = 1 + factor * t / s;
    // r = pow(1 + factor * t / s, -decay);
    // dr/ddecay = dr/ddecay e^(-decay * log(1 + factor * t / s))
    //           =  e^(-decay * log(1 + factor * t / s)) * d/ddecay (-decay * log(1 + factor * t / s))
    //           = d0 * d1
    float d0 = pow(1 + factor * t / s, -decay);
    // d1 = d/ddecay (-decay * log(1 + factor * t / s))
    //    = -decay * d/ddecay log(1 + factor * t / s) + (-1) * log(1 + factor * t / s)
    //    = d11 + d12
    // d/ddecay log(1 + factor * t / s) = 1 / (1 + factor * t / s) * t / s * d/ddecay factor
    // d/ddecay factor = 0.9^(1 / -decay) * log(0.9) * 1 / decay^2
    float d11 = -decay / (1 + factor * t / s) * t / s * pow(0.9, 1 / -decay) * log(0.9) / pow(decay, 2);
    float d12 = -log(inside);
    float d1 = d11 + d12;
    // float decay_intermediate = -decay / inside * t / s * pow(0.9, 1 / -decay) * pow(decay, -2);
    out_grad_params[20] += grad_r * d0 * d1;
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
        float new_s, new_d;
        if (l == 0) {
            new_s = params[rating[0] - 1]; 
            new_d = init_d(params, rating[0]);
        } else {
            float r = forgetting_curve(elapsed_days_int[l], state.s, params[20]);
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
        new_s = std::clamp(new_s, 0.001f, 36500.0f);
        new_d = std::clamp(new_d, 1.0f, 10.0f);
        state = {new_s, new_d};
        std::cout << "state back" << new_s << ' ' << new_d << '\n';
        // out_L[l] = forgetting_curve(label_elapsed_days_int[l], state.s, params[20]);

        // Backwards pass
        forgetting_curve_backward(out_grad_params, grad_state, grad_r_L[l], label_elapsed_days_int[l], state.s, params[20]);
        if (l == 0) {

        } else {

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