#pragma once

#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <math.h>
#include "fsrs_forward.h"


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
    at::Tensor out_grad_params = torch::empty(params_P.sizes(), params_P.options());
    float* out_grad_params_ptr = out_grad_params.data_ptr<float>();
    fsrs6_backward(L, grad_r_L_ptr, checkpoints_L_ptr, params_P_ptr, out_grad_params_ptr, rating_L_ptr, elapsed_days_real_L_ptr, elapsed_days_int_L_ptr, label_elapsed_days_real_L_ptr, label_elapsed_days_int_L_ptr);
    return out_grad_params;
}

namespace fsrs {
    TORCH_LIBRARY_IMPL(fsrs, CPU, m) {
        m.impl("fsrs6_backward_verify", &fsrs6_backward_verify);
    }
}