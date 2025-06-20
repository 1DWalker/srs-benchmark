#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

std::tuple<float, float> fsrs6_forward(
    const float* params, 
    const float s, 
    const float d, 
    const int rating,
    const float elapsed_days_real, 
    const float elapsed_days_int
    ) {
    return {3.14, 3.14};
}

torch::Tensor fsrs6_forward_verify(
    const at::Tensor& params_P,
    const at::Tensor& state_2,
    const at::Tensor& input_3
) {
    const float* params_P_ptr = params_P.data_ptr<float>();
    const float* state_2_ptr = state_2.data_ptr<float>();
    const float* input_3_ptr = input_3.data_ptr<float>();
    auto [new_s, new_d] = fsrs6_forward(params_P_ptr, state_2_ptr[0], state_2_ptr[1], input_3_ptr[0], input_3_ptr[1], input_3_ptr[2]);
    return torch::tensor({new_s, new_d});
}

namespace fsrs {
    TORCH_LIBRARY_IMPL(fsrs, CPU, m) {
        m.impl("fsrs6_forward_verify", &fsrs6_forward_verify);
        // m.impl("fsrs6_backward", &);
    }
}