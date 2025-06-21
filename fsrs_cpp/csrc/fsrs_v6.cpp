#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <math.h>

struct fsrs_state {
    float s;
    float d;
};

float init_d(const float* params, const int rating) {
    return std::clamp(params[4] - exp(params[5] * (rating - 1)) + 1, 1.0f, 10.0f);
}

float forgetting_curve(const float t, const float s, const float decay) {
    std::cout << "forgetting " << t << ' ' << s << ' ' << decay << '\n';
    float factor = pow(0.9, 1 / -decay) - 1;
    return pow(1 + factor * t / s, -decay);
}

void fsrs6_forward(
    const int L,
    const float* params, 
    float* out_L,
    fsrs_state* checkpoints_L,
    const int* rating,
    const float* elapsed_days_real,
    const float* elapsed_days_int,
    const float* label_elapsed_days_real,
    const float* label_elapsed_days_int
) {
    fsrs_state state;
    for (int l = 0; l < L; l++) {
        float new_s, new_d;
        if (l == 0) {
            new_s = params[rating[0] - 1]; 
            new_d = init_d(params, rating[0]);
        } else {
            new_s = 1.0;
            new_d = 5.0;
        }
        state = {new_s, new_d};
        out_L[l] = forgetting_curve(label_elapsed_days_int[l], state.s, params[20]);
        std::cout << "done " << out_L[l] << '\n';
        
        break;
    }
}

torch::Tensor fsrs6_forward_verify(
    const at::Tensor& params_P,
    const at::Tensor& rating_L,
    const at::Tensor& elapsed_days_real_L,
    const at::Tensor& elapsed_days_int_L,
    const at::Tensor& label_elapsed_days_real_L,
    const at::Tensor& label_elapsed_days_int_L
) {
    const int L = rating_L.size(0);
    const float* params_P_ptr = params_P.data_ptr<float>();
    const int* rating_L_ptr = rating_L.data_ptr<int>();
    const float* elapsed_days_real_L_ptr = elapsed_days_real_L.data_ptr<float>();
    const float* elapsed_days_int_L_ptr = elapsed_days_int_L.data_ptr<float>();
    const float* label_elapsed_days_real_L_ptr = label_elapsed_days_real_L.data_ptr<float>();
    const float* label_elapsed_days_int_L_ptr = label_elapsed_days_int_L.data_ptr<float>();
    at::Tensor out_L = torch::empty(elapsed_days_real_L.sizes(), elapsed_days_real_L.options());
    float* out_L_ptr = out_L.data_ptr<float>();
    // at::Tensor checkpoints_L = torch::empty({L, 2}, elapsed_days_real_L.options());
    // const float* checkpoints_L_ptr = out_L.data_ptr<float>();
    fsrs6_forward(L, params_P_ptr, out_L_ptr, nullptr, rating_L_ptr, elapsed_days_real_L_ptr, elapsed_days_int_L_ptr, label_elapsed_days_real_L_ptr, label_elapsed_days_int_L_ptr);
    // fsrs_state next_state = fsrs6_forward(params_P_ptr, state_L2_ptr[0], state_L2_ptr[1], input_3_ptr[0], input_3_ptr[1], input_3_ptr[2]);
    // return torch::tensor({next_state.s, next_state.d});
    return out_L;
}

namespace fsrs {
    TORCH_LIBRARY_IMPL(fsrs, CPU, m) {
        m.impl("fsrs6_forward_verify", &fsrs6_forward_verify);
        // m.impl("fsrs6_backward", &);
    }
}