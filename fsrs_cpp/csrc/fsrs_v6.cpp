#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <math.h>

struct fsrs_state {
    // All fields should be floats, otherwise there is potential UB
    float s;
    float d;
};

float forgetting_curve(const float t, const float s, const float decay) {
    float factor = pow(0.9, 1 / -decay) - 1;
    return pow(1 + factor * t / s, -decay);
}

float stability_short_term(const float* params, const fsrs_state state, const int rating) {
    float sinc = exp(params[17] * (rating - 3 + params[18])) * pow(state.s, -params[19]);
    return state.s * (rating >= 3 ? std::max(1.0f, sinc) : sinc);
}

float stability_after_success(const float* params, const fsrs_state state, const float r, const int rating) {
    float hard_penalty = rating == 2 ? params[15] : 1.0;
    float easy_bonus = rating == 4 ? params[16] : 1.0;
    float new_s = state.s * (
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

float stability_after_failure(const float* params, const fsrs_state state, const float r) {
    float new_s = (
        params[11]
        * pow(state.d, -params[12])
        * (pow(state.s + 1, params[13]) - 1)
        * exp((1 - r) * params[14])
    );
    float new_minimum_s = state.s / exp(params[17] * params[18]);
    return std::min(new_s, new_minimum_s);
}


float init_d(const float* params, const int rating) {
    return params[4] - exp(params[5] * (rating - 1)) + 1.0;
}

float linear_dampening(const float delta_d, const float old_d) {
    return delta_d * (10 - old_d) / 9;
}

float mean_reversion(const float* params, const float init, const float current) {
    return params[7] * init + (1 - params[7]) * current;
}

float next_d(const float* params, const fsrs_state state, const int rating) {
    float delta_d = -params[6] * (rating - 3);
    float new_d = state.d + linear_dampening(delta_d, state.d);
    return mean_reversion(params, init_d(params, 4), new_d);
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
    fsrs_state state = {};
    for (int l = 0; l < L; l++) {
        checkpoints_L[l] = state;
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
        out_L[l] = forgetting_curve(label_elapsed_days_int[l], state.s, params[20]);
    }
}

std::tuple<torch::Tensor, torch::Tensor> fsrs6_forward_verify(
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
    size_t num_chunks = (sizeof(fsrs_state) + 4 - 1) / 4;
    at::Tensor checkpoints_L = torch::empty({L, (int)num_chunks}, elapsed_days_real_L.options().requires_grad(false));
    fsrs_state* checkpoints_L_ptr = (fsrs_state*)checkpoints_L.data_ptr();
    fsrs6_forward(L, params_P_ptr, out_L_ptr, checkpoints_L_ptr, rating_L_ptr, elapsed_days_real_L_ptr, elapsed_days_int_L_ptr, label_elapsed_days_real_L_ptr, label_elapsed_days_int_L_ptr);
    return {out_L, checkpoints_L};
}

namespace fsrs {
    TORCH_LIBRARY_IMPL(fsrs, CPU, m) {
        m.impl("fsrs6_forward_verify", &fsrs6_forward_verify);
        // m.impl("fsrs6_backward", &);
    }
}