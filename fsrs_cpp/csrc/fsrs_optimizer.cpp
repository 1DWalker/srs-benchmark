#pragma once

#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <math.h>
#include "fsrs.h"
#include "fsrs_forward.cpp"
#include "fsrs_backward.cpp"
#include "adam.cpp"

int accum = 0;
const std::vector<float> initial_params = {
    0.212f,
    1.2931f,
    2.3065f,
    8.2956f,
    6.4133f,
    0.8334f,
    3.0194f,
    0.001f,
    1.8722f,
    0.1666f,
    0.796f,
    1.4835f,
    0.0614f,
    0.2629f,
    1.6483f,
    0.6014f,
    1.8729f,
    0.5425f,
    0.0912f,
    0.0658f,
    0.1542f,
};

const std::vector<float> default_params_stddev = {
    6.43f,
    9.66f,
    17.58f,
    27.85f,
    0.57f,
    0.28f,
    0.6f,
    0.12f,
    0.39f,
    0.18f,
    0.33f,
    0.3f,
    0.09f,
    0.16f,
    0.57f,
    0.25f,
    1.03f,
    0.31f,
    0.32f,
    0.14f,
    0.27f,
};

void clip_params(std::vector<float> &params) {
    const float S_MIN = 0.001f;
    const float INIT_S_MAX = 100.0f;
    params[0] = std::clamp(params[0], S_MIN, INIT_S_MAX);
    params[1] = std::clamp(params[1], S_MIN, INIT_S_MAX);
    params[2] = std::clamp(params[2], S_MIN, INIT_S_MAX);
    params[3] = std::clamp(params[3], S_MIN, INIT_S_MAX);
    params[4] = std::clamp(params[4], 1.0f, 10.0f);
    params[5] = std::clamp(params[5], 0.001f, 4.0f);
    params[6] = std::clamp(params[6], 0.001f, 4.0f);
    params[7] = std::clamp(params[7], 0.001f, 0.75f);
    params[8] = std::clamp(params[8], 0.0f, 4.5f);
    params[9] = std::clamp(params[9], 0.0f, 0.8f);
    params[10] = std::clamp(params[10], 0.001f, 3.5f);
    params[11] = std::clamp(params[11], 0.001f, 5.0f);
    params[12] = std::clamp(params[12], 0.001f, 0.25f);
    params[13] = std::clamp(params[13], 0.001f, 0.9f);
    params[14] = std::clamp(params[14], 0.0f, 4.0f);
    params[15] = std::clamp(params[15], 0.0f, 1.0f);
    params[16] = std::clamp(params[16], 1.0f, 6.0f);
    params[17] = std::clamp(params[17], 0.0f, 2.0f);
    params[18] = std::clamp(params[18], 0.0f, 2.0f);
    params[19] = std::clamp(params[19], 0.0f, 0.8f);
    params[20] = std::clamp(params[20], 0.1f, 0.8f);
}

float get_lr(int step, int total_steps) {
    const float pi = 3.141592653589f;
    return 4e-2 * 0.5 * (1 + std::cos((float) step / total_steps * pi));
}

float bce_loss(float input, float target) {
    return -(target * log(input) + (1 - target) * log(1 - input));
}

float bce_loss_grad(float input, float target) {
    return (input - target) / (input * (1 - input));
}

struct key_t {
    int start_loc;
    int l;
    int r;
    int L;
};

template <bool requires_grad, bool store_out>
float run_batch(
    std::vector<float> &param_grad_buffer,
    std::vector<float> &param_grad_buffer_2,
    std::vector<float> &out_card_buffer,
    std::vector<float> &y_buffer,
    std::vector<float> &y_pred_buffer,
    std::vector<float> &r_grad_buffer,
    std::vector<checkpoint_t<float>> &checkpoint_buffer,
    const std::vector<float> &params,
    const int num_keys,
    const key_t* keys_ptr,
    const int train_size,
    const int* train_ords_ptr,
    const int* locs_ptr,
    const int* packed_rating,
    const float* packed_elapsed_days_real,
    const float* packed_elapsed_days_int,
    const float* packed_label_elapsed_days_real,
    const float* packed_label_elapsed_days_int
) {
    // Both forward and backward passes are performed one by one for better cache locality
    float loss = 0;
    int y_buffer_offset = 0;
    for (int key_i = 0; key_i < num_keys; key_i++) {
        key_t key = keys_ptr[key_i];
        fsrs6_forward<float, requires_grad>(
            key.L,
            params.data(),
            out_card_buffer.data(),
            checkpoint_buffer.data(),
            packed_rating + key.start_loc,
            packed_elapsed_days_real + key.start_loc,
            packed_elapsed_days_int + key.start_loc,
            packed_label_elapsed_days_real + key.start_loc,
            packed_label_elapsed_days_int + key.start_loc
        );
        if constexpr (requires_grad) {
            r_grad_buffer.assign(key.L, 0.0f);
        }
        for (int i = key.l; i <= key.r; i++) {
            const float target = float(packed_rating[locs_ptr[i]] > 1);
            const int offset = locs_ptr[i] - key.start_loc - 1;
            loss += bce_loss(out_card_buffer[offset], target);
            if constexpr (store_out) {
                y_pred_buffer[y_buffer_offset] = out_card_buffer[offset];
                y_buffer[y_buffer_offset] = float(packed_rating[locs_ptr[i]] > 1);
                y_buffer_offset += 1;
            }
            if constexpr (requires_grad) {
                const int train_ord = train_ords_ptr[i];
                const float recency_weight = 0.25 + 0.75 * (float) pow((float) train_ord / train_size, 3);
                r_grad_buffer[offset] += recency_weight * bce_loss_grad(out_card_buffer[offset], target);
            }
        }

        if constexpr (requires_grad) {
            param_grad_buffer_2.assign((int)initial_params.size(), 0.0f);
            fsrs6_backward<float>(
                key.L,
                r_grad_buffer.data(),
                checkpoint_buffer.data(),
                params.data(),
                param_grad_buffer_2.data(),
                packed_rating + key.start_loc,
                packed_elapsed_days_real + key.start_loc,
                packed_elapsed_days_int + key.start_loc,
                packed_label_elapsed_days_real + key.start_loc,
                packed_label_elapsed_days_int + key.start_loc
            );
            for (int param_i = 0; param_i < (int)param_grad_buffer.size(); param_i++) {
                param_grad_buffer[param_i] += param_grad_buffer_2[param_i];
            }
        }
    }
    return loss;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fsrs_optimizer(
    const at::Tensor& pretrain_params,
    const at::Tensor& epochs,
    const at::Tensor& train_ords,
    const at::Tensor& locs,
    const at::Tensor& locs_lens,
    const at::Tensor& keys,
    const at::Tensor& keys_lens,
    const at::Tensor& train_set_locs,
    const at::Tensor& train_set_keys,
    const at::Tensor& test_set_locs,
    const at::Tensor& test_set_keys,
    const at::Tensor& packed_review_th_T,
    const at::Tensor& packed_rating_T, 
    const at::Tensor& packed_elapsed_days_real_T, 
    const at::Tensor& packed_elapsed_days_int_T, 
    const at::Tensor& packed_label_elapsed_days_real_T, 
    const at::Tensor& packed_label_elapsed_days_int_T
) {
    const int T = packed_review_th_T.size(0);
    const float* pretrain_params_ptr = pretrain_params.data_ptr<float>();
    const int* epochs_ptr = epochs.data_ptr<int>();
    const int* train_ords_ptr = train_ords.data_ptr<int>();
    const int* locs_ptr = locs.data_ptr<int>();
    const int* locs_lens_ptr = locs_lens.data_ptr<int>();
    const key_t* keys_ptr = (key_t*)keys.data_ptr();
    const int* keys_lens_ptr = keys_lens.data_ptr<int>();
    const int* train_set_locs_ptr = train_set_locs.data_ptr<int>();
    const key_t* train_set_keys_ptr = (key_t*)train_set_keys.data_ptr();
    const int* test_set_locs_ptr = test_set_locs.data_ptr<int>();
    const key_t* test_set_keys_ptr = (key_t*)test_set_keys.data_ptr();
    const int* packed_review_th_T_ptr = packed_review_th_T.data_ptr<int>();
    const int* packed_rating_T_ptr = packed_rating_T.data_ptr<int>();
    const float* packed_elapsed_days_real_T_ptr = packed_elapsed_days_real_T.data_ptr<float>();
    const float* packed_elapsed_days_int_T_ptr = packed_elapsed_days_int_T.data_ptr<float>();
    const float* packed_label_elapsed_days_real_T_ptr = packed_label_elapsed_days_real_T.data_ptr<float>();
    const float* packed_label_elapsed_days_int_T_ptr = packed_label_elapsed_days_int_T.data_ptr<float>();

    static std::vector<checkpoint_t<float>> checkpoint_buffer;
    static std::vector<float> y_pred_buffer, y_buffer, out_card_buffer, r_grad_buffer, param_grad_buffer, param_grad_buffer_2;

    // Ensure that buffer sizes are sufficient
    if ((int)checkpoint_buffer.size() < T) {
        checkpoint_t<float> empty_checkpoint = {};
        checkpoint_buffer.assign(T, empty_checkpoint);
        out_card_buffer.assign(T, 0.0f);
        y_pred_buffer.assign(T, 0.0f);
        y_buffer.assign(T, 0.0f);
        r_grad_buffer.assign(T, 0.0f);
    }

    std::vector<float> params = initial_params;
    for (int i = 0; i < 4; i++) {
        params[i] = pretrain_params_ptr[i];
    }

    adam optim(&params);

    const int train_size = train_set_locs.size(0);
    const int test_size = test_set_locs.size(0);
    const int total_steps = keys_lens.size(0);
    int locs_offset = 0;
    int keys_offset = 0;
    auto best_params = params;
    float best_loss = (float)1e9;
    for (int step = 0; step < total_steps; step++) {
        if (step == 0 || epochs_ptr[step] != epochs_ptr[step - 1]) {
            float eval_loss = run_batch<false, false>(param_grad_buffer, param_grad_buffer_2, out_card_buffer, y_buffer, y_pred_buffer, r_grad_buffer, checkpoint_buffer, params, train_set_keys.size(0), train_set_keys_ptr, -1, nullptr, train_set_locs_ptr, packed_rating_T_ptr, packed_elapsed_days_real_T_ptr, packed_elapsed_days_int_T_ptr, packed_label_elapsed_days_real_T_ptr, packed_label_elapsed_days_int_T_ptr);
            if (eval_loss < best_loss) {
                best_loss = eval_loss;
                best_params = params;
            }
        }

        const float lr = get_lr(step, total_steps);
        param_grad_buffer.assign((int)initial_params.size(), 0.0f); // zero_grad
        float loss = run_batch<true, false>(param_grad_buffer, param_grad_buffer_2, out_card_buffer, y_buffer, y_pred_buffer, r_grad_buffer, checkpoint_buffer, params, keys_lens_ptr[step], keys_ptr + keys_offset, train_size, train_ords_ptr + locs_offset, locs_ptr + locs_offset, packed_rating_T_ptr, packed_elapsed_days_real_T_ptr, packed_elapsed_days_int_T_ptr, packed_label_elapsed_days_real_T_ptr, packed_label_elapsed_days_int_T_ptr);
        for (int i = 0; i < (int)params.size(); i++) {
            param_grad_buffer[i] += 2.0f * (params[i] - initial_params[i]) / (default_params_stddev[i] * default_params_stddev[i]) * locs_lens_ptr[step] / train_size;
        }
        optim.step(param_grad_buffer, get_lr(step, total_steps));
        clip_params(params);

        locs_offset += locs_lens_ptr[step];
        keys_offset += keys_lens_ptr[step];
    }
    float eval_loss = run_batch<false, false>(param_grad_buffer, param_grad_buffer_2, out_card_buffer, y_buffer, y_pred_buffer, r_grad_buffer, checkpoint_buffer, params, train_set_keys.size(0), train_set_keys_ptr, -1, nullptr, train_set_locs_ptr, packed_rating_T_ptr, packed_elapsed_days_real_T_ptr, packed_elapsed_days_int_T_ptr, packed_label_elapsed_days_real_T_ptr, packed_label_elapsed_days_int_T_ptr);
    if (eval_loss < best_loss) {
        best_loss = eval_loss;
        best_params = params;
    }

    float test_loss = run_batch<false, true>(param_grad_buffer, param_grad_buffer_2, out_card_buffer, y_buffer, y_pred_buffer, r_grad_buffer, checkpoint_buffer, best_params, test_set_keys.size(0), test_set_keys_ptr, -1, nullptr, test_set_locs_ptr, packed_rating_T_ptr, packed_elapsed_days_real_T_ptr, packed_elapsed_days_int_T_ptr, packed_label_elapsed_days_real_T_ptr, packed_label_elapsed_days_int_T_ptr);
    at::Tensor test_loss_tensor = at::tensor(test_loss);
    at::Tensor test_loss_n_tensor = at::tensor(test_size);
    at::Tensor best_params_tensor = at::tensor(best_params);
    at::Tensor y_tensor = at::from_blob(y_buffer.data(), {test_size}, torch::kFloat).clone();
    at::Tensor y_pred_tensor = at::from_blob(y_pred_buffer.data(), {test_size}, torch::kFloat).clone();
    return {test_loss_tensor, test_loss_n_tensor, best_params_tensor, y_tensor, y_pred_tensor};
}

at::Tensor compute_rmse_bins(
    at::Tensor y,
    at::Tensor y_pred,
    at::Tensor rmse_bin_ind
) {
    const int L = y.size(0);
    const float* y_ptr = y.data_ptr<float>();
    const float* y_pred_ptr = y_pred.data_ptr<float>();
    const int* rmse_bin_ind_ptr = rmse_bin_ind.data_ptr<int>();
    static std::vector<float> rmse_bin_tot_buffer;
    static std::vector<int> rmse_bin_n_buffer;
    if ((int)rmse_bin_tot_buffer.size() < L) {
        rmse_bin_tot_buffer.assign(L, 0.0f);
        rmse_bin_n_buffer.assign(L, 0);
    }
    // Compute RMSE (bins)
    for (int i = 0; i < L; i++) {
        rmse_bin_tot_buffer[rmse_bin_ind_ptr[i]] += y_ptr[i] - y_pred_ptr[i];
        rmse_bin_n_buffer[rmse_bin_ind_ptr[i]] += 1;
    }
    float rmse_bins_tot = 0;
    for (int i = 0; i < L; i++) {
        int &cnt = rmse_bin_n_buffer[rmse_bin_ind_ptr[i]];
        if (cnt > 0) {
            float &diff = rmse_bin_tot_buffer[rmse_bin_ind_ptr[i]];
            rmse_bins_tot += diff * diff / cnt;
            cnt = 0;
            diff = 0;
        }
    }
    float rmse_bins = sqrt(rmse_bins_tot / L);
    return at::tensor(rmse_bins);
}

namespace fsrs {
    TORCH_LIBRARY_IMPL(fsrs, CPU, m) {
        m.impl("fsrs_optimizer", &fsrs_optimizer);
        m.impl("compute_rmse_bins", &compute_rmse_bins);
    }
}