#pragma once

#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <math.h>
#include "fsrs.h"
#include "fsrs_forward.cpp"

std::tuple<at::Tensor, at::Tensor> fsrs_batch_forward(
    const at::Tensor& params_P,
    const at::Tensor& review_th_B,
    const at::Tensor& packed_review_th_T,
    const at::Tensor& packed_rating_T,
    const at::Tensor& packed_elapsed_days_real_T,
    const at::Tensor& packed_elapsed_days_int_T,
    const at::Tensor& packed_label_elapsed_days_real_T,
    const at::Tensor& packed_label_elapsed_days_int_T,
    const at::Tensor& perm_T_tensor,
    const at::Tensor& perm_inv_T_tensor,
    const at::Tensor& card_locs_T
) {
    const int B = review_th_B.size(0);
    const int T = packed_review_th_T.size(0);
    const float* params_P_ptr = params_P.data_ptr<float>();
    const int* review_th_B_ptr = review_th_B.data_ptr<int>();
    const int* packed_review_th_T_ptr = packed_review_th_T.data_ptr<int>();
    const int* packed_rating_T_ptr = packed_rating_T.data_ptr<int>();
    const float* packed_elapsed_days_real_T_ptr = packed_elapsed_days_real_T.data_ptr<float>();
    const float* packed_elapsed_days_int_T_ptr = packed_elapsed_days_int_T.data_ptr<float>();
    const float* packed_label_elapsed_days_real_T_ptr = packed_label_elapsed_days_real_T.data_ptr<float>();
    const float* packed_label_elapsed_days_int_T_ptr = packed_label_elapsed_days_int_T.data_ptr<float>();
    const int* perm_T_ptr = perm_T_tensor.data_ptr<int>();
    const int* perm_inv_T_ptr = perm_inv_T_tensor.data_ptr<int>();
    const int* card_locs_T_ptr = card_locs_T.data_ptr<int>();
    
    // std::vector<std::tuple<int, int, int>> keys(B);
    at::Tensor keys = torch::empty({B, 3}, torch::TensorOptions().dtype(torch::kInt32).requires_grad(false));
    std::tuple<int, int, int>* keys_ptr = (std::tuple<int, int, int>*)keys.data_ptr();

    for (int i = 0; i < B; i++) {
        int review_th = review_th_B_ptr[i];
        keys_ptr[i] = {card_locs_T_ptr[review_th - 1], perm_inv_T_ptr[review_th - 1], i};
    }
    // Sort by the first element so that reviews with the same card_id are grouped together
    std::sort(keys_ptr, keys_ptr + B, [&](const auto &a, const auto &b) { return std::get<0>(a) < std::get<0>(b); });

    // std::vector<float> out(B);
    // std::vector<fsrs_state<float>> checkpoint(B);
    int total_review_history = 0;
    int longest_review_history = 0;
    for (int l = 0, r = 1; l < B; l = r++) {
        int start_loc = std::get<0>(keys_ptr[l]);
        while (r < B && start_loc == std::get<0>(keys_ptr[r])) r++;
        int L = 0;
        for (int i = l; i < r; i++) {
            L = std::max(L, std::get<1>(keys_ptr[i]) - start_loc);
        }
        longest_review_history = std::max(longest_review_history, L);
        total_review_history += L;
    }
    std::cout << "Longest: " << longest_review_history << '\n';
    std::cout << "Total: " << total_review_history << '\n';
    std::vector<float> out_card_buffer(longest_review_history);

    at::Tensor out_B = torch::empty(review_th_B.sizes(), params_P.options());
    float* out_B_ptr = out_B.data_ptr<float>();
    size_t num_chunks = (sizeof(fsrs_state<float>) + 4 - 1) / 4;
    at::Tensor checkpoints = torch::empty({total_review_history, (int)num_chunks}, params_P.options().requires_grad(false));
    fsrs_state<float>* checkpoints_ptr = (fsrs_state<float>*)checkpoints.data_ptr();
    int checkpoint_offset = 0;
    for (int l = 0, r = 1; l < B; l = r++) {
        int start_loc = std::get<0>(keys_ptr[l]);
        while (r < B && start_loc == std::get<0>(keys_ptr[r])) r++;
        // [l, r) is the range of elements in `keys`, each corresponding to some queried review_th such that the card_ids are the same
        int L = 0;
        for (int i = l; i < r; i++) {
            L = std::max(L, std::get<1>(keys_ptr[i]) - start_loc);
        }
        fsrs6_forward<float>(
            L,
            params_P_ptr,
            out_card_buffer.data(),
            checkpoints_ptr + checkpoint_offset,
            packed_rating_T_ptr + start_loc,
            packed_elapsed_days_real_T_ptr + start_loc,
            packed_elapsed_days_int_T_ptr + start_loc,
            packed_label_elapsed_days_real_T_ptr + start_loc,
            packed_label_elapsed_days_int_T_ptr + start_loc
        );
        for (int i = l; i < r; i++) {
            out_B_ptr[std::get<2>(keys_ptr[i])] = out_card_buffer[std::get<1>(keys_ptr[i]) - 1 - start_loc];
        }
        checkpoint_offset += L;
    }
    return {out_B, checkpoints};
}

namespace fsrs {
    TORCH_LIBRARY_IMPL(fsrs, CPU, m) {
        m.impl("fsrs_batch_forward", &fsrs_batch_forward);
    }
}