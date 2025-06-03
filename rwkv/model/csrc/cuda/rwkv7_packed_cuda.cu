#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include "rwkv7_cuda_utils.h"

using namespace nvcuda;

template <int CHUNK_LEN=32>
std::tuple<at::Tensor, int64_t> get_checkpoint_indices(const at::Tensor& indices_I, int T) {
    at::Tensor indices_appended = torch::cat({indices_I, torch::tensor({T}, indices_I.options().requires_grad(false))}, 0);
    at::Tensor indices_space_I = torch::floor_divide(indices_appended.slice(0, 1) - indices_appended.slice(0, 0, indices_I.size(0)), CHUNK_LEN);
    int64_t total_checkpoints = indices_space_I.sum().item<int64_t>();
    at::Tensor prefixsum = torch::cumsum(indices_space_I, 0);
    torch::Tensor zero = torch::zeros({1}, indices_I.options().requires_grad(false));
    at::Tensor prepended = torch::cat({zero, prefixsum}, 0);
    at::Tensor shifted = prepended.slice(0, 0, indices_I.size(0));
    return std::make_tuple<>(shifted, total_checkpoints);
}

namespace rwkv {
template <int CHUNK_LEN=32, typename F>
__global__ void rwkv7_packed_wkv_forward_kernel(
    const int I,
    const int T,
    const int H,
    const int64_t* indices_I,
    const F* __restrict__ r_THK,
    const F* __restrict__ k_THK,
    const F* __restrict__ v_THK,
    const float* __restrict__ w_THK,
    const F* __restrict__ a_THK,
    const F* __restrict__ k_deformed_THK,
    F* __restrict__ out_THK,
    const int L,
    float* __restrict__ state_checkpoints_LHKK,
    const int64_t* __restrict__ checkpoints_indices_I
    ) {
    const int K = 32;
    const int i = blockIdx.x;
    const int h = blockIdx.y;

    const int x = threadIdx.y;
    const int y = threadIdx.x;

    int64_t checkpoint_index = checkpoints_indices_I[i];
    const int start_index = indices_I[i];
    const int end_index_ex = i == I - 1 ? T : indices_I[i + 1];
    const int tot_t = end_index_ex - start_index;
    float state_xy = 0.0;
    for (int it = 0; it < tot_t; it++) {
        if (it > 0 && it % CHUNK_LEN == 0) {
            state_checkpoints_LHKK[get_index3(checkpoint_index, h, x, y, H, K, K)] = state_xy;
            checkpoint_index++;
        }
        int64_t t = start_index + it;
        int64_t global_x = get_index2(t, h, x, H, K);
        int64_t global_y = get_index2(t, h, y, H, K);
        float r_y = to_float<F>(r_THK[global_y]);
        float k_y = to_float<F>(k_THK[global_y]);
        float v_x = to_float<F>(v_THK[global_x]);
        float w_y = w_THK[global_y];
        float a_y = to_float<F>(a_THK[global_y]);
        float k_deformed_y = to_float<F>(k_deformed_THK[global_y]);
        float state_xy_decayed = state_xy * w_y;
        float state_k_dot = state_xy * k_deformed_y;
        for (int offset = 16; offset > 0; offset /= 2) {
            state_k_dot += __shfl_down_sync(FULL_MASK, state_k_dot, offset);
        }
        state_k_dot = __shfl_sync(FULL_MASK, state_k_dot, 0);
        state_xy = state_xy_decayed - state_k_dot * a_y * k_deformed_y;
        state_xy += v_x * k_y;

        // Compute S@r and store the result in out
        float state_r_dot = state_xy * r_y;
        for (int offset = 16; offset > 0; offset /= 2) {
            state_r_dot += __shfl_down_sync(FULL_MASK, state_r_dot, offset);
        }
        if (y == 0) {
            out_THK[global_x] = to_F<F>(state_r_dot);
        }
    }
}

template <int CHUNK_LEN=32, typename F>
__global__ void rwkv7_packed_wkv_backward_kernel(
    const int I,
    const int T,
    const int H,
    const int64_t* indices_I,
    const F* __restrict__ r_THK,
    const F* __restrict__ k_THK,
    const F* __restrict__ v_THK,
    const float* __restrict__ w_THK,
    const F* __restrict__ a_THK,
    const F* __restrict__ k_deformed_THK,
    const F* __restrict__ grad_THK,
    const int L,
    const float* __restrict__ state_checkpoints_LHKK,
    const int64_t* __restrict__ checkpoints_indices_I,
    F* __restrict__ r_grad_THK,
    F* __restrict__ k_grad_THK,
    F* __restrict__ v_grad_THK,
    float* __restrict__ w_grad_THK,
    F* __restrict__ a_grad_THK,
    F* __restrict__ k_deformed_grad_THK
    ) {
    const int K = 32;
    __shared__ float KK_state[32 * (32 + 1)];
    __shared__ float KK_state_prev[32 * (32 + 1)];
    __shared__ float KK_grad_decay_remove[32 * 32];
    __shared__ float KK_dS[32 * (32 + 1)];
    __shared__ float KK_grad_decay[32 * (32 + 1)];
    __shared__ float K_k_deformed[32];
    __shared__ float K_a[32];
    float state_xy_chunk[CHUNK_LEN]; 
    float state_prev_xy_chunk[CHUNK_LEN];
    const int i = blockIdx.x;
    const int h = blockIdx.y;
    const int x = threadIdx.y;
    const int y = threadIdx.x;

    const int64_t checkpoint_index_offset = checkpoints_indices_I[i];
    const int start_index = indices_I[i];
    const int end_index_ex = i == I - 1 ? T : indices_I[i + 1];
    const int tot_t = end_index_ex - start_index;
    const int num_chunks = (tot_t - 1 + CHUNK_LEN) / CHUNK_LEN;

    // TODO does this do anything given the initialization?
    if (x == 0) {
        a_grad_THK[get_index2(start_index, h, y, H, K)] = to_F<F>(0.0);
        k_deformed_grad_THK[get_index2(start_index, h, y, H, K)] = to_F<F>(0.0);
    }

    float dS_xy_contrib = 0.0;
    for (int chunk_i = num_chunks - 1; chunk_i >= 0; chunk_i--) {
        // recompute the states from the checkpoints
        float state_xy = chunk_i == 0 ? 0.0 : state_checkpoints_LHKK[get_index3(checkpoint_index_offset + chunk_i - 1, h, x, y, H, K, K)];
        for (int c = 0; c < CHUNK_LEN; c++) {
            int it = chunk_i * CHUNK_LEN + c;
            if (it >= tot_t) break;

            state_prev_xy_chunk[c] = state_xy;
            int64_t t = start_index + it;
            int64_t global_x = get_index2(t, h, x, H, K);
            int64_t global_y = get_index2(t, h, y, H, K);
            float r_y = to_float<F>(r_THK[global_y]);
            float k_y = to_float<F>(k_THK[global_y]);
            float v_x = to_float<F>(v_THK[global_x]);
            float w_y = w_THK[global_y];
            float a_y = to_float<F>(a_THK[global_y]);
            float k_deformed_y = to_float<F>(k_deformed_THK[global_y]);
            float state_xy_decayed = state_xy * w_y;
            float state_k_dot = state_xy * k_deformed_y;
            for (int offset = 16; offset > 0; offset /= 2) {
                state_k_dot += __shfl_down_sync(FULL_MASK, state_k_dot, offset);
            }
            state_k_dot = __shfl_sync(FULL_MASK, state_k_dot, 0);
            state_xy = state_xy_decayed - state_k_dot * a_y * k_deformed_y;
            state_xy += v_x * k_y;
            state_xy_chunk[c] = state_xy;
        }
        
        for (int t = std::min(end_index_ex - 1, start_index + (chunk_i + 1) * CHUNK_LEN - 1); t >= start_index + chunk_i * CHUNK_LEN; t--) {
            int c = t - start_index - chunk_i * CHUNK_LEN;
            float state_xy = state_xy_chunk[c];
            KK_state[get_index1(x, y, K+1)] = state_xy;
            KK_state_prev[get_index1(x, y, K+1)] = state_prev_xy_chunk[c];
            int64_t global_x = get_index2(t, h, x, H, K);
            int64_t global_y = get_index2(t, h, y, H, K);
            float r_y = to_float<F>(r_THK[global_y]);
            float k_y = to_float<F>(k_THK[global_y]);
            float v_y = to_float<F>(v_THK[global_y]);
            float w_y = w_THK[global_y];
            float a_y = to_float<F>(a_THK[global_y]);
            float k_deformed_x = to_float<F>(k_deformed_THK[global_x]);
            float k_deformed_y = to_float<F>(k_deformed_THK[global_y]);
            float grad_x = to_float<F>(grad_THK[global_x]);
            float grad_y = to_float<F>(grad_THK[global_y]);
            float dS_xy = grad_x * r_y + dS_xy_contrib;
            dS_xy_contrib = 0.0;
            float dS_xy_decay = dS_xy * w_y;
            float dS_xy_remove = dS_xy * a_y * k_deformed_y;
            KK_dS[get_index1(x, y, K + 1)] = dS_xy;
            if (x == 0) {
                K_k_deformed[y] = k_deformed_y;
                K_a[y] = a_y;
            }

            __syncthreads(); // for KK_state, KK_dS

            float grad_decay_remove_xy = 0.0;
            for (int k = 0; k < K; k++) {
                grad_decay_remove_xy += KK_state_prev[get_index1(k, x, K+1)] * KK_dS[get_index1(k, y, K+1)];
            }
            if (x == y) {
                w_grad_THK[get_index2(t, h, x, H, K)] = grad_decay_remove_xy;
            }
            KK_grad_decay[get_index1(x, y, K+1)] = grad_decay_remove_xy;

            float state_mT_xy = KK_state[get_index1(y, x, K + 1)];
            float state_grad_dot = state_mT_xy * grad_y;
            float v_grad_x = dS_xy * k_y;
            float k_grad_x = KK_dS[get_index1(y, x, K + 1)] * v_y;

            for (int offset = 16; offset > 0; offset /= 2) {
                v_grad_x += __shfl_down_sync(FULL_MASK, v_grad_x, offset);
                k_grad_x += __shfl_down_sync(FULL_MASK, k_grad_x, offset);
                state_grad_dot += __shfl_down_sync(FULL_MASK, state_grad_dot, offset);
                dS_xy_remove += __shfl_down_sync(FULL_MASK, dS_xy_remove, offset);
            }
            if (y == 0) {
                v_grad_THK[get_index2(t, h, x, H, K)] = to_F<F>(v_grad_x);
                k_grad_THK[get_index2(t, h, x, H, K)] = to_F<F>(k_grad_x);
                r_grad_THK[get_index2(t, h, x, H, K)] = to_F<F>(state_grad_dot);
            }
            __syncthreads(); // for KK_grad_decay

            float KK_grad_decay_yx = KK_grad_decay[get_index1(y, x, K+1)];
            float a_grad_x = -KK_grad_decay_yx * K_k_deformed[y];
            float k_deformed_t1 = -grad_decay_remove_xy * K_a[y] * K_k_deformed[y];
            float k_deformed_t2 = -K_a[x] * KK_grad_decay_yx * K_k_deformed[y];
            // TODO potential tensor core optimization
            for (int offset = 16; offset > 0; offset /= 2) {
                a_grad_x += __shfl_down_sync(FULL_MASK, a_grad_x, offset);
                k_deformed_t1 += __shfl_down_sync(FULL_MASK, k_deformed_t1, offset);
                k_deformed_t2 += __shfl_down_sync(FULL_MASK, k_deformed_t2, offset);
            }
            
            if (y == 0) {
                a_grad_THK[get_index2(t, h, x, H, K)] = to_F<F>(a_grad_x * K_k_deformed[x]);
                k_deformed_grad_THK[get_index2(t, h, x, H, K)] = to_F<F>(k_deformed_t1 + k_deformed_t2);
            }

            dS_xy_remove = __shfl_sync(FULL_MASK, dS_xy_remove, 0);
            dS_xy_contrib += dS_xy_decay - dS_xy_remove * k_deformed_y;
            __syncthreads();
        }
    }
}

template <int CHUNK_LEN=32, typename F>
std::tuple<at::Tensor, at::Tensor> rwkv7_packed_wkv_forward_cuda(
    const at::Tensor& indices_I,
    const at::Tensor& r_THK, 
    const at::Tensor& k_THK,
    const at::Tensor& v_THK,
    const at::Tensor& w_THK,
    const at::Tensor& a_THK,
    const at::Tensor& k_deformed_THK
    ) {
    const int I = indices_I.size(0);
    TORCH_INTERNAL_ASSERT(indices_I.dtype() == torch::kLong);
    const int T = r_THK.size(0);
    const int H = r_THK.size(1);
    const int K = r_THK.size(2);
    TORCH_INTERNAL_ASSERT(r_THK.device().type() == at::DeviceType::CUDA);
    const int64_t* indices_ptr = (int64_t*)indices_I.data_ptr();
    const F* r_ptr = (F*)r_THK.data_ptr();
    const F* k_ptr = (F*)k_THK.data_ptr();
    const F* v_ptr = (F*)v_THK.data_ptr();
    const float* w_ptr = w_THK.data_ptr<float>();
    const F* a_ptr = (F*)a_THK.data_ptr();
    const F* k_deformed_ptr = (F*)k_deformed_THK.data_ptr();
    
    at::Tensor out_THK = torch::empty(r_THK.sizes(), r_THK.options());
    F* out_ptr = (F*)out_THK.data_ptr();
    auto [checkpoint_indices_I, total_checkpoints] = get_checkpoint_indices<CHUNK_LEN>(indices_I, T);
    const int64_t* checkpoint_indices_ptr = (int64_t*)checkpoint_indices_I.data_ptr();
    at::Tensor state_checkpoints_LHKK = torch::empty({total_checkpoints, H, K, K}, r_THK.options().dtype(torch::kFloat32)).requires_grad_(false);
    float* state_checkpoints_ptr = state_checkpoints_LHKK.data_ptr<float>();

    dim3 block_dim(32, 32);
    dim3 grid_dim(I, H);
    rwkv7_packed_wkv_forward_kernel<CHUNK_LEN><<<grid_dim, block_dim>>>(I, T, H, indices_ptr, r_ptr, k_ptr, v_ptr, w_ptr, a_ptr, k_deformed_ptr, out_ptr, total_checkpoints, state_checkpoints_ptr, checkpoint_indices_ptr);
    return std::make_tuple(out_THK, state_checkpoints_LHKK);
}

template <int CHUNK_LEN=32, typename F>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> rwkv7_packed_wkv_backward_cuda(
    const at::Tensor& indices_I,
    const at::Tensor& r_THK, 
    const at::Tensor& k_THK,
    const at::Tensor& v_THK,
    const at::Tensor& w_THK,
    const at::Tensor& a_THK,
    const at::Tensor& k_deformed_THK,
    const at::Tensor& state_checkpoints_LHKK,
    const at::Tensor& grad_THK
    ) {
    const int I = indices_I.size(0);
    const int T = r_THK.size(0);
    const int H = r_THK.size(1);
    const int K = r_THK.size(2);
    const int L = state_checkpoints_LHKK.size(1);
    TORCH_INTERNAL_ASSERT(r_THK.device().type() == at::DeviceType::CUDA);
    const int64_t* indices_ptr = (int64_t*)indices_I.data_ptr();
    const F* r_ptr = (F*)r_THK.data_ptr();
    const F* k_ptr = (F*)k_THK.data_ptr();
    const F* v_ptr = (F*)v_THK.data_ptr();
    const float* w_ptr = w_THK.data_ptr<float>();
    const F* a_ptr = (F*)a_THK.data_ptr();
    const F* k_deformed_ptr = (F*)k_deformed_THK.data_ptr();
    const float* state_checkpoints_ptr = state_checkpoints_LHKK.data_ptr<float>();
    const F* grad_ptr = (F*)grad_THK.data_ptr();
    at::Tensor r_grad_THK = torch::zeros_like(r_THK);
    at::Tensor k_grad_THK = torch::zeros_like(r_THK);
    at::Tensor v_grad_THK = torch::zeros_like(r_THK);
    at::Tensor w_grad_THK = torch::zeros_like(r_THK, torch::dtype(torch::kFloat32));
    at::Tensor a_grad_THK = torch::zeros_like(r_THK);
    at::Tensor k_deformed_grad_THK = torch::zeros_like(r_THK);
    F* r_grad_ptr = (F*)r_grad_THK.data_ptr();
    F* k_grad_ptr = (F*)k_grad_THK.data_ptr();
    F* v_grad_ptr = (F*)v_grad_THK.data_ptr();
    float* w_grad_ptr = w_grad_THK.data_ptr<float>();
    F* a_grad_ptr = (F*)a_grad_THK.data_ptr();
    F* k_deformed_grad_ptr = (F*)k_deformed_grad_THK.data_ptr();
    auto [checkpoint_indices_I, total_checkpoints] = get_checkpoint_indices<CHUNK_LEN>(indices_I, T);
    const int64_t* checkpoint_indices_ptr = (int64_t*)checkpoint_indices_I.data_ptr();

    dim3 block_dim(32, 32);
    dim3 grid_dim(I, H);
    rwkv7_packed_wkv_backward_kernel<CHUNK_LEN><<<grid_dim, block_dim>>>(I, T, H, indices_ptr, r_ptr, k_ptr, v_ptr, w_ptr, a_ptr, k_deformed_ptr, 
        grad_ptr, total_checkpoints, state_checkpoints_ptr, checkpoint_indices_ptr, r_grad_ptr, k_grad_ptr, v_grad_ptr, w_grad_ptr, a_grad_ptr, k_deformed_grad_ptr);
    return std::make_tuple(r_grad_THK, k_grad_THK, v_grad_THK, w_grad_THK, a_grad_THK, k_deformed_grad_THK);
}

const int CHECKPOINT_LEN = 32;
TORCH_LIBRARY_IMPL(rwkv, CUDA, m) {
    m.impl("rwkv7_packed_wkv_forward_float", &rwkv7_packed_wkv_forward_cuda<CHECKPOINT_LEN, float>);
    m.impl("rwkv7_packed_wkv_backward_float", &rwkv7_packed_wkv_backward_cuda<CHECKPOINT_LEN, float>);
    m.impl("rwkv7_packed_wkv_forward_bfloat16", &rwkv7_packed_wkv_forward_cuda<CHECKPOINT_LEN, __nv_bfloat16>);
    m.impl("rwkv7_packed_wkv_backward_bfloat16", &rwkv7_packed_wkv_backward_cuda<CHECKPOINT_LEN, __nv_bfloat16>);
    m.impl("rwkv7_packed_wkv_forward_half", &rwkv7_packed_wkv_forward_cuda<CHECKPOINT_LEN, __half>);
    m.impl("rwkv7_packed_wkv_backward_half", &rwkv7_packed_wkv_backward_cuda<CHECKPOINT_LEN, __half>);
}
}