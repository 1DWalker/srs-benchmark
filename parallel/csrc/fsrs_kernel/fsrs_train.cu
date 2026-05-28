#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include "../fsrs/fsrs7.cu"
#include "fsrs_train.cuh"

int __device__ enzyme_dup;
int __device__ enzyme_dupnoneed;
int __device__ enzyme_out;
int __device__ enzyme_const;

__device__ __forceinline__
void fsrs7_forgetting_curve_wrapper(
    const fsrs_params_t &fsrs_params,
    const float elapsed_time,
    const fsrs_state_t &state,
    float *p
) {
    *p = fsrs7_forgetting_curve(fsrs_params, elapsed_time, state);
}


__device__
void __enzyme_autodiff_forgetting_curve(
    void*,
    int, const fsrs_params_t, fsrs_params_t*,
    int, const float,
    int, const fsrs_state_t, fsrs_state_t*,
    int, float*, float*
);

__device__ __forceinline__
void fsrs7_step_wrapper(
    const fsrs_params_t &fsrs_params,
    const fsrs_state_t fsrs_state,
    const float elapsed_time,
    const int8_t rating,
    fsrs_state_t &out_state
) {
    out_state = fsrs7_step(fsrs_params, fsrs_state, elapsed_time, rating);
}

__device__
void __enzyme_autodiff_fsrs7_step(
    void*,
    int, const fsrs_params_t, fsrs_params_t*,
    int, const fsrs_state_t, fsrs_state_t*,
    int, const float,
    int, const int8_t,
    int, fsrs_state_t, fsrs_state_t*
);

__device__ __forceinline__
void fsrs7_init_wrapper(
    const fsrs_params_t &fsrs_params,
    const int8_t first_rating,
    fsrs_state_t &out_state
) {
    out_state = fsrs7_init(fsrs_params, first_rating);
}

__device__
void __enzyme_autodiff_fsrs7_init(
    void*,
    int, const fsrs_params_t, fsrs_params_t*,
    int, const int8_t,
    int, fsrs_state_t, fsrs_state_t*
);

__device__ __forceinline__ 
int32_t idx2(
    int32_t i,
    int32_t j,
    int32_t J
) {
    return i * J + j;
}

__device__ __forceinline__ 
int32_t idx3(
    int32_t i,
    int32_t j,
    int32_t k,
    int32_t J,
    int32_t K
) {
    return (i * J + j) * K + k;
}

// __device__ __forceinline__
// float loss(
//     float p,
//     float y
// ) {
//     constexpr float eps = 1e-7f;
//     p = fminf(fmaxf(p, eps), 1.0f - eps);
//     return -(y * logf(p) + (1.0f - y) * logf(1.0f - p));
// }

__device__ __forceinline__
float dloss_dp(
    float p,
    const bool label
) {
    constexpr float eps = 1e-5f;
    p = fminf(fmaxf(p, eps), 1.0f - eps);

    return label ? (-1.0f / p) : (1.0f / (1.0f - p));
}

__global__ void fsrs_train_kernel(
    const float* __restrict__ elapsed_days_real_flat,
    const int8_t* __restrict__ rating_flat,
    const int32_t* __restrict__ start_index,
    const float* __restrict__ grad_weight,
    const int32_t* __restrict__ seq_len_p,
    const int32_t* __restrict__ seq_len_max,
    const int32_t* __restrict__ seq_len_max_cumsum,
    const fsrs_params_t* __restrict__ fsrs_params_p,
    fsrs_state_t* __restrict__ state_buffer,
    fsrs_params_t* __restrict__ grad
) {
    // blockIdx.x is up to U
    // blockIdx.y is up to x
    // gridDim.x == U
    // gridDim.y == x
    // threadIdx.x is contiguous in a warp
    // threadIdx.y is different warps
    // blockDim.x == 32
    // use thread_i for thread indexing

    __shared__ fsrs_params_t params;
    __shared__ int32_t _state_buffer_step;
    __shared__ int32_t _state_buffer_offset;
    const int32_t thread_i_within_block = threadIdx.y * blockDim.x + threadIdx.x;
    const int32_t threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    const int32_t i = idx3(blockIdx.x, blockIdx.y, thread_i_within_block, gridDim.y, threads_per_block);

    if (threadIdx.y == 0 && threadIdx.x == 0) {
        params = fsrs_params_p[blockIdx.x];
    }
    if (threadIdx.y == 1) {
        if (threadIdx.x == 0) {
            _state_buffer_offset = seq_len_max_cumsum[idx2(blockIdx.x, blockIdx.y, gridDim.y)];
        }
        if (threadIdx.x == 1) {
            _state_buffer_step = seq_len_max[idx2(blockIdx.x, blockIdx.y, gridDim.y)];
        }
    }
    __syncthreads();

    const int32_t len = seq_len_p[i];
    // Moved to registers for speedup(?)
    const int32_t state_buffer_offset = _state_buffer_offset;
    const int32_t state_buffer_step = _state_buffer_step;
    
    const int32_t start = start_index[i];
    fsrs_state_t state = fsrs7_init(params, rating_flat[start]);

    auto get_state_buffer_index = [&](int32_t len) {
        return state_buffer_offset + len * threads_per_block + thread_i_within_block;
    };

    for (int32_t l = 1; l < len - 1; ++l) {
        const int32_t review_index = start + l;
        state_buffer[get_state_buffer_index(l)] = state;
        state = fsrs7_step(
            params,
            state,
            elapsed_days_real_flat[review_index],
            rating_flat[review_index]
        );
    }

    const int32_t target_index = start + len - 1;
    float p = fsrs7_forgetting_curve(
        params,
        elapsed_days_real_flat[target_index],
        state
    );
    const float label = (float) (rating_flat[target_index] > 1);
    float grad_p = grad_weight[i] * dloss_dp(p, label);

    fsrs_state_t grad_state{};
    fsrs_params_t grad_params{};
    float _blank_p;
    __enzyme_autodiff_forgetting_curve(
        (void*) fsrs7_forgetting_curve_wrapper, 
        enzyme_dup, params, &grad_params,
        enzyme_const, elapsed_days_real_flat[target_index],
        enzyme_dup, state, &grad_state,
        enzyme_dupnoneed, &_blank_p, &grad_p
    );

    fsrs_state_t _blank_state{};
    for (int32_t l = len - 2; l >= 1; l--) {
        const int32_t review_index = start + l;
        fsrs_state_t new_grad_state{};
        state = state_buffer[get_state_buffer_index(l)];
        __enzyme_autodiff_fsrs7_step(
            (void*) fsrs7_step_wrapper, 
            enzyme_dup, params, &grad_params,
            enzyme_dup, state, &new_grad_state,
            enzyme_const, elapsed_days_real_flat[review_index],
            enzyme_const, rating_flat[review_index],
            enzyme_dupnoneed, _blank_state, &grad_state
        );
        grad_state = new_grad_state;
    }

    __enzyme_autodiff_fsrs7_init(
        (void*) fsrs7_init_wrapper,
        enzyme_dup, params, &grad_params,
        enzyme_const, rating_flat[start],
        enzyme_dupnoneed, _blank_state, &grad_state
    );

    grad[i] = grad_params;
}

void fsrs_train_cuda(
    const float* __restrict__ elapsed_days_real_flat,
    const int8_t* __restrict__ rating_flat,
    const int32_t* __restrict__ start_index,
    const float* __restrict__ grad_weight,
    const int32_t* __restrict__ seq_len_UxT,
    const int32_t* __restrict__ seq_len_Ux_max,
    const int32_t* __restrict__ seq_len_Ux_max_cumsum,
    const fsrs_params_t* __restrict__ fsrs_params,
    const int32_t U,
    const int32_t x,
    const int32_t THREADS_PER_BLOCK,
    cudaStream_t stream,
    fsrs_state_t* __restrict__ state_buffer,
    fsrs_params_t* __restrict__ grad
) {
    dim3 block(32, THREADS_PER_BLOCK / 32);
    dim3 grid(U, x);
    fsrs_train_kernel<<<grid, block, 0, stream>>>(
        elapsed_days_real_flat,
        rating_flat,
        start_index,
        grad_weight,
        seq_len_UxT,
        seq_len_Ux_max,
        seq_len_Ux_max_cumsum,
        fsrs_params,
        state_buffer,
        grad
    );
}
