#include <cuda_runtime.h>
#include <stdint.h>

#include "../fsrs/fsrs7.cu"
#include "fsrs_test.cuh"

__global__ void fsrs_test_kernel(
    const float* __restrict__ elapsed_days_real_flat,
    const int8_t* __restrict__ rating_flat,
    const int32_t* __restrict__ start_index,
    const int32_t* __restrict__ seq_len,
    const fsrs_params_t* __restrict__ fsrs_params,
    const int32_t N,
    float* __restrict__ p
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int32_t start = start_index[i];
    const int32_t len = seq_len[i];
    const fsrs_params_t params = fsrs_params[i];

    fsrs_state_t state = fsrs7_init(params, rating_flat[start]);
    for (int32_t l = 1; l < len - 1; ++l) {
        const int32_t review_index = start + l;
        state = fsrs7_step(
            params,
            state,
            elapsed_days_real_flat[review_index],
            rating_flat[review_index]
        );
    }

    const int32_t target_index = start + len - 1;
    p[i] = fsrs7_forgetting_curve(
        params,
        elapsed_days_real_flat[target_index],
        state
    );
}

void fsrs_test_cuda(
    const float* __restrict__ elapsed_days_real_flat,
    const int8_t* __restrict__ rating_flat,
    const int32_t* __restrict__ start_index,
    const int32_t* __restrict__ seq_len,
    const fsrs_params_t* __restrict__ fsrs_params,
    float* __restrict__ p,
    const int32_t N,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((N + threads - 1) / threads);
    fsrs_test_kernel<<<blocks, threads, 0, stream>>>(
        elapsed_days_real_flat,
        rating_flat,
        start_index,
        seq_len,
        fsrs_params,
        N,
        p
    );
}
