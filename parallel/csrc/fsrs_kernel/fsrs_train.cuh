#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#include "../fsrs/fsrs7.cuh"

extern "C" void fsrs_train_cuda(
    const float* elapsed_days_real_flat,
    const int8_t* rating_flat,
    const int32_t* start_index,
    const float* grad_weight,
    const int32_t* seq_len_UxT,
    const int32_t* seq_len_Ux_max,
    const int32_t* seq_len_Ux_max_cumsum,
    const fsrs_params_t* fsrs_params,
    int32_t U,
    int32_t x,
    int32_t threads_per_block,
    cudaStream_t stream,
    fsrs_state_t* state_buffer,
    fsrs_params_t* grad
);
