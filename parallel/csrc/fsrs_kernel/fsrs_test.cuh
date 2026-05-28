#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#include "../fsrs/fsrs7.cuh"

extern "C" void fsrs_test_cuda(
    const float* elapsed_days_real_flat,
    const int8_t* rating_flat,
    const int32_t* start_index,
    const int32_t* seq_len,
    const fsrs_params_t* fsrs_params,
    float* p,
    int32_t num_sequences,
    cudaStream_t stream
);
