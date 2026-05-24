#include <cuda_runtime.h>
#include <stdint.h>

#include "fsrs/fsrs7.cu"


__device__ void square_impl(double* x, double* y) {
    y[0] = x[0] * x[0] + 0.0;
}

typedef void (*square_fn)(double*, double*);

extern void __device__ __enzyme_autodiff(
    square_fn,
    int, double*, double*,
    int, double*, double*
);

int __device__ enzyme_dup;

__global__ void square_forward_kernel(const double* x, double* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    double x_local = x[i];
    double y_local = 0.0;
    square_impl(&x_local, &y_local);
    y[i] = y_local;
}

__global__ void square_backward_kernel(
    const double* x,
    const double* grad_out,
    double* grad_x,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    double x_local = x[i];
    double grad_x_local = 0.0;
    double y_local = 0.0;
    double grad_y_local = grad_out[i];

    __enzyme_autodiff(
        square_impl,
        enzyme_dup, &x_local, &grad_x_local,
        enzyme_dup, &y_local, &grad_y_local
    );

    grad_x[i] = grad_x_local;
}

__device__ __forceinline__
fsrs_params_t fsrs7_params_from_flat(const float* p) {
    return fsrs_params_t{
        .s0_again = p[0],
        .s0_hard = p[1],
        .s0_good = p[2],
        .s0_easy = p[3],
        .init_d0 = p[4],
        .init_d1 = p[5],
        .next_d_mult = p[6],
        .long_sinc_base = p[7],
        .long_sinc_s_exp = p[8],
        .long_sinc_r_mult = p[9],
        .long_fail_mult = p[10],
        .long_fail_d_exp = p[11],
        .long_fail_s_exp = p[12],
        .long_fail_r_mult = p[13],
        .long_hard_penalty = p[14],
        .long_easy_bonus = p[15],
        .short_sinc_base = p[16],
        .short_sinc_s_exp = p[17],
        .short_sinc_r_mult = p[18],
        .short_fail_mult = p[19],
        .short_fail_d_exp = p[20],
        .short_fail_s_exp = p[21],
        .short_fail_r_mult = p[22],
        .short_hard_penalty = p[23],
        .short_easy_bonus = p[24],
        .transition_decay = p[25],
        .transition_scale = p[26],
        .decay1 = p[27],
        .decay2 = p[28],
        .base1 = p[29],
        .base2 = p[30],
        .base_weight1 = p[31],
        .base_weight2 = p[32],
        .s_weight_power1 = p[33],
        .s_weight_power2 = p[34],
    };
}

__global__ void fsrs_test_kernel(
    const float* __restrict__ elapsed_days_real_flat,
    const int8_t* __restrict__ rating_flat,
    const int32_t* __restrict__ start_index,
    const int32_t* __restrict__ seq_len,
    const float* __restrict__ fsrs_params,
    const int32_t N,
    float* __restrict__ p
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int32_t start = start_index[i];
    const int32_t len = seq_len[i];
    const fsrs_params_t params = fsrs7_params_from_flat(fsrs_params + i * 35);

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
    // std::cout << state.s << ' ' << elapsed_days_real_flat[target_index] << '\n';
    p[i] = fsrs7_forgetting_curve(
        params,
        elapsed_days_real_flat[target_index],
        state.s
    );
}

extern "C" void fsrs_test_cuda(
    const float* elapsed_days_real_flat,
    const int8_t* rating_flat,
    const int32_t* start_index,
    const int32_t* seq_len,
    const float* fsrs_params,
    float* p,
    const int32_t N,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    int blocks = static_cast<int>((N + threads - 1) / threads);
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

extern "C" void enzyme_square_backward_cuda(
    const double* x,
    const double* grad_out,
    double* grad_x,
    int64_t n,
    cudaStream_t stream
) {
    if (n <= 0) {
        return;
    }

    constexpr int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    square_backward_kernel<<<blocks, threads, 0, stream>>>(x, grad_out, grad_x, n);
}
