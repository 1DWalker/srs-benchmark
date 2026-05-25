#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "fsrs7.cu"

int __device__ enzyme_dup;
int __device__ enzyme_dupnoneed;
int __device__ enzyme_out;
int __device__ enzyme_const;

template < typename return_type, typename ... T >
return_type __device__ __enzyme_autodiff(void*, T ... );

__global__ void fsrs_train_kernel(
    const float* __restrict__ elapsed_days_real_flat,
    const int8_t* __restrict__ rating_flat,
    const int32_t* __restrict__ start_index,
    const int32_t* __restrict__ seq_len,
    const fsrs_params_t* __restrict__ fsrs_params,
    const int32_t N,
    float* __restrict__ p
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
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
        state.s
    );

    // float x = 5.0;
    // float y = 2.0;
    // // float df_dx = __enzyme_fwddiff<float>((void*)square, enzyme_dup, x, dx); 
    // auto [df_dx, df_dy] = __enzyme_autodiff<float2>((void*)square, enzyme_out, x, enzyme_out, y); 
    // p[i] = df_dx;
    // printf("%f %f\n", df_dx, df_dy);
    B b = {3.0, 2.0};
    B db;
    A a;
    // A lambda = {1.0, 0.0};
    // A lambda = {0.0, 1.0};
    A lambda = {0.0, 1.0, 1.0};
    // auto db = __enzyme_autodiff<B>((void*)bar, enzyme_out, b); 
    // auto db = __enzyme_autodiff<B>((void*)bar, enzyme_out, b); 
    __enzyme_autodiff<B>((void*)barwrap, enzyme_dup, b, &db, enzyme_dupnoneed, a, lambda); 
    printf("%f %f\n", b.a, b.b);
    // printf("%f %f\n", db.x, db.y);
    printf("%f %f\n", db.a, db.b);
    // printf("%f %f\n", a.a, a.b);
    // p[i] = square(p[i]);
}

extern "C" void fsrs_train_cuda(
    // const float* __restrict__ elapsed_days_real_flat,
    // const int8_t* __restrict__ rating_flat,
    // const int32_t* __restrict__ start_index,
    // const int32_t* __restrict__ seq_len,
    // const fsrs_params_t* __restrict__ fsrs_params,
    // float* __restrict__ p,
    // const int32_t N,
    cudaStream_t stream
) {
    std::cout << "Hello world!\n";
    // constexpr int threads = 256;
    // int blocks = static_cast<int>((N + threads - 1) / threads);
    // fsrs_train_kernel<<<blocks, threads, 0, stream>>>(
    //     elapsed_days_real_flat,
    //     rating_flat,
    //     start_index,
    //     seq_len,
    //     fsrs_params,
    //     N,
    //     p
    // );
}
