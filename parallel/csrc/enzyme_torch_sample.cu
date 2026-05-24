#include <cuda_runtime.h>
#include <stdint.h>

#include "fsrs/fsrs_test.cu"


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
