#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>


extern "C" void fsrs_test_cuda(
    const float* elapsed_days_real_flat,
    const int8_t* rating_flat,
    const int32_t* start_index,
    const int32_t* seq_len,
    const float* fsrs_params,
    float* p,
    int64_t num_sequences,
    cudaStream_t stream
);

extern "C" void enzyme_square_backward_cuda(
    const double* x,
    const double* grad_out,
    double* grad_x,
    int64_t n,
    cudaStream_t stream
);


namespace {

void check_sample_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_sample_tensor(
    const torch::Tensor& tensor,
    const char* name,
    const c10::ScalarType dtype
) {
    check_sample_tensor(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
}

}  // namespace

torch::Tensor fsrs7_test(
    const torch::Tensor& elapsed_days_real_flat,
    const torch::Tensor& rating_flat,
    const torch::Tensor& start_index,
    const torch::Tensor& seq_len,
    const torch::Tensor& fsrs_params
) {
    check_sample_tensor(elapsed_days_real_flat, "elapsed_days_real_flat", torch::kFloat32);
    check_sample_tensor(rating_flat, "rating_flat", torch::kInt8);
    check_sample_tensor(start_index, "start_index", torch::kInt32);
    check_sample_tensor(seq_len, "seq_len", torch::kInt32);
    check_sample_tensor(fsrs_params, "fsrs_params", torch::kFloat32);

    c10::cuda::CUDAGuard device_guard(elapsed_days_real_flat.device());
    const int32_t N = start_index.numel();

    torch::Tensor p = torch::empty(
        start_index.sizes(),
        fsrs_params.options()
    );

    fsrs_test_cuda(
        elapsed_days_real_flat.data_ptr<float>(),
        rating_flat.data_ptr<int8_t>(),
        start_index.data_ptr<int32_t>(),
        seq_len.data_ptr<int32_t>(),
        fsrs_params.data_ptr<float>(),
        p.data_ptr<float>(),
        N,
        at::cuda::getCurrentCUDAStream().stream()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return p;
}


torch::Tensor square_backward(torch::Tensor x, torch::Tensor grad_out) {
    check_sample_tensor(x, "x");
    check_sample_tensor(grad_out, "grad_out");
    TORCH_CHECK(x.sizes() == grad_out.sizes(), "x and grad_out must have the same shape");
    TORCH_CHECK(x.device() == grad_out.device(), "x and grad_out must be on the same CUDA device");

    c10::cuda::CUDAGuard device_guard(x.device());
    torch::Tensor grad_x = torch::empty_like(x);

    enzyme_square_backward_cuda(
        x.data_ptr<double>(),
        grad_out.data_ptr<double>(),
        grad_x.data_ptr<double>(),
        x.numel(),
        at::cuda::getCurrentCUDAStream().stream()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return grad_x;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fsrs7_forward", &fsrs7_test, "fsrs7 forward");
    m.def("square_backward", &square_backward, "Enzyme square backward (CUDA)");
}
