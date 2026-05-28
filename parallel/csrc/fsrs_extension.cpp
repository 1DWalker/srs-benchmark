#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <stdio.h>

#include "fsrs/fsrs7.cuh"
#include "fsrs_kernel/fsrs_test.cuh"
#include "fsrs_kernel/fsrs_train.cuh"


namespace {
void check_sample_tensor(
    const torch::Tensor& tensor,
    const char* name,
    const c10::ScalarType dtype
) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
}

constexpr int64_t fsrs_param_count() {
    return static_cast<int64_t>(sizeof(fsrs_params_t) / sizeof(float));
}

void check_fsrs_params_tensor(
    const torch::Tensor& tensor,
    const char* name,
    const int64_t rows
) {
    check_sample_tensor(tensor, name, torch::kFloat32);
    constexpr int64_t param_bytes = static_cast<int64_t>(sizeof(fsrs_params_t));
    constexpr int64_t float_bytes = static_cast<int64_t>(sizeof(float));
    constexpr int64_t param_count = fsrs_param_count();
    TORCH_CHECK(
        param_count * float_bytes == param_bytes,
        "sizeof(fsrs_params_t) must be representable as a whole number of float32 values"
    );
    TORCH_CHECK(
        tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == param_count,
        name, " must have shape (", rows, ", ", param_count, ")"
    );
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

    c10::cuda::CUDAGuard device_guard(elapsed_days_real_flat.device());
    const int32_t N = start_index.numel();
    check_fsrs_params_tensor(fsrs_params, "fsrs_params", N);

    torch::Tensor p = torch::empty(
        start_index.sizes(),
        fsrs_params.options()
    );

    fsrs_test_cuda(
        elapsed_days_real_flat.data_ptr<float>(),
        rating_flat.data_ptr<int8_t>(),
        start_index.data_ptr<int32_t>(),
        seq_len.data_ptr<int32_t>(),
        reinterpret_cast<const fsrs_params_t*>(fsrs_params.data_ptr<float>()),
        p.data_ptr<float>(),
        N,
        at::cuda::getCurrentCUDAStream().stream()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return p;
}

constexpr int THREADS_PER_BLOCK = 128; // must be a multiple of 32 and a divisor of the batch size
constexpr int64_t TRAIN_SCRATCH_BYTES = 500LL * 1000LL * 1000LL;

torch::Tensor fsrs7_train_dispatch(
    const torch::Tensor& elapsed_days_real_flat,
    const torch::Tensor& rating_flat,
    const torch::Tensor& start_index_UxT,
    const torch::Tensor& grad_weight_UxT,
    const torch::Tensor& seq_len_UxT,
    const torch::Tensor& seq_len_Ux_max,
    const torch::Tensor& seq_len_Ux_max_cumsum,
    const torch::Tensor& fsrs_params_UP
) {
    check_sample_tensor(elapsed_days_real_flat, "elapsed_days_real_flat", torch::kFloat32);
    check_sample_tensor(rating_flat, "rating_flat", torch::kInt8);
    check_sample_tensor(start_index_UxT, "start index", torch::kInt32);
    check_sample_tensor(grad_weight_UxT, "grad weight", torch::kFloat32);
    check_sample_tensor(seq_len_UxT, "seq_len", torch::kInt32);
    check_sample_tensor(seq_len_Ux_max, "seq_len max", torch::kInt32);
    check_sample_tensor(seq_len_Ux_max_cumsum, "seq_len max cumsum", torch::kInt32);

    const int U = seq_len_UxT.size(0);
    const int x = seq_len_UxT.size(1);
    const int T = seq_len_UxT.size(2);
    check_fsrs_params_tensor(fsrs_params_UP, "fsrs_params_UP", U);
    constexpr int64_t P = fsrs_param_count();
    TORCH_CHECK(T == THREADS_PER_BLOCK, "seq_len_UxT last dimension must equal THREADS_PER_BLOCK");

    c10::cuda::CUDAGuard device_guard(elapsed_days_real_flat.device());
    torch::Tensor state_buffer_tensor = torch::empty(
        {TRAIN_SCRATCH_BYTES},
        torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(elapsed_days_real_flat.device())
    );
    fsrs_state_t *state_buffer_ptr =
        reinterpret_cast<fsrs_state_t*>(state_buffer_tensor.data_ptr<uint8_t>());
    torch::Tensor grad = torch::zeros(
        {U, x * T, P},
        fsrs_params_UP.options()
    );

    fsrs_train_cuda(
        elapsed_days_real_flat.data_ptr<float>(),
        rating_flat.data_ptr<int8_t>(),
        start_index_UxT.data_ptr<int32_t>(),
        grad_weight_UxT.data_ptr<float>(),
        seq_len_UxT.data_ptr<int32_t>(),
        seq_len_Ux_max.data_ptr<int32_t>(),
        seq_len_Ux_max_cumsum.data_ptr<int32_t>(),
        reinterpret_cast<const fsrs_params_t*>(fsrs_params_UP.data_ptr<float>()),
        U,
        x,
        THREADS_PER_BLOCK,
        at::cuda::getCurrentCUDAStream().stream(),
        state_buffer_ptr,
        reinterpret_cast<fsrs_params_t*>(grad.data_ptr<float>())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return grad;
}

int threads_per_block() {
    return THREADS_PER_BLOCK;
}

TORCH_LIBRARY(srs, m) {
    m.def(
        "fsrs7_train("
        "Tensor elapsed_days_real_flat, "
        "Tensor rating_flat, "
        "Tensor start_index_UxT, "
        "Tensor grad_weight_UxT, "
        "Tensor seq_len_UxT, "
        "Tensor seq_len_Ux_max, "
        "Tensor seq_len_Ux_max_cumsum, "
        "Tensor fsrs_params_UP"
        ") -> Tensor"
    );
}

TORCH_LIBRARY_IMPL(srs, CUDA, m) {
    m.impl("fsrs7_train", &fsrs7_train_dispatch);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fsrs7_test", &fsrs7_test, "fsrs7 test forward pass");
    m.def("threads_per_block", &threads_per_block, "threads per block");
}
