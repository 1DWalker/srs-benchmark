#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <stdio.h>
#include "buffer.hpp"

#include "fsrs/fsrs7_constants.cuh"
#include "fsrs/fsrs_test.cuh"
#include "fsrs/fsrs_train.cuh"


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
    constexpr int64_t fsrs_param_count =
        static_cast<int64_t>(sizeof(fsrs_params_t) / sizeof(float));
    TORCH_CHECK(
        fsrs_params.dim() == 2 && fsrs_params.size(0) == N && fsrs_params.size(1) == fsrs_param_count,
        "fsrs_params must have shape (N, ", fsrs_param_count, ")"
    );

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

constexpr int THREADS_PER_BLOCK = 256; // must be a multiple of 32
StateBuffer<fsrs_state_t> state_buffer;

torch::Tensor fsrs7_train(
    const torch::Tensor& elapsed_days_real_flat,
    const torch::Tensor& rating_flat,
    const torch::Tensor& start_index_UxT,
    const torch::Tensor& seq_len_UxT,
    const torch::Tensor& seq_len_Ux_max,
    const torch::Tensor& seq_len_Ux_max_cumsum,
    const torch::Tensor& fsrs_params_UP,
    const int buffer_req_size
) {
    check_sample_tensor(elapsed_days_real_flat, "elapsed_days_real_flat", torch::kFloat32);
    check_sample_tensor(rating_flat, "rating_flat", torch::kInt8);
    check_sample_tensor(start_index_UxT, "start index", torch::kInt32);
    check_sample_tensor(seq_len_UxT, "seq_len", torch::kInt32);
    check_sample_tensor(seq_len_Ux_max, "seq_len max", torch::kInt32);
    check_sample_tensor(seq_len_Ux_max_cumsum, "seq_len max cumsum", torch::kInt32);
    check_sample_tensor(fsrs_params_UP, "fsrs_params", torch::kFloat32);
    const int U = seq_len_UxT.size(0);
    const int x = seq_len_UxT.size(1);
    const int T = seq_len_UxT.size(2);
    const int P = fsrs_params_UP.size(1);

    c10::cuda::CUDAGuard device_guard(elapsed_days_real_flat.device());
    fsrs_state_t *state_buffer_ptr = state_buffer.ensure(buffer_req_size);
    torch::Tensor grad = torch::zeros(
        {U, x * T, P},
        fsrs_params_UP.options()
    );

    fsrs_train_cuda(
        elapsed_days_real_flat.data_ptr<float>(),
        rating_flat.data_ptr<int8_t>(),
        start_index_UxT.data_ptr<int32_t>(),
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fsrs7_train", &fsrs7_train, "fsrs7 gradient");
    m.def("fsrs7_test", &fsrs7_test, "fsrs7 test forward pass");
    m.def("threads_per_block", &threads_per_block, "threads per block");
}
