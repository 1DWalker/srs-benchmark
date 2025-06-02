#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include "rwkv7_cuda_utils.h"

using namespace nvcuda;

namespace rwkv {
template <int CHUNK_LEN=32, typename F>
std::tuple<at::Tensor, at::Tensor> rwkv7_packed_wkv_forward_cuda(
    const at::Tensor& indices_I,
    const at::Tensor& r_THK, 
    const at::Tensor& k_THK,
    const at::Tensor& v_THK,
    const at::Tensor& w_THK,
    const at::Tensor& a_THK,
    const at::Tensor& k_deformed_THK
    ) {
    printf("Here forward\n");
    const int T = r_THK.size(0);
    const int H = r_THK.size(1);
    const int K = r_THK.size(2);
    TORCH_INTERNAL_ASSERT(r_THK.device().type() == at::DeviceType::CUDA);
    const F* r_ptr = (F*)r_THK.data_ptr();
    const F* k_ptr = (F*)k_THK.data_ptr();
    const F* v_ptr = (F*)v_THK.data_ptr();
    const float* w_ptr = w_THK.data_ptr<float>();
    const F* a_ptr = (F*)a_THK.data_ptr();
    const F* k_deformed_ptr = (F*)k_deformed_THK.data_ptr();
    
    at::Tensor out_THK = torch::empty(r_THK.sizes(), r_THK.options());
    F* out_ptr = (F*)out_THK.data_ptr();
    int L = (T + CHUNK_LEN) / CHUNK_LEN;
    at::Tensor state_checkpoints_LHKK = torch::empty({L, H, K, K}, r_THK.options().dtype(torch::kFloat32)).requires_grad_(false);
    float* state_checkpoints_ptr = state_checkpoints_LHKK.data_ptr<float>();
    return std::make_tuple(out_THK, state_checkpoints_LHKK);
}

template <int CHUNK_LEN=32, typename F>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> rwkv7_packed_wkv_backward_cuda(
    const at::Tensor& indices_I,
    const at::Tensor& r_THK, 
    const at::Tensor& k_THK,
    const at::Tensor& v_THK,
    const at::Tensor& w_THK,
    const at::Tensor& a_THK,
    const at::Tensor& k_deformed_THK,
    const at::Tensor& state_checkpoints_LHKK,
    const at::Tensor& grad_THK
    ) {
    printf("Here backwar\n");
    const int T = r_THK.size(0);
    const int H = r_THK.size(1);
    const int K = r_THK.size(2);
    const int L = state_checkpoints_LHKK.size(1);
    TORCH_INTERNAL_ASSERT(r_THK.device().type() == at::DeviceType::CUDA);
    const F* r_ptr = (F*)r_THK.data_ptr();
    const F* k_ptr = (F*)k_THK.data_ptr();
    const F* v_ptr = (F*)v_THK.data_ptr();
    const float* w_ptr = w_THK.data_ptr<float>();
    const F* a_ptr = (F*)a_THK.data_ptr();
    const F* k_deformed_ptr = (F*)k_deformed_THK.data_ptr();
    const float* state_checkpoints_ptr = state_checkpoints_LHKK.data_ptr<float>();
    const F* grad_ptr = (F*)grad_THK.data_ptr();
    at::Tensor r_grad_THK = torch::zeros_like(r_THK);
    at::Tensor k_grad_THK = torch::zeros_like(r_THK);
    at::Tensor v_grad_THK = torch::zeros_like(r_THK);
    at::Tensor w_grad_THK = torch::zeros_like(r_THK, torch::dtype(torch::kFloat32));
    at::Tensor a_grad_THK = torch::zeros_like(r_THK);
    at::Tensor k_deformed_grad_THK = torch::zeros_like(r_THK);
    F* r_grad_ptr = (F*)r_grad_THK.data_ptr();
    F* k_grad_ptr = (F*)k_grad_THK.data_ptr();
    F* v_grad_ptr = (F*)v_grad_THK.data_ptr();
    float* w_grad_ptr = w_grad_THK.data_ptr<float>();
    F* a_grad_ptr = (F*)a_grad_THK.data_ptr();
    F* k_deformed_grad_ptr = (F*)k_deformed_grad_THK.data_ptr();

    return std::make_tuple(r_grad_THK, k_grad_THK, v_grad_THK, w_grad_THK, a_grad_THK, k_deformed_grad_THK);
}

const int CHECKPOINT_LEN = 32;
TORCH_LIBRARY_IMPL(rwkv, CUDA, m) {
    m.impl("rwkv7_packed_wkv_forward_float", &rwkv7_packed_wkv_forward_cuda<CHECKPOINT_LEN, float>);
    m.impl("rwkv7_packed_wkv_backward_float", &rwkv7_packed_wkv_backward_cuda<CHECKPOINT_LEN, float>);
    m.impl("rwkv7_packed_wkv_forward_bfloat16", &rwkv7_packed_wkv_forward_cuda<CHECKPOINT_LEN, __nv_bfloat16>);
    m.impl("rwkv7_packed_wkv_backward_bfloat16", &rwkv7_packed_wkv_backward_cuda<CHECKPOINT_LEN, __nv_bfloat16>);
    m.impl("rwkv7_packed_wkv_forward_half", &rwkv7_packed_wkv_forward_cuda<CHECKPOINT_LEN, __half>);
    m.impl("rwkv7_packed_wkv_backward_half", &rwkv7_packed_wkv_backward_cuda<CHECKPOINT_LEN, __half>);
}
}