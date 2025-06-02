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
std::tuple<at::Tensor, at::Tensor> rwkv7_wkv_forward_cuda(
    const at::Tensor& indices_I,
    const at::Tensor& r_BTHK, 
    const at::Tensor& k_BTHK,
    const at::Tensor& v_BTHK,
    const at::Tensor& w_BTHK,
    const at::Tensor& a_BTHK,
    const at::Tensor& k_deformed_BTHK
    ) {
    const int B = r_BTHK.size(0);
    const int T = r_BTHK.size(1);
    const int H = r_BTHK.size(2);
    const int K = r_BTHK.size(3);
    TORCH_INTERNAL_ASSERT(r_BTHK.device().type() == at::DeviceType::CUDA);
    const F* r_ptr = (F*)r_BTHK.data_ptr();
    const F* k_ptr = (F*)k_BTHK.data_ptr();
    const F* v_ptr = (F*)v_BTHK.data_ptr();
    const float* w_ptr = w_BTHK.data_ptr<float>();
    const F* a_ptr = (F*)a_BTHK.data_ptr();
    const F* k_deformed_ptr = (F*)k_deformed_BTHK.data_ptr();
    
    at::Tensor out_BTHK = torch::empty(r_BTHK.sizes(), r_BTHK.options());
    F* out_ptr = (F*)out_BTHK.data_ptr();
    int L = (T + CHUNK_LEN) / CHUNK_LEN;
    at::Tensor state_checkpoints_BLHKK = torch::empty({B, L, H, K, K}, r_BTHK.options().dtype(torch::kFloat32)).requires_grad_(false);
    float* state_checkpoints_ptr = state_checkpoints_BLHKK.data_ptr<float>();
    printf("Here forward\n");
    return std::make_tuple(out_BTHK, state_checkpoints_BLHKK);
}

template <int CHUNK_LEN=32, typename F>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> rwkv7_wkv_backward_cuda(
    const at::Tensor& indices_I,
    const at::Tensor& r_BTHK, 
    const at::Tensor& k_BTHK,
    const at::Tensor& v_BTHK,
    const at::Tensor& w_BTHK,
    const at::Tensor& a_BTHK,
    const at::Tensor& k_deformed_BTHK,
    const at::Tensor& state_checkpoints_BLHKK,
    const at::Tensor& grad_BTHK
    ) {
    const int B = r_BTHK.size(0);
    const int T = r_BTHK.size(1);
    const int H = r_BTHK.size(2);
    const int K = r_BTHK.size(3);
    const int L = state_checkpoints_BLHKK.size(1);
    TORCH_INTERNAL_ASSERT(r_BTHK.device().type() == at::DeviceType::CUDA);
    const F* r_ptr = (F*)r_BTHK.data_ptr();
    const F* k_ptr = (F*)k_BTHK.data_ptr();
    const F* v_ptr = (F*)v_BTHK.data_ptr();
    const float* w_ptr = w_BTHK.data_ptr<float>();
    const F* a_ptr = (F*)a_BTHK.data_ptr();
    const F* k_deformed_ptr = (F*)k_deformed_BTHK.data_ptr();
    const float* state_checkpoints_ptr = state_checkpoints_BLHKK.data_ptr<float>();
    const F* grad_ptr = (F*)grad_BTHK.data_ptr();
    at::Tensor r_grad_BTHK = torch::zeros_like(r_BTHK);
    at::Tensor k_grad_BTHK = torch::zeros_like(r_BTHK);
    at::Tensor v_grad_BTHK = torch::zeros_like(r_BTHK);
    at::Tensor w_grad_BTHK = torch::zeros_like(r_BTHK, torch::dtype(torch::kFloat32));
    at::Tensor a_grad_BTHK = torch::zeros_like(r_BTHK);
    at::Tensor k_deformed_grad_BTHK = torch::zeros_like(r_BTHK);
    F* r_grad_ptr = (F*)r_grad_BTHK.data_ptr();
    F* k_grad_ptr = (F*)k_grad_BTHK.data_ptr();
    F* v_grad_ptr = (F*)v_grad_BTHK.data_ptr();
    float* w_grad_ptr = w_grad_BTHK.data_ptr<float>();
    F* a_grad_ptr = (F*)a_grad_BTHK.data_ptr();
    F* k_deformed_grad_ptr = (F*)k_deformed_grad_BTHK.data_ptr();

    printf("Here backwar\n");
    return std::make_tuple(r_grad_BTHK, k_grad_BTHK, v_grad_BTHK, w_grad_BTHK, a_grad_BTHK, k_deformed_grad_BTHK);
}

const int CHECKPOINT_LEN = 32;
TORCH_LIBRARY_IMPL(rwkv, CUDA, m) {
    m.impl("rwkv7_packed_wkv_forward_float", &rwkv7_wkv_forward_cuda<CHECKPOINT_LEN, float>);
    m.impl("rwkv7_packed_wkv_backward_float", &rwkv7_wkv_backward_cuda<CHECKPOINT_LEN, float>);
    m.impl("rwkv7_packed_wkv_forward_bfloat16", &rwkv7_wkv_forward_cuda<CHECKPOINT_LEN, __nv_bfloat16>);
    m.impl("rwkv7_packed_wkv_backward_bfloat16", &rwkv7_wkv_backward_cuda<CHECKPOINT_LEN, __nv_bfloat16>);
    m.impl("rwkv7_packed_wkv_forward_half", &rwkv7_wkv_forward_cuda<CHECKPOINT_LEN, __half>);
    m.impl("rwkv7_packed_wkv_backward_half", &rwkv7_wkv_backward_cuda<CHECKPOINT_LEN, __half>);
}
}