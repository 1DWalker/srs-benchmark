#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

extern "C" {
  PyObject* PyInit_RWKV_CUDA(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "RWKV_CUDA",
          NULL,
          -1,
          NULL,
      };
      return PyModule_Create(&module_def);
  }
}

inline int64_t get_index1(int b, int t, int T) {
    return (int64_t) b * T + t;
}

inline int64_t get_index2(int b, int t, int h, int T, int H) {
    return ((int64_t) b * T + t) * H + h;
}

inline int64_t get_index3(int b, int t, int h, int k, int T, int H, int K) {
    return (((int64_t) b * T + t) * H + h) * K + k;
}

inline int64_t get_index4(int b, int t, int h, int k, int k2, int T, int H, int K, int K2) {
    return ((((int64_t) b * T + t) * H + h) * K + k) * K2 + k2;
}
template <int CHUNK_LEN=32, typename F>
std::tuple<at::Tensor, at::Tensor> rwkv7_wkv_forward_cpu(
    const at::Tensor& r_BTHK, 
    const at::Tensor& k_BTHK,
    const at::Tensor& v_BTHK,
    const at::Tensor& w_BTHK,
    const at::Tensor& a_BTHK,
    const at::Tensor& k_deformed_BTHK,
    const at::Tensor& skip_BT
    ) {
    const int B = r_BTHK.size(0);
    const int T = r_BTHK.size(1);
    const int H = r_BTHK.size(2);
    const int K = 32;
    TORCH_INTERNAL_ASSERT(r_BTHK.size(3) == K);
    TORCH_INTERNAL_ASSERT(r_BTHK.device().type() == at::DeviceType::CPU);
    const F* r_ptr = (F*)r_BTHK.data_ptr();
    const F* k_ptr = (F*)k_BTHK.data_ptr();
    const F* v_ptr = (F*)v_BTHK.data_ptr();
    const float* w_ptr = w_BTHK.data_ptr<float>();
    const F* a_ptr = (F*)a_BTHK.data_ptr();
    const F* k_deformed_ptr = (F*)k_deformed_BTHK.data_ptr();
    const bool* skip_ptr = (bool*)skip_BT.data_ptr();
    
    at::Tensor out_BTHK = torch::empty(r_BTHK.sizes(), r_BTHK.options());
    F* out_ptr = (F*)out_BTHK.data_ptr();
    int L = (T + CHUNK_LEN) / CHUNK_LEN;
    // Not supported
    at::Tensor state_checkpoints_BLHKK = torch::empty({0, 0, 0, 0, 0}, r_BTHK.options().dtype(torch::kFloat32)).requires_grad_(false);

    F state[K][K];
    F state_times_w[K][K];
    F state_times_k_deformed[K];
    F buf[K];
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            std::memset(state, 0, sizeof(state));
            for (int t = 0; t < T; t++) {
                // Decay and removal
                for (int i = 0; i < K; i++) {
                    for (int j = 0; j < K; j++) {
                        state_times_w[i][j] = state[i][j] * w_ptr[get_index3(b, t, h, j, T, H, K)];
                    }
                }
                memset(state_times_k_deformed, 0, sizeof(state_times_k_deformed));
                for (int i = 0; i < K; i++) {
                    for (int j = 0; j < K; j++) {
                        state_times_k_deformed[i] += state[i][j] * k_deformed_ptr[get_index3(b, t, h, j, T, H, K)];
                    }
                }
                for (int i = 0; i < K; i++) {
                    for (int j = 0; j < K; j++) {
                        F a_j = a_ptr[get_index3(b, t, h, j, T, H, K)];
                        F k_deformed_j = k_deformed_ptr[get_index3(b, t, h, j, T, H, K)];
                        state[i][j] = state_times_w[i][j] - state_times_k_deformed[i] * a_j * k_deformed_j;
                    }
                }

                // Add
                for (int i = 0; i < K; i++) {
                    F v_i = v_ptr[get_index3(b, t, h, i, T, H, K)];
                    for (int j = 0; j < K; j++) {
                        F k_j = k_ptr[get_index3(b, t, h, j, T, H, K)];
                        state[i][j] += v_i * k_j;
                    }
                }

                memset(buf, 0, sizeof(buf));
                // Compute S@r
                for (int i = 0; i < K; i++) {
                    for (int j = 0; j < K; j++) {
                        buf[i] += state[i][j] * r_ptr[get_index3(b, t, h, j, T, H, K)];
                    }
                }
                for (int i = 0; i < K; i++) {
                    out_ptr[get_index3(b, t, h, i, T, H, K)] = buf[i];
                }

            }
        }
    }
    return std::make_tuple(out_BTHK, state_checkpoints_BLHKK);
}

template <int CHUNK_LEN=32, typename F>
std::tuple<at::Tensor, at::Tensor> rwkv7_packed_wkv_forward_cpu(
    const at::Tensor& indices_I,
    const at::Tensor& r_THK, 
    const at::Tensor& k_THK,
    const at::Tensor& v_THK,
    const at::Tensor& w_THK,
    const at::Tensor& a_THK,
    const at::Tensor& k_deformed_THK
    ) {
    const int I = indices_I.size(0);
    const int T = r_THK.size(0);
    const int H = r_THK.size(1);
    const int K = 32;
    TORCH_INTERNAL_ASSERT(r_THK.size(2) == K);
    TORCH_INTERNAL_ASSERT(r_THK.device().type() == at::DeviceType::CPU);
    const int64_t* indices_ptr = (int64_t*)indices_I.data_ptr();
    const F* r_ptr = (F*)r_THK.data_ptr();
    const F* k_ptr = (F*)k_THK.data_ptr();
    const F* v_ptr = (F*)v_THK.data_ptr();
    const float* w_ptr = w_THK.data_ptr<float>();
    const F* a_ptr = (F*)a_THK.data_ptr();
    const F* k_deformed_ptr = (F*)k_deformed_THK.data_ptr();
    
    at::Tensor out_THK = torch::empty(r_THK.sizes(), r_THK.options());
    F* out_ptr = (F*)out_THK.data_ptr();
    int L = (T + CHUNK_LEN) / CHUNK_LEN;
    // Not supported
    at::Tensor state_checkpoints_LHKK = torch::empty({0, 0, 0, 0}, r_THK.options().dtype(torch::kFloat32)).requires_grad_(false);

    F state[K][K];
    F state_times_w[K][K];
    F state_times_k_deformed[K];
    F buf[K];
    for (int h = 0; h < H; h++) {
        int indices_l = 0;
        for (int t = 0; t < T; t++) {
            if (indices_l < I && t == indices_ptr[indices_l]) {
                std::memset(state, 0, sizeof(state));
                indices_l++;
            }
            // Decay and removal
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    state_times_w[i][j] = state[i][j] * w_ptr[get_index2(t, h, j, H, K)];
                }
            }
            memset(state_times_k_deformed, 0, sizeof(state_times_k_deformed));
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    state_times_k_deformed[i] += state[i][j] * k_deformed_ptr[get_index2(t, h, j, H, K)];
                }
            }
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    F a_j = a_ptr[get_index2(t, h, j, H, K)];
                    F k_deformed_j = k_deformed_ptr[get_index2(t, h, j, H, K)];
                    state[i][j] = state_times_w[i][j] - state_times_k_deformed[i] * a_j * k_deformed_j;
                }
            }

            // Add
            for (int i = 0; i < K; i++) {
                F v_i = v_ptr[get_index2(t, h, i, H, K)];
                for (int j = 0; j < K; j++) {
                    F k_j = k_ptr[get_index2(t, h, j, H, K)];
                    state[i][j] += v_i * k_j;
                }
            }

            memset(buf, 0, sizeof(buf));
            // Compute S@r
            for (int i = 0; i < K; i++) {
                for (int j = 0; j < K; j++) {
                    buf[i] += state[i][j] * r_ptr[get_index2(t, h, j, H, K)];
                }
            }
            for (int i = 0; i < K; i++) {
                out_ptr[get_index2(t, h, i, H, K)] = buf[i];
            }

        }
    }
    return std::make_tuple(out_THK, state_checkpoints_LHKK);
}

namespace rwkv {
    TORCH_LIBRARY(rwkv, m) {
        m.def("rwkv7_wkv_forward_float(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH) -> (Tensor, Tensor)");
        m.def("rwkv7_wkv_backward_float(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, Tensor state_checkpoints_BLHKK, Tensor grad_BTHK) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
        m.def("rwkv7_wkv_forward_bfloat16(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH) -> (Tensor, Tensor)");
        m.def("rwkv7_wkv_backward_bfloat16(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, Tensor state_checkpoints_BLHKK, Tensor grad_BTHK) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
        m.def("rwkv7_wkv_forward_half(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH) -> (Tensor, Tensor)");
        m.def("rwkv7_wkv_backward_half(Tensor r_BTHK, Tensor k_BTHK, Tensor v_BTHK, Tensor w_BTHK, Tensor a_BTHK, Tensor k_deformed_BTHK, Tensor skip_BTH, Tensor state_checkpoints_BLHKK, Tensor grad_BTHK) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
        m.def("rwkv7_packed_wkv_forward_float(Tensor indices_I, Tensor r_THK, Tensor k_THK, Tensor V_THK, Tensor w_THK, Tensor a_THK, Tensor k_deformed_THK) -> (Tensor, Tensor)");
        m.def("rwkv7_packed_wkv_backward_float(Tensor indices_I, Tensor r_THK, Tensor k_THK, Tensor V_THK, Tensor w_THK, Tensor a_THK, Tensor k_deformed_THK, Tensor state_checkpoints_LHKK, Tensor grad_THK) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
        m.def("rwkv7_packed_wkv_forward_bfloat16(Tensor indices_I, Tensor r_THK, Tensor k_THK, Tensor V_THK, Tensor w_THK, Tensor a_THK, Tensor k_deformed_THK) -> (Tensor, Tensor)");
        m.def("rwkv7_packed_wkv_backward_bfloat16(Tensor indices_I, Tensor r_THK, Tensor k_THK, Tensor V_THK, Tensor w_THK, Tensor a_THK, Tensor k_deformed_THK, Tensor state_checkpoints_LHKK, Tensor grad_THK) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
        m.def("rwkv7_packed_wkv_forward_half(Tensor indices_I, Tensor r_THK, Tensor k_THK, Tensor V_THK, Tensor w_THK, Tensor a_THK, Tensor k_deformed_THK) -> (Tensor, Tensor)");
        m.def("rwkv7_packed_wkv_backward_half(Tensor indices_I, Tensor r_THK, Tensor k_THK, Tensor V_THK, Tensor w_THK, Tensor a_THK, Tensor k_deformed_THK, Tensor state_checkpoints_LHKK, Tensor grad_THK) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    }
    const int CHECKPOINT_LEN = 32;
    TORCH_LIBRARY_IMPL(rwkv, CPU, m) {
        m.impl("rwkv7_wkv_forward_float", &rwkv7_wkv_forward_cpu<CHECKPOINT_LEN, float>);
        m.impl("rwkv7_packed_wkv_forward_float", &rwkv7_packed_wkv_forward_cpu<CHECKPOINT_LEN, float>);
    }
}
