#pragma once

#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

extern "C" {
  PyObject* PyInit__FSRS_CPP(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_FSRS_CPP",
          NULL,
          -1,
          NULL,
      };
      return PyModule_Create(&module_def);
  }
}

namespace fsrs {
    TORCH_LIBRARY(fsrs, m) {
        m.def("fsrs6_forward_verify_float(Tensor parameters, Tensor rating_L, Tensor elapsed_days_real_L, Tensor elapsed_days_int_L, Tensor label_elapsed_days_real_L, Tensor label_elapsed_days_int_L) -> (Tensor, Tensor)");
        m.def("fsrs6_forward_verify_double(Tensor parameters, Tensor rating_L, Tensor elapsed_days_real_L, Tensor elapsed_days_int_L, Tensor label_elapsed_days_real_L, Tensor label_elapsed_days_int_L) -> (Tensor, Tensor)");
        m.def("fsrs6_backward_verify_float(Tensor grad_out, Tensor checkpoints_L, Tensor parameters, Tensor rating_L, Tensor elapsed_days_real_L, Tensor elapsed_days_int_L, Tensor label_elapsed_days_real_L, Tensor label_elapsed_days_int_L) -> Tensor");
        m.def("fsrs6_backward_verify_double(Tensor grad_out, Tensor checkpoints_L, Tensor parameters, Tensor rating_L, Tensor elapsed_days_real_L, Tensor elapsed_days_int_L, Tensor label_elapsed_days_real_L, Tensor label_elapsed_days_int_L) -> Tensor");
        m.def("fsrs_batch_forward(Tensor parameters, Tensor review_ths, Tensor packed_review_th_T, Tensor packed_rating_T, Tensor packed_elapsed_days_real_T, Tensor packed_elapsed_days_int_T, Tensor packed_label_elapsed_days_real_T, Tensor packed_label_elapsed_days_int_T, Tensor perm_T_tensor, Tensor perm_inv_T_tensor, Tensor card_locs_T) -> (Tensor, Tensor, Tensor)");
        m.def("fsrs_batch_backward(Tensor grad_out, Tensor parameters, Tensor review_ths, Tensor checkpoints, Tensor keys, Tensor packed_review_th_T, Tensor packed_rating_T, Tensor packed_elapsed_days_real_T, Tensor packed_elapsed_days_int_T, Tensor packed_label_elapsed_days_real_T, Tensor packed_label_elapsed_days_int_T, Tensor perm_T_tensor, Tensor perm_inv_T_tensor, Tensor card_locs_T) -> Tensor");
        m.def("fsrs_optimizer(Tensor pretrain_params, Tensor epochs, Tensor train_ords, Tensor locs, Tensor locs_lens, Tensor keys, Tensor keys_lens, Tensor train_set_locs, Tensor train_set_keys, Tensor test_set_locs, Tensor test_set_keys, Tensor packed_review_th_T, Tensor packed_rating_T, Tensor packed_elapsed_days_real_T, Tensor packed_elapsed_days_int_T, Tensor packed_label_elapsed_days_real_T, Tensor packed_label_elapsed_days_int_T) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
        m.def("compute_rmse_bins(Tensor y, Tensor y_pred, Tensor rmse_bin_ind) -> Tensor");
    }
}