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
        m.def("fsrs6_forward_verify(Tensor parameters, Tensor rating_L, Tensor elapsed_days_real_L, Tensor elapsed_days_int_L, Tensor label_elapsed_days_real_L, Tensor label_elapsed_days_int_L) -> Tensor");
        m.def("fsrs6_backward_verify(Tensor parameters, Tensor state_2) -> Tensor");
    }
}