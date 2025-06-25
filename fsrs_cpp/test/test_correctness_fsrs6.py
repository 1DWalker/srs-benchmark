"""Check that the python and c++ implementations of FSRS-6 match."""

import random
import time
import unittest
import numpy as np
import torch
from fsrs_cpp import _FSRS_CPP
from fsrs_cpp.fsrs6_reference import FSRS6

class TestCorrectness(unittest.TestCase):
    def setUp(self):
        self.params = torch.tensor([
            0.212,
            1.2931,
            2.3065,
            8.2956,
            6.4133,
            0.8334,
            3.0194,
            0.001,
            1.8722,
            0.1666,
            0.796,
            1.4835,
            0.0614,
            0.2629,
            1.6483,
            0.6014,
            1.8729,
            0.5425,
            0.0912,
            0.0658,
            0.1542,
        ])
        # self.params = torch.tensor([1.0000e-03, 1.7030e-01, 1.3751e+01, 1.8097e+01, 8.2403e+00, 1.0000e-03,
        #     3.7933e+00, 1.6532e-01, 1.2000e+00, 8.0000e-01, 9.9291e-02, 2.0709e+00,
        #     1.0000e-03, 7.8550e-01, 1.9190e+00, 0.0000e+00, 6.0000e+00, 0.0000e+00,
        #     0.0000e+00, 7.2275e-01, 8.0000e-01])


    def _test_forward(self, dtype):
        reference_fsrs = FSRS6()
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        for _ in range(100):
            L = random.randint(1, 50)
            rating_L = torch.tensor(np.random.randint(1, 5, size=L))
            elapsed_days_real_L = torch.tensor(np.random.uniform(0.1, 400.0, size=L), dtype=dtype)
            elapsed_days_int_L = torch.tensor(np.random.randint(0, 400, size=L), dtype=dtype)
            label_elapsed_days_real_L = torch.tensor(np.random.uniform(0.1, 400.0, size=L), dtype=dtype)
            label_elapsed_days_int_L = torch.tensor(np.random.randint(0, 400, size=L), dtype=dtype)
            params = self.params.clone().detach().to(dtype).requires_grad_(True)
            if dtype == torch.double:
                cpp_out, _ = torch.ops.fsrs.fsrs6_forward_verify_double(params, rating_L, elapsed_days_real_L, elapsed_days_int_L, label_elapsed_days_real_L, label_elapsed_days_int_L)
            else:
                cpp_out, _ = torch.ops.fsrs.fsrs6_forward_verify_float(params, rating_L, elapsed_days_real_L, elapsed_days_int_L, label_elapsed_days_real_L, label_elapsed_days_int_L)
            reference_out = reference_fsrs.forward(params, elapsed_days_real_L.unsqueeze(0), elapsed_days_int_L.unsqueeze(0), rating_L.unsqueeze(0), label_elapsed_days_real_L.unsqueeze(0), label_elapsed_days_int_L.unsqueeze(0))
            torch.testing.assert_close(cpp_out, reference_out.squeeze(0))
    
    def test_forward_double(self):
        self._test_forward(torch.double)

    def test_forward_float(self):
        self._test_forward(torch.float)

    def _test_backward(self, dtype, iters, L, atol=None, rtol=None):
        reference_fsrs = FSRS6()
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        for iter in range(iters):
            MX = random.randint(1, 400)
            print(f"iter: {iter + 1} / {iters}, L:", L, "MX:", MX)
            rating_L = torch.tensor(np.random.randint(1, 5, size=L))
            elapsed_days_real_L = torch.tensor(np.random.uniform(0.1, MX, size=L), dtype=dtype)
            elapsed_days_int_L = torch.tensor(np.random.randint(0, MX, size=L), dtype=dtype)
            label_elapsed_days_real_L = torch.tensor(np.random.uniform(0.1, MX, size=L), dtype=dtype)
            label_elapsed_days_int_L = torch.tensor(np.random.randint(0, MX, size=L), dtype=dtype)
            params = self.params.clone().detach().to(dtype).requires_grad_(True)
            print("start forward ref")
            ref_start = time.time()
            reference_out = reference_fsrs.forward(params, elapsed_days_real_L.unsqueeze(0), elapsed_days_int_L.unsqueeze(0), rating_L.unsqueeze(0), label_elapsed_days_real_L.unsqueeze(0), label_elapsed_days_int_L.unsqueeze(0))
            grad_out = torch.randn_like(reference_out)
            print("start backward ref")
            grad_reference = torch.autograd.grad(reference_out, params, grad_out)[0]
            print(f"done ref in {time.time() - ref_start:.4f} seconds")

            print("start forward cpp")
            cpp_start = time.time()
            if dtype == torch.double:
                _, checkpoints = torch.ops.fsrs.fsrs6_forward_verify_double(params, rating_L, elapsed_days_real_L, elapsed_days_int_L, label_elapsed_days_real_L, label_elapsed_days_int_L)
            else:
                _, checkpoints = torch.ops.fsrs.fsrs6_forward_verify_float(params, rating_L, elapsed_days_real_L, elapsed_days_int_L, label_elapsed_days_real_L, label_elapsed_days_int_L)
            print("start backward cpp")
            if dtype == torch.double:
                grad_cpp = torch.ops.fsrs.fsrs6_backward_verify_double(grad_out, checkpoints, params, rating_L, elapsed_days_real_L, elapsed_days_int_L, label_elapsed_days_real_L, label_elapsed_days_int_L)
            else:
                grad_cpp = torch.ops.fsrs.fsrs6_backward_verify_float(grad_out, checkpoints, params, rating_L, elapsed_days_real_L, elapsed_days_int_L, label_elapsed_days_real_L, label_elapsed_days_int_L)
            print(f"done cpp in {time.time() - cpp_start:.4f} seconds")

            print("grad cpp", grad_cpp)
            print("grad ref", grad_reference)
            torch.testing.assert_close(grad_reference, grad_cpp, atol=atol, rtol=rtol)

    def test_backward_double(self):
        self._test_backward(torch.double, 40, 200)

    # def test_backward_float(self):
    #     self._test_backward(torch.float, 100, 100, atol=1e-7, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()

