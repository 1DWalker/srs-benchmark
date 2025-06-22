"""Check that the python and c++ implementations of FSRS-6 match."""

import random
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


    # def test_forward(self):
    #     reference_fsrs = FSRS6()
    #     random.seed(123)
    #     np.random.seed(123)
    #     torch.manual_seed(123)
    #     for _ in range(100):
    #         L = random.randint(1, 50)
    #         rating_L = torch.tensor(np.random.randint(1, 5, size=L))
    #         elapsed_days_real_L = torch.tensor(np.random.uniform(0.1, 400.0, size=L), dtype=torch.float)
    #         elapsed_days_int_L = torch.tensor(np.random.randint(0, 400, size=L), dtype=torch.float)
    #         label_elapsed_days_real_L = torch.tensor(np.random.uniform(0.1, 400.0, size=L), dtype=torch.float)
    #         label_elapsed_days_int_L = torch.tensor(np.random.randint(0, 400, size=L), dtype=torch.float)
    #         cpp_out, _ = torch.ops.fsrs.fsrs6_forward_verify(self.params, rating_L, elapsed_days_real_L, elapsed_days_int_L, label_elapsed_days_real_L, label_elapsed_days_int_L)
    #         reference_out = reference_fsrs.forward(self.params, elapsed_days_real_L.unsqueeze(0), elapsed_days_int_L.unsqueeze(0), rating_L.unsqueeze(0), label_elapsed_days_real_L.unsqueeze(0), label_elapsed_days_int_L.unsqueeze(0))
    #         torch.testing.assert_close(cpp_out, reference_out.squeeze(0))

    def test_backward(self):
        reference_fsrs = FSRS6()
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        for _ in range(5):
            # L = random.randint(1, 100)
            L = 1
            rating_L = torch.tensor(np.random.randint(1, 5, size=L))
            elapsed_days_real_L = torch.tensor(np.random.uniform(0.1, 400.0, size=L), dtype=torch.float)
            elapsed_days_int_L = torch.tensor(np.random.randint(0, 400, size=L), dtype=torch.float)
            label_elapsed_days_real_L = torch.tensor(np.random.uniform(0.1, 400.0, size=L), dtype=torch.float)
            label_elapsed_days_int_L = torch.tensor(np.random.randint(0, 400, size=L), dtype=torch.float)
            params = self.params.clone().detach().requires_grad_(True)
            _, checkpoints = torch.ops.fsrs.fsrs6_forward_verify(params, rating_L, elapsed_days_real_L, elapsed_days_int_L, label_elapsed_days_real_L, label_elapsed_days_int_L)
            reference_out = reference_fsrs.forward(params, elapsed_days_real_L.unsqueeze(0), elapsed_days_int_L.unsqueeze(0), rating_L.unsqueeze(0), label_elapsed_days_real_L.unsqueeze(0), label_elapsed_days_int_L.unsqueeze(0))

            grad_out = torch.randn_like(reference_out)
            grad_reference = torch.autograd.grad(reference_out, params, grad_out)[0]

            grad_cpp = torch.ops.fsrs.fsrs6_backward_verify(grad_out, checkpoints, params, rating_L, elapsed_days_real_L, elapsed_days_int_L, label_elapsed_days_real_L, label_elapsed_days_int_L)
            print("grad cpp", grad_cpp)
            print("grad ref", grad_reference)
            torch.testing.assert_close(grad_reference, grad_cpp)



if __name__ == '__main__':
    unittest.main()

