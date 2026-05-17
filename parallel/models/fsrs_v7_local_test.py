"""Temporary local scaffolding for comparing fsrs_v7.py to fsrs_7_orig.py.

Delete this file once the batched-parameter migration is validated.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch


ROOT = Path(__file__).resolve().parent


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


orig = load_module("fsrs_7_orig_for_local_tests", ROOT / "fsrs_7_orig.py")
new = load_module("fsrs_v7_for_local_tests", ROOT / "fsrs_v7.py")


def random_feature_batch(batch_size: int, seq_len: int, generator: torch.Generator):
    feature_elapsed_days_real_bl = torch.rand(
        (batch_size, seq_len),
        generator=generator,
    ) * 120.0
    feature_elapsed_days_real_bl[:, 0] = 0.0
    feature_rating_bl = torch.randint(
        1,
        5,
        (batch_size, seq_len),
        generator=generator,
    )
    label_elapsed_days_real_bl = torch.rand(
        (batch_size, seq_len),
        generator=generator,
    ) * 120.0
    return feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_real_bl


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def assert_grad_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=5e-6, atol=5e-4)


def test_param_transform_matches_original() -> None:
    sampler = torch.distributions.studentT.StudentT(torch.full((35,), 1.0))

    for _ in range(100):
        raw_p = sampler.sample()
        expected_p = orig.nn_vec_to_fsrs7_params(raw_p)
        actual_p = new.nn_vec_to_fsrs7_params(raw_p)
        torch.testing.assert_close(actual_p, expected_p, rtol=0.0, atol=0.0)


def test_batched_param_transform_matches_original_rowwise() -> None:
    sampler = torch.distributions.studentT.StudentT(torch.full((35,), 1.0))

    for batch_size in [1, 2, 3, 8, 17]:
        raw_bp = sampler.sample((batch_size,))
        expected_bp = torch.stack(
            [orig.nn_vec_to_fsrs7_params(raw_bp[b]) for b in range(batch_size)],
            dim=0,
        )
        actual_bp = new.nn_vec_to_fsrs7_params(raw_bp)
        assert_close(actual_bp, expected_bp)


def test_shared_parameters_match_original_forward() -> None:
    generator = torch.Generator().manual_seed(20260517)
    sampler = torch.distributions.studentT.StudentT(torch.full((35,), 1.0))

    for _ in range(50):
        batch_size = int(torch.randint(1, 9, (), generator=generator).item())
        seq_len = int(torch.randint(1, 12, (), generator=generator).item())
        parameters_p = orig.nn_vec_to_fsrs7_params(sampler.sample())
        (
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            label_elapsed_days_real_bl,
        ) = random_feature_batch(batch_size, seq_len, generator)

        expected_retention_bl, expected_stability_bl = orig.forward(
            parameters_p,
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            label_elapsed_days_real_bl,
        )
        actual_retention_bl, actual_stability_bl = new.forward(
            parameters_p.expand(batch_size, -1),
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            label_elapsed_days_real_bl,
        )

        assert_close(actual_retention_bl, expected_retention_bl)
        assert_close(actual_stability_bl, expected_stability_bl)


def test_batched_parameters_match_original_rowwise_forward() -> None:
    generator = torch.Generator().manual_seed(20260518)
    sampler = torch.distributions.studentT.StudentT(torch.full((35,), 1.0))

    for _ in range(50):
        batch_size = int(torch.randint(1, 9, (), generator=generator).item())
        seq_len = int(torch.randint(1, 12, (), generator=generator).item())
        raw_bp = sampler.sample((batch_size,))
        parameters_bp = new.nn_vec_to_fsrs7_params(raw_bp)
        (
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            label_elapsed_days_real_bl,
        ) = random_feature_batch(batch_size, seq_len, generator)

        actual_retention_bl, actual_stability_bl = new.forward(
            parameters_bp,
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            label_elapsed_days_real_bl,
        )

        for b in range(batch_size):
            expected_retention_bl, expected_stability_bl = orig.forward(
                parameters_bp[b],
                feature_elapsed_days_real_bl[b : b + 1],
                feature_rating_bl[b : b + 1],
                label_elapsed_days_real_bl[b : b + 1],
            )
            assert_close(actual_retention_bl[b : b + 1], expected_retention_bl)
            assert_close(actual_stability_bl[b : b + 1], expected_stability_bl)


def test_batched_parameter_gradients_match_original_rowwise() -> None:
    generator = torch.Generator().manual_seed(20260519)
    sampler = torch.distributions.studentT.StudentT(torch.full((35,), 1.0))

    for _ in range(10):
        batch_size = int(torch.randint(1, 5, (), generator=generator).item())
        seq_len = int(torch.randint(2, 8, (), generator=generator).item())
        raw_bp = sampler.sample((batch_size,)).detach().requires_grad_(True)
        (
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            label_elapsed_days_real_bl,
        ) = random_feature_batch(batch_size, seq_len, generator)

        parameters_bp = new.nn_vec_to_fsrs7_params(raw_bp)
        actual_retention_bl, actual_stability_bl = new.forward(
            parameters_bp,
            feature_elapsed_days_real_bl,
            feature_rating_bl,
            label_elapsed_days_real_bl,
        )
        actual_loss = actual_retention_bl.sum() + 0.01 * actual_stability_bl.sum()
        actual_loss.backward()
        actual_grad_bp = raw_bp.grad.detach().clone()

        expected_grad_rows = []
        for b in range(batch_size):
            raw_p = raw_bp.detach()[b].clone().requires_grad_(True)
            parameters_p = orig.nn_vec_to_fsrs7_params(raw_p)
            expected_retention_bl, expected_stability_bl = orig.forward(
                parameters_p,
                feature_elapsed_days_real_bl[b : b + 1],
                feature_rating_bl[b : b + 1],
                label_elapsed_days_real_bl[b : b + 1],
            )
            expected_loss = expected_retention_bl.sum() + 0.01 * expected_stability_bl.sum()
            expected_loss.backward()
            expected_grad_rows.append(raw_p.grad.detach())

        expected_grad_bp = torch.stack(expected_grad_rows, dim=0)
        assert_grad_close(actual_grad_bp, expected_grad_bp)


def main() -> None:
    torch.manual_seed(20260517)
    test_param_transform_matches_original()
    test_batched_param_transform_matches_original_rowwise()
    test_shared_parameters_match_original_forward()
    test_batched_parameters_match_original_rowwise_forward()
    test_batched_parameter_gradients_match_original_rowwise()
    print("Local FSRS-7 comparison tests passed.")


if __name__ == "__main__":
    main()
