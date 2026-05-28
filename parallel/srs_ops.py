from __future__ import annotations

import importlib

import torch


fsrs_extension = importlib.import_module("parallel._fsrs_extension")
THREADS_PER_BLOCK = int(fsrs_extension.threads_per_block())


def _fake_srs_fsrs7_train(
    elapsed_days_real_flat,
    rating_flat,
    start_index_UxT,
    grad_weight_UxT,
    seq_len_UxT,
    seq_len_Ux_max,
    seq_len_Ux_max_cumsum,
    fsrs_params_UP,
):
    return fsrs_params_UP.new_empty(
        (
            seq_len_UxT.shape[0],
            seq_len_UxT.shape[1] * seq_len_UxT.shape[2],
            fsrs_params_UP.shape[1],
        )
    )


try:
    torch.library.register_fake("srs::fsrs7_train")(_fake_srs_fsrs7_train)
except RuntimeError as exc:
    if "already" not in str(exc):
        raise
