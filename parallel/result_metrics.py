from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
import torch
from tqdm import tqdm


@dataclass(frozen=True)
class FsrsParamSummary:
    count: int
    mean: torch.Tensor
    median: torch.Tensor
    bottom_10: torch.Tensor
    top_10: torch.Tensor
    std: torch.Tensor


def flatten_fsrs_params_for_summary(fsrs_params: torch.Tensor) -> torch.Tensor:
    return fsrs_params.detach().cpu().reshape(-1, fsrs_params.size(-1))


def summarize_fsrs_param_rows(rows: torch.Tensor) -> FsrsParamSummary:
    rows = rows.to(dtype=torch.float32)
    return FsrsParamSummary(
        count=rows.size(0),
        mean=rows.mean(dim=0),
        median=torch.quantile(rows, 0.5, dim=0),
        bottom_10=torch.quantile(rows, 0.1, dim=0),
        top_10=torch.quantile(rows, 0.9, dim=0),
        std=rows.std(dim=0, unbiased=False),
    )


def summarize_fsrs_param_parts(parts: list[torch.Tensor]) -> FsrsParamSummary:
    return summarize_fsrs_param_rows(torch.cat(parts, dim=0))


def format_param_stat(values: torch.Tensor) -> str:
    return str([round(float(value), 6) for value in values.tolist()])


def rmse_bins_score(
    p_array: np.ndarray,
    label_array: np.ndarray,
    rmse_bins_array: np.ndarray,
) -> float:
    if p_array.size == 0:
        return float("nan")
    _, inverse = np.unique(rmse_bins_array, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    p_mean = np.bincount(inverse, weights=p_array) / counts
    label_mean = np.bincount(inverse, weights=label_array) / counts
    return float(np.sqrt(np.sum(counts * np.square(label_mean - p_mean)) / np.sum(counts)))


def rounded_parameter_list(parameters: torch.Tensor) -> list[float]:
    return [round(float(value), 6) for value in parameters.tolist()]


def result_parameters(parameters: torch.Tensor) -> dict[str, list[float]]:
    if parameters.ndim == 1:
        return {"0": rounded_parameter_list(parameters)}
    return {"0": rounded_parameter_list(parameters[-1])}


def metrics_for_user(
    p: torch.Tensor,
    label: torch.Tensor,
    rmse_bins: torch.Tensor,
) -> dict[str, float | None]:
    import relplot
    from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore

    p_array = p.numpy()
    label_array = label.numpy().astype(int)
    rmse_bins_array = rmse_bins.numpy()
    p_calibrated = lowess(
        label_array,
        p_array,
        it=0,
        delta=0.01 * (float(p_array.max()) - float(p_array.min())),
        return_sorted=False,
    )
    y_hat_90 = (p_array >= 0.9).astype(int)
    try:
        auc = round(float(roc_auc_score(y_true=label_array, y_score=p_array)), 6)
    except Exception:
        auc = None
    return {
        "RMSE": round(float(root_mean_squared_error(y_true=label_array, y_pred=p_array)), 6),
        "LogLoss": round(float(log_loss(y_true=label_array, y_pred=p_array, labels=[0, 1])), 6),
        "RMSE(bins)": round(rmse_bins_score(p_array, label_array, rmse_bins_array), 6),
        "smECE": round(float(relplot.smECE(p_array, label_array)), 6),
        "AUC": auc,
        "precision@90": round(float(precision_score(label_array, y_hat_90, zero_division=0)), 6),
        "recall@90": round(float(recall_score(label_array, y_hat_90, zero_division=0)), 6),
        "ICI": round(float(np.mean(np.abs(p_calibrated - p_array))), 6),
        "MBE": round(float(np.mean(p_array - label_array)), 6),
    }


def write_user_result_jsonl(
    output_file: str | Path,
    users: list[int],
    p_by_user: dict[int, torch.Tensor],
    label_by_user: dict[int, torch.Tensor],
    rmse_bins_by_user: dict[int, torch.Tensor],
    fsrs_params_by_user: dict[int, torch.Tensor],
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for user in tqdm(sorted(users), desc="Writing to file"):
            label = label_by_user[user]
            result = {
                "metrics": metrics_for_user(
                    p_by_user[user],
                    label,
                    rmse_bins_by_user[user],
                ),
                "user": int(user),
                "size": int(label.numel()),
                "parameters": result_parameters(fsrs_params_by_user[user]),
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
