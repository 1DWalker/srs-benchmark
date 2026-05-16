from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class UserTensorBlob:
    ratings: torch.Tensor
    elapsed_days_int: torch.Tensor
    elapsed_days_real: torch.Tensor
    test_index: torch.Tensor
    rmse_bins: torch.Tensor
    split: torch.Tensor
    batch_order: torch.Tensor
    batch_order_epochs: torch.Tensor
    train_index: torch.Tensor
    train_batch_lengths: torch.Tensor
    train_split_lengths: torch.Tensor

    def to_dict(self) -> dict[str, torch.Tensor]:
        return asdict(self)

    @classmethod
    def from_dict(cls, tensors: dict[str, torch.Tensor]) -> UserTensorBlob:
        return cls(**tensors)
