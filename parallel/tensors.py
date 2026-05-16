from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class UserTensorBlob:
    ratings: torch.Tensor
    elapsed_days_int: torch.Tensor
    elapsed_days_real: torch.Tensor

    # Test
    test_index: torch.Tensor
    rmse_bins: torch.Tensor
    split: torch.Tensor

    # Train
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

class ConcatTensors:
    def __init__(self, user_data_list: list[UserTensorBlob]) -> None:
        self.ratings = torch.concat([user_data.ratings for user_data in user_data_list], dim=-1)
        self.elapsed_days_real = torch.concat([user_data.elapsed_days_real for user_data in user_data_list], dim=-1)
        user_lengths = torch.tensor(
            [user_data.ratings.size(0) for user_data in user_data_list],
            device=self.ratings.device,
        )
        self.user_flat_offset = torch.nn.functional.pad(
            torch.cumsum(user_lengths, dim=-1)[:-1],
            (1, 0),
        )
        print(self.ratings.size())
        print(self.user_flat_offset)

        # self.test_index: torch.Tensor
        # self.split: torch.Tensor
        # self.batch_order: torch.Tensor
        # self.batch_order_epochs: torch.Tensor
        # self.train_index: torch.Tensor
        # self.train_batch_lengths: torch.Tensor
        # self.train_split_lengths: torch.Tensor
