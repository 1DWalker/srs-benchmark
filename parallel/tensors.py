from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch


@dataclass(frozen=True)
class ParamKey:
    user_index: torch.Tensor
    split_index: torch.Tensor


@dataclass(frozen=True)
class ReviewData:
    rating: torch.Tensor
    elapsed_days_real: torch.Tensor
    seq_len: torch.Tensor


@dataclass(frozen=True)
class UserTensorBlob:
    rating: torch.Tensor
    elapsed_days_int: torch.Tensor
    elapsed_days_real: torch.Tensor
    card_sorted_index: torch.Tensor
    seq_len: torch.Tensor
    card_last_index: torch.Tensor

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

    def pretty(self) -> str:
        lines = ["("]
        total_bytes = 0
        for field in fields(self):
            tensor = getattr(self, field.name)
            total_bytes += tensor.numel() * tensor.element_size()
            shape = "[]" if tensor.dim() == 0 else list(tensor.shape)
            lines.append(
                f"  {field.name}: dtype={tensor.dtype}, shape={shape}, "
                f"numel={tensor.numel():,}, "
                f"{tensor}",
            )
        lines.append(f"  total_bytes={total_bytes:,}")
        lines.append(")")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.pretty()

    def __str__(self) -> str:
        return self.pretty()

    @classmethod
    def from_dict(cls, tensors: dict[str, torch.Tensor]) -> UserTensorBlob:
        return cls(**tensors)


class Data:
    def __init__(self, user_data_list: list[UserTensorBlob]) -> None:
        self.review_data = ReviewData(
            rating=torch.concat([user_data.rating for user_data in user_data_list], dim=-1),
            elapsed_days_real=torch.concat([user_data.elapsed_days_real for user_data in user_data_list], dim=-1),
            seq_len=torch.concat([user_data.seq_len for user_data in user_data_list], dim=-1),
        )
        self.device = self.review_data.rating.device
        user_lengths = torch.tensor(
            [user_data.rating.size(0) for user_data in user_data_list],
            device=self.review_data.rating.device,
        )
        self.user_flat_offset = torch.nn.functional.pad(
            torch.cumsum(user_lengths, dim=-1)[:-1],
            (1, 0),
        )
        per_element_offsets = torch.repeat_interleave(
            self.user_flat_offset,
            user_lengths,
            output_size=self.review_data.rating.size(0),
        )
        self.train_index = self.concat_with_offset(
            [user_data.train_index for user_data in user_data_list],
            self.user_flat_offset,
        )

        self.test_index = self.concat_with_offset(
            [user_data.test_index for user_data in user_data_list],
            self.user_flat_offset,
        )
        self.test_index_lens = [user_data.test_index.size(-1) for user_data in user_data_list]
        self.splits = [user_data.split for user_data in user_data_list]

    @staticmethod
    def concat_with_offset(xs: list[torch.Tensor], offsets: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [x + offsets[i] for i, x in enumerate(xs)],
            dim=-1,
        )

    def get_test_index_param_key(self) -> ParamKey:
        # Delay the computation of this to save a bit of memory
        per_user_interleave = [
            torch.repeat_interleave(
                torch.arange(len(split), device=split.device),
                split.to(torch.long),
            ) 
            for split in self.splits]
        split_index = torch.cat(per_user_interleave, dim=-1)
        user_index = torch.repeat_interleave(torch.arange(len(self.test_index_lens), device=self.device), torch.tensor(self.test_index_lens, device=self.device, dtype=torch.long))
        return ParamKey(user_index=user_index, split_index=split_index)
