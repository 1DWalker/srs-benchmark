from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import jax
import jax.numpy as jnp
import torch


@dataclass(frozen=True)
class ParamKey:
    user_index: jax.Array
    split_index: jax.Array


@dataclass(frozen=True)
class ReviewData:
    rating: jax.Array
    elapsed_days_real: jax.Array
    seq_len: jax.Array


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
    train_index: torch.Tensor
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
        field_names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in tensors.items() if key in field_names})


def torch_tensor_to_jax_array(tensor: torch.Tensor) -> jax.Array:
    tensor = tensor.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.device.type == "cpu":
        return jnp.asarray(tensor.numpy())
    try:
        return jax.dlpack.from_dlpack(tensor, copy=False)
    except ValueError as exc:
        if "requires a copy" not in str(exc):
            raise
        return jax.dlpack.from_dlpack(tensor, copy=None)


class Data:
    def __init__(self, user_data_list: list[UserTensorBlob]) -> None:
        rating = torch.cat([user_data.rating for user_data in user_data_list], dim=-1)
        elapsed_days_real = torch.cat(
            [user_data.elapsed_days_real for user_data in user_data_list],
            dim=-1,
        )
        seq_len = torch.cat([user_data.seq_len for user_data in user_data_list], dim=-1)
        self.review_data = ReviewData(
            rating=torch_tensor_to_jax_array(rating),
            elapsed_days_real=torch_tensor_to_jax_array(elapsed_days_real),
            seq_len=torch_tensor_to_jax_array(seq_len),
        )
        user_lengths = torch.tensor(
            [user_data.rating.shape[0] for user_data in user_data_list],
            dtype=torch.int32,
        )
        user_flat_offset = torch.nn.functional.pad(
            torch.cumsum(user_lengths, dim=-1, dtype=torch.int32)[:-1],
            (1, 0),
        )
        self.user_flat_offset = torch_tensor_to_jax_array(user_flat_offset)
        # per_element_offsets = torch.repeat_interleave(
        #     self.user_flat_offset,
        #     user_lengths,
        #     output_size=self.review_data.rating.size(0),
        # )
        train_index = self.concat_with_offset(
            [user_data.train_index for user_data in user_data_list],
            user_flat_offset,
        )
        self.train_index = torch_tensor_to_jax_array(train_index)
        train_split_lengths = torch.cat(
            [user_data.train_split_lengths for user_data in user_data_list],
            dim=-1,
        )
        self.train_split_lengths = torch_tensor_to_jax_array(train_split_lengths)

        test_index = self.concat_with_offset(
            [user_data.test_index for user_data in user_data_list],
            user_flat_offset,
        )
        self.test_index = torch_tensor_to_jax_array(test_index)
        self.test_index_lens = [user_data.test_index.shape[-1] for user_data in user_data_list]
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
                torch.arange(len(split), dtype=torch.int32),
                split.to(torch.int32),
            ) 
            for split in self.splits]
        split_index = torch.cat(per_user_interleave, dim=-1)
        user_index = torch.repeat_interleave(
            torch.arange(len(self.test_index_lens), dtype=torch.int32),
            torch.tensor(self.test_index_lens, dtype=torch.int32),
        )
        return ParamKey(
            user_index=torch_tensor_to_jax_array(user_index),
            split_index=torch_tensor_to_jax_array(split_index),
        )
