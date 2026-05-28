from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from collections.abc import Iterable

import numpy as np
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
    train_index: torch.Tensor
    split_review_ord: torch.Tensor
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


def _device_matches(tensor: torch.Tensor, device: torch.device) -> bool:
    if tensor.device.type != device.type:
        return False
    return device.index is None or tensor.device.index == device.index


def _device_specs_match(left: torch.device, right: torch.device) -> bool:
    if left.type != right.type:
        return False
    return left.index is None or right.index is None or left.index == right.index


_TORCH_TO_NUMPY_DTYPE: dict[torch.dtype, np.dtype] = {
    torch.bool: np.dtype(np.bool_),
    torch.int8: np.dtype(np.int8),
    torch.int16: np.dtype(np.int16),
    torch.int32: np.dtype(np.int32),
    torch.int64: np.dtype(np.int64),
    torch.uint8: np.dtype(np.uint8),
    torch.float16: np.dtype(np.float16),
    torch.float32: np.dtype(np.float32),
    torch.float64: np.dtype(np.float64),
}


def _numpy_dtype_for(torch_dtype: torch.dtype) -> np.dtype:
    try:
        return _TORCH_TO_NUMPY_DTYPE[torch_dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported tensor dtype for DataBuilder: {torch_dtype}") from exc


class _TensorVector:
    def __init__(
        self,
        dtype: torch.dtype | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        device = torch.device(device)
        if device.type != "cpu":
            raise ValueError("DataBuilder only supports CPU-side construction.")
        self.dtype = dtype
        self.device = torch.device("cpu")
        self._buffer: np.ndarray | None = None
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def append(
        self,
        tensor: torch.Tensor,
        dtype: torch.dtype | None = None,
        offset: int = 0,
    ) -> None:
        target_dtype = dtype if dtype is not None else self.dtype
        if tensor.device.type != "cpu":
            raise ValueError("DataBuilder only accepts CPU tensors.")

        tensor = tensor.detach().reshape(-1)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        if target_dtype is None:
            target_dtype = tensor.dtype

        target_np_dtype = _numpy_dtype_for(target_dtype)
        array = tensor.numpy()
        if array.dtype != target_np_dtype:
            array = array.astype(target_np_dtype, copy=False)

        if self.dtype is None:
            self.dtype = target_dtype
        elif target_dtype != self.dtype:
            target_dtype = self.dtype
            target_np_dtype = _numpy_dtype_for(target_dtype)
            if array.dtype != target_np_dtype:
                array = array.astype(target_np_dtype, copy=False)

        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)

        required = self._size + array.size
        if self._buffer is None:
            capacity = max(1, required)
            self._buffer = np.empty(capacity, dtype=target_np_dtype)
        elif required > self._buffer.size:
            capacity = max(required, int(self._buffer.size * 1.2))
            new_buffer = np.empty(capacity, dtype=self._buffer.dtype)
            new_buffer[: self._size] = self._buffer[: self._size]
            self._buffer = new_buffer

        if array.size > 0:
            target = self._buffer[self._size : required]
            if offset:
                np.add(array, offset, out=target, casting="unsafe")
            else:
                target[...] = array
        self._size = required

    def finish(self, dtype: torch.dtype | None = None, shrink: bool = True) -> torch.Tensor:
        if self._buffer is None:
            return torch.empty(0, dtype=dtype or self.dtype or torch.float32, device=self.device)
        result_array = self._buffer[: self._size]
        if shrink and result_array.size != self._buffer.size:
            result_array = result_array.copy()

        self._buffer = None
        self._size = 0
        result = torch.from_numpy(result_array)
        if dtype is not None and result.dtype != dtype:
            result = result.to(dtype=dtype)
        return result


class DataBuilder:
    def __init__(self, device: torch.device | str = "cpu") -> None:
        device = torch.device(device)
        if device.type != "cpu":
            raise ValueError("DataBuilder only supports CPU-side construction.")
        self.device = torch.device("cpu")
        self.rating = _TensorVector(device=self.device)
        self.elapsed_days_real = _TensorVector(device=self.device)
        self.seq_len = _TensorVector(device=self.device)
        self.train_index = _TensorVector(torch.int32, device=self.device)
        self.split_review_ord = _TensorVector(torch.int32, device=self.device)
        self.test_index = _TensorVector(torch.int32, device=self.device)
        self.rmse_bins = _TensorVector(torch.int32, device=self.device)
        self.train_split_lengths = _TensorVector(torch.int32, device=self.device)
        self.splits = _TensorVector(torch.int32, device=self.device)
        self.split_counts = _TensorVector(torch.int32, device=self.device)

        self.user_lengths: list[int] = []
        self.test_index_lens: list[int] = []
        self.first_device: torch.device | None = None
        self.review_offset = 0

    def append(self, user_data: UserTensorBlob) -> None:
        if self.first_device is None:
            self.first_device = user_data.rating.device

        self.rating.append(user_data.rating)
        assert user_data.rating.dtype == torch.int8
        self.elapsed_days_real.append(user_data.elapsed_days_real)
        self.seq_len.append(user_data.seq_len)

        self.train_index.append(
            user_data.train_index,
            offset=self.review_offset,
        )
        self.split_review_ord.append(user_data.split_review_ord)
        self.test_index.append(
            user_data.test_index,
            offset=self.review_offset,
        )
        self.rmse_bins.append(user_data.rmse_bins)

        self.train_split_lengths.append(user_data.train_split_lengths)
        self.test_index_lens.append(user_data.test_index.numel())
        self.splits.append(user_data.split)
        self.split_counts.append(
            torch.tensor([user_data.split.numel()], dtype=torch.int32),
        )

        user_length = user_data.rating.size(0)
        self.user_lengths.append(user_length)
        self.review_offset += user_length

    def finish(self, device: torch.device | str | None = None) -> Data:
        target_device = torch.device(device) if device is not None else self.first_device
        will_transfer = target_device is not None and not _device_specs_match(
            self.device,
            target_device,
        )
        shrink = not will_transfer

        data = object.__new__(Data)
        data.review_data = ReviewData(
            rating=self.rating.finish(shrink=shrink),
            elapsed_days_real=self.elapsed_days_real.finish(shrink=shrink),
            seq_len=self.seq_len.finish(shrink=shrink),
        )
        data.device = data.review_data.rating.device

        user_lengths_t = torch.tensor(self.user_lengths, dtype=torch.int32, device=self.device)
        if user_lengths_t.numel() == 0:
            data.user_flat_offset = torch.empty(0, dtype=torch.int32, device=self.device)
        else:
            data.user_flat_offset = torch.nn.functional.pad(
                torch.cumsum(user_lengths_t, dim=-1, dtype=torch.int32)[:-1],
                (1, 0),
            )

        data.train_index = self.train_index.finish(shrink=shrink)
        data.split_review_ord = self.split_review_ord.finish(torch.int32, shrink=shrink)
        data.train_split_lengths = self.train_split_lengths.finish(torch.int32, shrink=shrink)

        data.test_index = self.test_index.finish(shrink=shrink)
        data.rmse_bins = self.rmse_bins.finish(torch.int32, shrink=shrink)
        data.test_index_lens = torch.tensor(
            self.test_index_lens,
            dtype=torch.int32,
            device=self.device,
        )
        data.splits = self.splits.finish(torch.int32, shrink=shrink)
        data.split_counts = self.split_counts.finish(shrink=shrink)

        data._assert_valid()

        if target_device is not None and not _device_matches(data.review_data.rating, target_device):
            data.to_(target_device)
        return data


class Data:
    def __init__(
        self,
        user_data_list: Iterable[UserTensorBlob],
        device: torch.device | str | None = None,
        build_device: torch.device | str = "cpu",
    ) -> None:
        builder = DataBuilder(device=build_device)
        for user_data in user_data_list:
            builder.append(user_data)
        self.__dict__.update(builder.finish(device=device).__dict__)

    def _assert_valid(self) -> None:
        assert (self.review_data.elapsed_days_real[self.train_index] > 0).all()
        assert (self.review_data.elapsed_days_real[self.test_index] > 0).all()

    def to_(self, device: torch.device | str) -> Data:
        device = torch.device(device)
        self.review_data = ReviewData(
            rating=self.review_data.rating.to(device),
            elapsed_days_real=self.review_data.elapsed_days_real.to(device),
            seq_len=self.review_data.seq_len.to(device),
        )
        self.device = self.review_data.rating.device
        self.user_flat_offset = self.user_flat_offset.to(device)
        self.train_index = self.train_index.to(device)
        self.split_review_ord = self.split_review_ord.to(device)
        self.train_split_lengths = self.train_split_lengths.to(device)
        self.test_index = self.test_index.to(device)
        self.rmse_bins = self.rmse_bins.to(device)
        self.test_index_lens = self.test_index_lens.to(device)
        self.splits = self.splits.to(device)
        self.split_counts = self.split_counts.to(device)
        return self


    @staticmethod
    def concat_with_offset(xs: list[torch.Tensor], offsets: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [x + offsets[i] for i, x in enumerate(xs)],
            dim=-1,
        )

    def get_test_index_param_key(self) -> ParamKey:
        # Delay the computation of this to save a bit of memory
        split_counts = self.split_counts
        split_offsets = torch.nn.functional.pad(
            torch.cumsum(split_counts, dim=-1, dtype=torch.int32)[:-1],
            (1, 0),
        )
        split_owner_offsets = torch.repeat_interleave(split_offsets, split_counts)
        split_index_per_split = torch.arange(
            self.splits.numel(),
            device=self.device,
            dtype=torch.int32,
        ) - split_owner_offsets
        split_index = torch.repeat_interleave(
            split_index_per_split,
            self.splits,
        )
        user_index = torch.repeat_interleave(
            torch.arange(self.test_index_lens.size(0), device=self.device, dtype=torch.int32),
            self.test_index_lens,
        )
        return ParamKey(user_index=user_index, split_index=split_index)
