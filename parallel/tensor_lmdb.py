from __future__ import annotations

from dataclasses import dataclass
import json
import warnings

import lmdb
import numpy as np
import torch


@dataclass(frozen=True)
class TensorMeta:
    dtype: str
    shape: tuple[int, ...]


_TORCH_DTYPE_TO_NAME: dict[torch.dtype, str] = {
    torch.bool: "bool",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
}

_NAME_TO_TORCH_DTYPE = {name: dtype for dtype, name in _TORCH_DTYPE_TO_NAME.items()}

_NAME_TO_NUMPY_DTYPE: dict[str, np.dtype] = {
    "bool": np.dtype(np.bool_),
    "int8": np.dtype(np.int8),
    "int16": np.dtype(np.int16),
    "int32": np.dtype(np.int32),
    "int64": np.dtype(np.int64),
    "uint8": np.dtype(np.uint8),
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}

_NUMPY_DTYPE_TO_NAME = {dtype: name for name, dtype in _NAME_TO_NUMPY_DTYPE.items()}


def tensor_meta_key(prefix: str) -> bytes:
    return f"{prefix}:meta".encode()


def tensor_data_key(prefix: str) -> bytes:
    return f"{prefix}:data".encode()


def user_tensor_prefix(user_id: int, field: str) -> str:
    return f"user:{user_id}:tensor:{field}"


def user_done_key(user_id: int) -> bytes:
    return f"user:{user_id}:done".encode()


def tensor_field_keys(prefix: str) -> list[bytes]:
    return [tensor_meta_key(prefix), tensor_data_key(prefix)]


def _encode_meta(meta: TensorMeta) -> bytes:
    return json.dumps(
        {
            "dtype": meta.dtype,
            "shape": list(meta.shape),
        },
        separators=(",", ":"),
    ).encode()


def _decode_meta(raw, prefix: str) -> TensorMeta:
    if raw is None:
        raise KeyError(f"Missing LMDB tensor metadata key: {prefix}:meta")
    raw_bytes = raw if isinstance(raw, bytes) else bytes(raw)
    payload = json.loads(raw_bytes.decode())
    return TensorMeta(
        dtype=str(payload["dtype"]),
        shape=tuple(int(dim) for dim in payload["shape"]),
    )


def _dtype_name_for_array(array: np.ndarray) -> str:
    dtype = np.dtype(array.dtype)
    try:
        return _NUMPY_DTYPE_TO_NAME[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported numpy dtype for LMDB tensor: {dtype}") from exc


def put_array(txn: lmdb.Transaction, prefix: str, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array)
    meta = TensorMeta(
        dtype=_dtype_name_for_array(array),
        shape=tuple(int(dim) for dim in array.shape),
    )
    txn.put(tensor_meta_key(prefix), _encode_meta(meta))
    txn.put(tensor_data_key(prefix), memoryview(array))


def put_tensor(txn: lmdb.Transaction, prefix: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    try:
        dtype_name = _TORCH_DTYPE_TO_NAME[tensor.dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported torch dtype for LMDB tensor: {tensor.dtype}") from exc

    array = tensor.numpy()
    meta = TensorMeta(
        dtype=dtype_name,
        shape=tuple(int(dim) for dim in tensor.shape),
    )
    txn.put(tensor_meta_key(prefix), _encode_meta(meta))
    txn.put(tensor_data_key(prefix), memoryview(array))


def get_tensor_meta(txn: lmdb.Transaction, prefix: str) -> TensorMeta:
    return _decode_meta(txn.get(tensor_meta_key(prefix)), prefix)


def get_array(txn: lmdb.Transaction, prefix: str) -> np.ndarray:
    meta = get_tensor_meta(txn, prefix)
    raw = txn.get(tensor_data_key(prefix))
    if raw is None:
        raise KeyError(f"Missing LMDB tensor data key: {prefix}:data")
    try:
        dtype = _NAME_TO_NUMPY_DTYPE[meta.dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported LMDB tensor dtype: {meta.dtype}") from exc
    return np.frombuffer(raw, dtype=dtype).reshape(meta.shape)


def get_tensor(
    txn: lmdb.Transaction,
    prefix: str,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    meta = get_tensor_meta(txn, prefix)
    raw = txn.get(tensor_data_key(prefix))
    if raw is None:
        raise KeyError(f"Missing LMDB tensor data key: {prefix}:data")
    try:
        dtype = _NAME_TO_TORCH_DTYPE[meta.dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported LMDB tensor dtype: {meta.dtype}") from exc

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable.*",
            category=UserWarning,
        )
        tensor = torch.frombuffer(raw, dtype=dtype).reshape(meta.shape)
    device = torch.device(device)
    if device.type == "cpu":
        return tensor.clone()
    return tensor.to(device)
