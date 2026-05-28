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
    chunk_count: int = 1
    chunk_elems: int | None = None


LMDB_MAX_VALUE_BYTES = 2_147_483_647


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


def tensor_chunk_key(prefix: str, chunk_i: int) -> bytes:
    return f"{prefix}:data:{chunk_i}".encode()


def user_tensor_prefix(user_id: int, field: str) -> str:
    return f"user:{user_id}:tensor:{field}"


def user_done_key(user_id: int) -> bytes:
    return f"user:{user_id}:done".encode()


def tensor_field_keys(prefix: str) -> list[bytes]:
    return [tensor_meta_key(prefix)]


def _encode_meta(meta: TensorMeta) -> bytes:
    return json.dumps(
        {
            "dtype": meta.dtype,
            "shape": list(meta.shape),
            "chunk_count": meta.chunk_count,
            "chunk_elems": meta.chunk_elems,
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
        chunk_count=int(payload.get("chunk_count", 1)),
        chunk_elems=(
            int(payload["chunk_elems"])
            if payload.get("chunk_elems") is not None
            else None
        ),
    )


def _dtype_name_for_array(array: np.ndarray) -> str:
    dtype = np.dtype(array.dtype)
    try:
        return _NUMPY_DTYPE_TO_NAME[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported numpy dtype for LMDB tensor: {dtype}") from exc


def _chunk_elems_for_dtype(dtype: np.dtype) -> int:
    itemsize = np.dtype(dtype).itemsize
    return max(1, LMDB_MAX_VALUE_BYTES // itemsize)


def _meta_for_array(array: np.ndarray, dtype_name: str | None = None) -> TensorMeta:
    dtype_name = dtype_name or _dtype_name_for_array(array)
    chunk_elems = _chunk_elems_for_dtype(array.dtype)
    chunk_count = max(1, (array.size + chunk_elems - 1) // chunk_elems)
    return TensorMeta(
        dtype=dtype_name,
        shape=tuple(int(dim) for dim in array.shape),
        chunk_count=chunk_count,
        chunk_elems=chunk_elems if chunk_count > 1 else None,
    )


def _iter_flat_chunks(array: np.ndarray, meta: TensorMeta):
    flat = array.reshape(-1)
    chunk_elems = meta.chunk_elems or flat.size
    for chunk_i in range(meta.chunk_count):
        start = chunk_i * chunk_elems
        end = min(start + chunk_elems, flat.size)
        yield chunk_i, flat[start:end]


def _delete_existing_tensor_data(txn: lmdb.Transaction, prefix: str) -> None:
    old_meta_raw = txn.get(tensor_meta_key(prefix))
    txn.delete(tensor_data_key(prefix))
    if old_meta_raw is not None:
        old_meta = _decode_meta(old_meta_raw, prefix)
        for chunk_i in range(old_meta.chunk_count):
            txn.delete(tensor_chunk_key(prefix, chunk_i))
    txn.delete(tensor_meta_key(prefix))


def _put_array_chunks(txn: lmdb.Transaction, prefix: str, array: np.ndarray, meta: TensorMeta) -> None:
    if meta.chunk_count == 1:
        txn.put(tensor_data_key(prefix), memoryview(array))
        return
    for chunk_i, chunk in _iter_flat_chunks(array, meta):
        txn.put(tensor_chunk_key(prefix, chunk_i), memoryview(chunk))


def put_array(txn: lmdb.Transaction, prefix: str, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array)
    meta = _meta_for_array(array)
    _delete_existing_tensor_data(txn, prefix)
    _put_array_chunks(txn, prefix, array, meta)
    txn.put(tensor_meta_key(prefix), _encode_meta(meta))


def put_array_to_env(
    env: lmdb.Environment,
    prefix: str,
    array: np.ndarray,
    on_chunk_write=None,
) -> None:
    array = np.ascontiguousarray(array)
    meta = _meta_for_array(array)
    if meta.chunk_count == 1:
        if on_chunk_write is not None:
            on_chunk_write(0, 1, array.nbytes)
        with env.begin(write=True) as txn:
            put_array(txn, prefix, array)
        return

    with env.begin(write=True) as txn:
        _delete_existing_tensor_data(txn, prefix)

    for chunk_i, chunk in _iter_flat_chunks(array, meta):
        if on_chunk_write is not None:
            on_chunk_write(chunk_i, meta.chunk_count, chunk.nbytes)
        with env.begin(write=True) as txn:
            txn.put(tensor_chunk_key(prefix, chunk_i), memoryview(chunk))

    with env.begin(write=True) as txn:
        txn.put(tensor_meta_key(prefix), _encode_meta(meta))


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
    meta = _meta_for_array(array, dtype_name=dtype_name)
    _delete_existing_tensor_data(txn, prefix)
    _put_array_chunks(txn, prefix, array, meta)
    txn.put(tensor_meta_key(prefix), _encode_meta(meta))


def get_tensor_meta(txn: lmdb.Transaction, prefix: str) -> TensorMeta:
    return _decode_meta(txn.get(tensor_meta_key(prefix)), prefix)


def get_array(txn: lmdb.Transaction, prefix: str) -> np.ndarray:
    meta = get_tensor_meta(txn, prefix)
    try:
        dtype = _NAME_TO_NUMPY_DTYPE[meta.dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported LMDB tensor dtype: {meta.dtype}") from exc

    if meta.chunk_count == 1:
        raw = txn.get(tensor_data_key(prefix))
        if raw is None:
            raw = txn.get(tensor_chunk_key(prefix, 0))
        if raw is None:
            raise KeyError(f"Missing LMDB tensor data key: {prefix}:data")
        return np.frombuffer(raw, dtype=dtype).reshape(meta.shape)

    out = np.empty(int(np.prod(meta.shape)), dtype=dtype)
    offset = 0
    for chunk_i in range(meta.chunk_count):
        raw = txn.get(tensor_chunk_key(prefix, chunk_i))
        if raw is None:
            raise KeyError(f"Missing LMDB tensor chunk key: {prefix}:data:{chunk_i}")
        chunk = np.frombuffer(raw, dtype=dtype)
        next_offset = offset + chunk.size
        out[offset:next_offset] = chunk
        offset = next_offset
    return out.reshape(meta.shape)


def get_tensor(
    txn: lmdb.Transaction,
    prefix: str,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    meta = get_tensor_meta(txn, prefix)
    try:
        dtype = _NAME_TO_TORCH_DTYPE[meta.dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported LMDB tensor dtype: {meta.dtype}") from exc
    device = torch.device(device)

    if meta.chunk_count == 1:
        raw = txn.get(tensor_data_key(prefix))
        if raw is None:
            raw = txn.get(tensor_chunk_key(prefix, 0))
        if raw is None:
            raise KeyError(f"Missing LMDB tensor data key: {prefix}:data")
        tensor = _tensor_from_buffer(raw, dtype).reshape(meta.shape)
        if device.type == "cpu":
            return tensor.clone()
        return tensor.to(device)

    out = torch.empty(int(np.prod(meta.shape)), dtype=dtype, device=device)
    offset = 0
    for chunk_i in range(meta.chunk_count):
        raw = txn.get(tensor_chunk_key(prefix, chunk_i))
        if raw is None:
            raise KeyError(f"Missing LMDB tensor chunk key: {prefix}:data:{chunk_i}")
        chunk = _tensor_from_buffer(raw, dtype)
        next_offset = offset + chunk.numel()
        out[offset:next_offset].copy_(chunk)
        offset = next_offset
    return out.reshape(meta.shape)


def _tensor_from_buffer(raw, dtype: torch.dtype) -> torch.Tensor:
    if len(raw) == 0:
        return torch.empty(0, dtype=dtype)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable.*",
            category=UserWarning,
        )
        return torch.frombuffer(raw, dtype=dtype)
