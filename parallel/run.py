from __future__ import annotations

from io import BytesIO

import lmdb
import torch
from tqdm import tqdm

from parallel.config import (
    LMDB_PATH,
    LMDB_SIZE,
    DEVICE,
)
from parallel.tensors import ConcatTensors, UserTensorBlob

def load_blob_bytes(blob_bytes: bytes) -> UserTensorBlob:
    buffer = BytesIO(blob_bytes)
    tensors = torch.load(buffer, weights_only=True, map_location=DEVICE)
    return UserTensorBlob.from_dict(tensors)

def load_metadata_tensor(txn: lmdb.Transaction, key: str) -> torch.Tensor:
    tensor_bytes = txn.get(key.encode())
    if tensor_bytes is None:
        raise KeyError(f"Missing LMDB metadata key: {key}")
    return torch.load(
        BytesIO(tensor_bytes),
        weights_only=True,
        map_location="cpu",
    )


def load_user_blob(txn: lmdb.Transaction, user_id: int) -> UserTensorBlob:
    blob_bytes = txn.get(f"{user_id}_packed".encode())
    if blob_bytes is None:
        raise LookupError(f"Packed blob not found for user {user_id}.")
    return load_blob_bytes(blob_bytes)


def run(
    env: lmdb.Environment,
    users: list[int],
) -> None:
    with env.begin(write=False) as txn:
        blobs = [
            load_user_blob(txn, user_id)
            for user_id in tqdm(users, total=len(users), smoothing=0.03, desc="Loading user data")
        ]

    print(blobs)

    data = ConcatTensors(blobs)
    print(data)

    # evaluate
    with torch.no_grad():
        pass

def main() -> None:
    env = lmdb.open(
        str(LMDB_PATH),
        map_size=LMDB_SIZE,
        readonly=True,
        lock=False,
    )

    # run(env, list(range(1, 1001)))
    run(env, [3, 4, 5])

    # with env.begin(write=False) as txn:
    #     user_max_train_split_lengths = load_metadata_tensor(
    #         txn,
    #         USER_MAX_TRAIN_SPLIT_LENGTHS_KEY,
    #     )
    # env.close()

    # values, indices = torch.sort(user_max_train_split_lengths, descending=True)

    # for index, value in zip(indices.tolist(), values.tolist()):
    #     user_id = USER_IDS[index] if index < len(USER_IDS) else index
    #     print(f"{index}\t{user_id}\t{value}")


if __name__ == "__main__":
    main()
