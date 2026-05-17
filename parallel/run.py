from __future__ import annotations

from io import BytesIO

import lmdb
import torch
from tqdm import tqdm

from parallel.config import (
    LMDB_PATH,
    LMDB_SIZE,
    DEVICE,
    N_SPLITS,
    TEST_BATCH_SIZE_MAX,
)
from parallel.tensors import ConcatTensors, ParamKey, ReviewData, UserTensorBlob

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

def predict(review_data: ReviewData, index, params):
    assert index.size(0) == params.size(0)
    seq_lens = review_data.seq_len[index]
    max_seq_len = seq_lens.max()
    print(seq_lens)
    print(max_seq_len)
    pass
    

def evaluate_on_test_set(fsrs_params: torch.Tensor, data: ConcatTensors):
    print("eval on test")
    param_keys = data.get_test_index_param_key()

    test_seq_len = data.review_data.seq_len[data.test_index]
    sorted_test_index_permutation = torch.argsort(test_seq_len)
    # sorted_test_index = data.test_index[sorted_test_index_permutation]
    # sorted_param_key = ParamKey(
    #     user_index=unsorted_param_key.user_index[sorted_test_index_permutation],
    #     split_index=unsorted_param_key.split_index[sorted_test_index_permutation],
    # )
    restore_test_index_permutation = torch.empty_like(sorted_test_index_permutation)
    restore_test_index_permutation[sorted_test_index_permutation] = torch.arange(
        sorted_test_index_permutation.numel(),
        device=sorted_test_index_permutation.device,
    )
    print("sort done")

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    N = data.test_index.size(0)
    num_batches = ceil_div(N, TEST_BATCH_SIZE_MAX)
    batch_size = ceil_div(N, num_batches)
    for perm_slice in sorted_test_index_permutation.split(batch_size):
        batch = test_seq_len[perm_slice]
        batch_fsrs_params = fsrs_params[param_keys.user_index[batch], param_keys.split_index[batch]]
        p = predict(data.review_data, batch, batch_fsrs_params)
    

def run(
    env: lmdb.Environment,
    users: list[int],
) -> None:
    with env.begin(write=False) as txn:
        blobs = [
            load_user_blob(txn, user_id)
            for user_id in tqdm(users, total=len(users), smoothing=0.03, desc="Loading user data")
        ]

    # TODO sort users by train split length
    # TODO train

    print(blobs)

    fsrs_params = torch.zeros((len(users), N_SPLITS, 35), device=DEVICE)
    data = ConcatTensors(blobs)
    print(data)

    # evaluate
    with torch.no_grad():
        evaluate_on_test_set(fsrs_params, data)

def main() -> None:
    env = lmdb.open(
        str(LMDB_PATH),
        map_size=LMDB_SIZE,
        readonly=True,
        lock=False,
    )

    # run(env, list(range(1, 1001)))
    run(env, [1, 2])

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
