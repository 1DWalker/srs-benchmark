from __future__ import annotations

from io import BytesIO

import lmdb
from sklearn.metrics import log_loss
import torch
from tqdm import tqdm

from parallel.config import (
    LMDB_PATH,
    LMDB_SIZE,
    DEVICE,
    N_SPLITS,
    TEST_BATCH_SIZE_MAX,
)
from parallel.models import fsrs_v7
from parallel.tensors import Data, ParamKey, ReviewData, UserTensorBlob

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
    card_start_index = index - seq_lens + 1
    data_len = review_data.elapsed_days_real.size(0)
    take_indices = (card_start_index.unsqueeze(-1) + torch.arange(max_seq_len.item(), device=index.device).repeat(index.size(0), 1)).clamp_max(data_len - 1)

    # TODO possible bug from overlap, if it hits 0 elapsed time
    rating_bl = review_data.rating[take_indices]
    elapsed_time_real_bl = review_data.elapsed_days_real[take_indices]

    # fsrs_params = fsrs_v7.nn_vec_to_fsrs7_params(params)
    fsrs_params = fsrs_v7.FSRS7_DEFAULT_35.to(params.device).unsqueeze(0).expand(params.size(0), -1)
    return fsrs_v7.forward(fsrs_params, elapsed_time_real_bl, rating_bl, seq_lens)
    

def evaluate_on_test_set(fsrs_params: torch.Tensor, users: list[int], data: Data):
    print("eval on test")
    param_keys = data.get_test_index_param_key()

    test_seq_len = data.review_data.seq_len[data.test_index]
    sorted_test_index_permutation = torch.argsort(test_seq_len, stable=True)
    print("sort done")

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    # TODO use the same load balancing as training
    N = data.test_index.size(0)
    num_batches = ceil_div(N, TEST_BATCH_SIZE_MAX)
    batch_size = ceil_div(N, num_batches)
    gather_p = []
    for perm_slice in sorted_test_index_permutation.split(batch_size):
        batch_fsrs_params = fsrs_params[param_keys.user_index[perm_slice], param_keys.split_index[perm_slice]]
        p = predict(data.review_data, data.test_index[perm_slice], batch_fsrs_params)
        gather_p.append(p)
    
    # TODO reduce memory by removing the inverse perm tensor
    restore_test_index_permutation = torch.empty_like(sorted_test_index_permutation)
    restore_test_index_permutation[sorted_test_index_permutation] = torch.arange(
        sorted_test_index_permutation.numel(),
        device=sorted_test_index_permutation.device,
    )
    gather_p = torch.cat(gather_p, dim=-1)[restore_test_index_permutation]

    p_by_user = gather_p.split(data.test_index_lens)
    label_by_user = (data.review_data.rating[data.test_index] > 1).split(data.test_index_lens)

    for user, pred, label in tqdm(zip(users, p_by_user, label_by_user), smoothing=0.03):
        logloss = log_loss(y_true=label.cpu().numpy(), y_pred=pred.cpu().numpy(), labels=[0, 1])
        print(f"User: {user}, logloss={logloss:.3f}")

def run(
    env: lmdb.Environment,
    users: list[int],
) -> None:
    with env.begin(write=False) as txn:
        blobs = [
            load_user_blob(txn, user_id)
            for user_id in tqdm(users, total=len(users), smoothing=0.03, desc="Loading user data")
        ]
        data = Data(blobs)
        del blobs

    # TODO sort users by train split length
    # TODO train


    fsrs_params = torch.zeros((len(users), N_SPLITS, 35), device=DEVICE)
    print(data)

    # evaluate
    with torch.no_grad():
        evaluate_on_test_set(fsrs_params, users, data)

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
