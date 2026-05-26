from __future__ import annotations

from pathlib import Path


LMDB_PATH = Path("parallel_db")
LMDB_SIZE = 50_000_000_000
TENSOR_CACHE_PATH = Path("/tensor-cache/lmdb")
TENSOR_CACHE_SIZE = 100_000_000_000
TENSOR_CACHE_VERSION = 1
USER_IDS = list(range(1, 10001))
DEVICE = "cuda"
USER_MAX_TRAIN_SPLIT_LENGTHS_KEY = "metadata_user_max_train_split_lengths"
BATCH_PERM_SEED = 2023


BATCH_SIZE = 1024
N_EPOCHS = 8
N_SPLITS = 5 
TRAIN_BUFFER_SIZE_GB = 0.5
TEST_BATCH_SIZE_MAX = 100_000
