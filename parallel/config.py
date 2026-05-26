from __future__ import annotations

from pathlib import Path


LMDB_PATH = Path("parallel_db")
LMDB_SIZE = 32_569_171_968
TENSOR_CACHE_PATH = Path("/tensor-cache/lmdb")
TENSOR_CACHE_SIZE = 22_319_083_520
TENSOR_CACHE_VERSION = 2
USER_START = 1
USER_END = 5
DEVICE = "cuda"
USER_MAX_TRAIN_SPLIT_LENGTHS_KEY = "metadata_user_max_train_split_lengths"
BATCH_PERM_SEED = 2023


BATCH_SIZE = 1024
N_EPOCHS = 8
N_SPLITS = 5 
TEST_BATCH_SIZE_MAX = 1_000_000

PREPARE_USER_IDS = list(range(1, 10001))
