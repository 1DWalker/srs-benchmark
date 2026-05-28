from __future__ import annotations

from pathlib import Path

DEVICE = "cuda"  # not optional

LMDB_PATH = Path("parallel_db")
LMDB_SIZE = 32_569_171_968
TENSOR_CACHE_PATH = Path("/tensor-cache/lmdb")
TENSOR_CACHE_SIZE = 26_319_083_520
TENSOR_CACHE_VERSION = 4
USER_MAX_TRAIN_SPLIT_LENGTHS_KEY = "metadata_user_max_train_split_lengths"
TEST_BATCH_SIZE_MAX = 10_000_000

BATCH_PERM_SEED = 1234

# Invalidates the cache
USER_START = 1
USER_END = 10000
BATCH_SIZE = 1024
N_EPOCHS = 8 # nonnegative integer

# Requires a prepare.py run to change, and untested
N_SPLITS = 5 
