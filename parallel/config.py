from __future__ import annotations

from pathlib import Path

DEVICE = "cuda"  # not optional

LMDB_PATH = Path("parallel_db")
LMDB_SIZE = 32_569_171_968
TENSOR_CACHE_PATH = Path("/tensor-cache/lmdb")
TENSOR_CACHE_SIZE = 26_319_083_520
TENSOR_CACHE_VERSION = 6
USER_MAX_TRAIN_SPLIT_LENGTHS_KEY = "metadata_user_max_train_split_lengths"
TEST_BATCH_SIZE_MAX = 10_000_000

BATCH_PERM_SEED = 1234

# Writes to the result file if set to True, but incurs a large time cost.
WRITE_RESULT = True
WRITE_RESULT_FILE = "result/FSRS-7-dev.jsonl"

# Invalidates the cache
USER_START = 1
USER_END = 1000
BATCH_SIZE = 1024
N_EPOCHS = 0 # nonnegative integer

# Requires a prepare.py run to change, and untested
N_SPLITS = 5 
