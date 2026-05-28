from __future__ import annotations

from pathlib import Path

DEVICE = "cuda"  # not optional

# These paths point to Docker volumes
LMDB_PATH = Path("/data/parallel_db")
TENSOR_CACHE_PATH = Path("/data/tensor_cache_db")
LMDB_SIZE = 32_569_171_968
TENSOR_CACHE_SIZE = 26_319_083_520

TENSOR_CACHE_VERSION = 6
USER_MAX_TRAIN_SPLIT_LENGTHS_KEY = "metadata_user_max_train_split_lengths"
TEST_BATCH_SIZE_MAX = 10_000_000

BATCH_PERM_SEED = 1234

# Writes to the result file if set to True, but incurs a large time cost.
WRITE_RESULT = False
WRITE_RESULT_FILE = "result/FSRS-7-dev.jsonl"

BATCH_SIZE = 1024  # Should be a multiple of 128, the block sized used by the cuda kernel
N_EPOCHS = 8 # nonnegative integer, set to 0 for no optimization

# Invalidates the cache (slow)
USER_START = 1
USER_END = 10000

# Requires a prepare.py run to change, and untested
N_SPLITS = 5 
