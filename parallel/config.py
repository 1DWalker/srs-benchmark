from __future__ import annotations

from pathlib import Path


LMDB_PATH = Path("parallel_db")
LMDB_SIZE = 50_000_000_000
USER_IDS = list(range(1, 10001))
DEVICE = "cuda"
DATA_BUILD_DEVICE = "cpu"
USER_MAX_TRAIN_SPLIT_LENGTHS_KEY = "metadata_user_max_train_split_lengths"


BATCH_SIZE = 1024
N_EPOCHS = 8
N_SPLITS = 5 
TRAIN_BUFFER_SIZE_GB = 5
TEST_BATCH_SIZE_MAX = 10_000_000
