from __future__ import annotations

from pathlib import Path


LMDB_PATH = Path("parallel_db")
LMDB_SIZE = 1_000_000_000
USER_IDS = list(range(1, 3))
DEVICE = "cuda"
USER_MAX_TRAIN_SPLIT_LENGTHS_KEY = "metadata_user_max_train_split_lengths"


TEST_PARALLEL = 