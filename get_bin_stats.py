import json
from pathlib import Path

import numpy as np
import pandas as pd
from config import create_parser
import pyarrow.parquet as pq

from other import create_features  # type: ignore


parser = create_parser()
args = parser.parse_args()

DATA_PATH = Path(args.data)

def count_lapse(r_history, t_history):
    lapse = 0
    for r, t in zip(r_history.split(","), t_history.split(",")):
        if t != "0" and r == "1":
            lapse += 1
    return lapse

def get_bin(row):
    raw_lapse = count_lapse(row["r_history"], row["t_history"])
    lapse = round(1.65 * np.power(1.73, np.floor(np.log(raw_lapse) / np.log(1.73))), 0) if raw_lapse != 0 else 0
    delta_t = round(2.48 * np.power(3.62, np.floor(np.log(row["delta_t"]) / np.log(3.62))), 2)
    i = round(1.99 * np.power(1.89, np.floor(np.log(row["i"]) / np.log(1.89))), 0)
    return (lapse, delta_t, i)

class RMSEBinsExploit:
    def __init__(self):
        super().__init__()
        self.state = {}
        self.global_succ = 0
        self.global_n = 0

    def adapt(self, bin_key, y):
        if bin_key not in self.state:
            self.state[bin_key] = (0, 0, 0)

        pred_sum, truth_sum, bin_n = self.state[bin_key]
        self.state[bin_key] = (pred_sum, truth_sum + y, bin_n + 1)
        self.global_succ += y
        self.global_n += 1

def process(user_id):
    print("Process", user_id)
    df_revlogs = pd.read_parquet(
        DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    dataset = create_features(df_revlogs, "GET BIN STATS")
    model = RMSEBinsExploit()
    for i in range(len(dataset)):
        row = dataset.iloc[i].copy()
        bin = get_bin(row)
        model.adapt(bin, row["y"])

    out_dict = {}
    for k, v in model.state.items():
        _, truth_sum, bin_n = v
        lapse, delta_t, i = k
        out_dict[f"{lapse},{delta_t},{i}"] = (truth_sum / bin_n, bin_n)

    with open(f"bin-stats/{user_id}.jsonl", "w") as f:
        f.write(json.dumps(out_dict, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    dataset = pq.ParquetDataset(DATA_PATH / "revlogs")

    user_ids = []
    for user_id in dataset.partitioning.dictionaries[0]:
        user_ids.append(user_id.as_py())

    user_ids.sort()
    user_ids = list(filter(lambda x: x <= 20, user_ids))

    for user_id in user_ids:
        process(user_id)