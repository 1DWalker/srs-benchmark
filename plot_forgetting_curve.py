import lmdb
import torch

from full_db_config import FULL_DB_PATH
from utils import load_tensor

FILE_NAMES = ["RWKV", "FSRS-6-recency"]

def main():
    user_id = 1
    env = lmdb.open(FULL_DB_PATH, readonly=True, lock=False)

    with env.begin() as txn:
        print("start")
        rwkv_w = load_tensor(txn, f"{user_id}_RWKV_w", device=torch.device("cpu"))
        fsrs_s = load_tensor(txn, f"{user_id}_FSRS-6-recency_s", device=torch.device("cpu")).numpy()
        fsrs_decay = load_tensor(txn, f"{user_id}_FSRS-6-recency_decay", device=torch.device("cpu")).numpy()
        card_ids = load_tensor(txn, f"{user_id}_card_id", device=torch.device("cpu")).numpy()
        ys = load_tensor(txn, f"{user_id}_y", device=torch.device("cpu")).numpy()
        elapsed_seconds_list = load_tensor(txn, f"{user_id}_elapsed_seconds", device=torch.device("cpu")).numpy()
        print(rwkv_w)
        print(fsrs_s)
        print(rwkv_w.shape, fsrs_s.shape)
        print(card_ids)
        card_id_locs = {}
        for i, card_id in enumerate(card_ids):
            if fsrs_s[i] != -1:
                if card_id not in card_id_locs:
                    elapsed_seconds_list[i] = 0  # Fix some values
                    card_id_locs[card_id] = []
                card_id_locs[card_id].append(i)

        # print(card_id_locs)
        # for k, v in card_id_locs.items():
        #     tot_elapsed_time = 0
        #     if len(v) > 8:
        #         for i in v:
        #             tot_elapsed_time += elapsed_seconds_list[i]
        #         if tot_elapsed_time < 20 * 86400:
        #             print(k, v)
        #             exit()


        # card_id = 1725
        card_id = 2558
        locs = card_id_locs[card_id]
        for i in locs:
            print(i, fsrs_s[i], fsrs_decay[i], elapsed_seconds_list[i] / 86400, ys[i])
            # print(rwkv_w[i])



        # with txn.cursor() as cursor:
        #     for key in cursor.iternext(keys=True, values=False):
        #         print(key)

if __name__ == '__main__':
    main()