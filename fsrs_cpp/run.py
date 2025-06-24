import lmdb
import torch
from fsrs_cpp.fsrs_ops import FSRSCppFunction, RevlogTensors
from utils import load_tensor, parse_toml

def process(config, user_id):
    print("Process:", user_id)
    env = lmdb.open(config.DB_PATH, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        device = torch.device("cpu")
        packed_review_th_T = load_tensor(txn, f"{user_id}_packed_review_th_T", device)
        packed_rating_T = load_tensor(txn, f"{user_id}_packed_rating_T", device)
        packed_elapsed_days_real_T = load_tensor(txn, f"{user_id}_packed_elapsed_days_real_T", device)
        packed_elapsed_days_int_T = load_tensor(txn, f"{user_id}_packed_elapsed_days_int_T", device)
        packed_label_elapsed_days_real_T = load_tensor(txn, f"{user_id}_packed_label_elapsed_days_real_T", device)
        packed_label_elapsed_days_int_T = load_tensor(txn, f"{user_id}_packed_label_elapsed_days_int_T", device)
        perm_T_tensor = load_tensor(txn, f"{user_id}_perm_T_tensor", device)
        perm_inv_T_tensor = load_tensor(txn, f"{user_id}_perm_inv_T_tensor", device)
        card_locs_T = load_tensor(txn, f"{user_id}_card_locs_T", device)
        revlog_tensors = RevlogTensors(
            packed_review_th_T=packed_review_th_T,
            packed_rating_T=packed_rating_T,
            packed_elapsed_days_real_T=packed_elapsed_days_real_T,
            packed_elapsed_days_int_T=packed_elapsed_days_int_T,
            packed_label_elapsed_days_real_T=packed_label_elapsed_days_real_T,
            packed_label_elapsed_days_int_T=packed_label_elapsed_days_int_T,
            perm_T_tensor=perm_T_tensor,
            perm_inv_T_tensor=perm_inv_T_tensor,
            card_locs_T=card_locs_T,
        )

        for split_i in range(5):
            pretrain_params = load_tensor(txn, f"{user_id}_split_{split_i}_pretrain_params", device)
            epochs = load_tensor(txn, f"{user_id}_split_{split_i}_epochs", device)
            review_ths = load_tensor(txn, f"{user_id}_split_{split_i}_review_ths", device)
            batch_lens = load_tensor(txn, f"{user_id}_split_{split_i}_batch_lens", device)
            review_th_batches = review_ths.split(batch_lens.tolist())
            lrs = load_tensor(txn, f"{user_id}_split_{split_i}_lrs", device)
            train_set_review_ths = load_tensor(txn, f"{user_id}_split_{split_i}_train_set_review_ths", device)
            test_set_review_ths = load_tensor(txn, f"{user_id}_split_{split_i}_test_set_review_ths", device)

            params = pretrain_params.clone().requires_grad_(True)
            optim = torch.optim.Adam([params], lr=4e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=lrs.size(0)
            )
            
            # print(params)
            # for review_th_batch in review_th_batches:
            #     pred_y = FSRSCppFunction.apply(params, review_th_batch, revlog_tensors)
            #     print("Got pred_y", pred_y)
            #     exit()

            # TODO eval after every epoch

            test_pred_y = FSRSCppFunction.apply(params, test_set_review_ths, revlog_tensors)
            print("got pred", test_pred_y)
            # TODO get loss



def main(config):
    USER_IDS = list(range(config.USER_START, config.USER_END + 1))
    if 4371 in USER_IDS:
        USER_IDS.remove(4371)
        print("Removed user 4371.")

    for user in USER_IDS:
        process(config, user)
    

if __name__ == '__main__':
    config = parse_toml()
    main(config)