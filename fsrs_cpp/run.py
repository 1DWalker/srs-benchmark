import lmdb
import numpy as np
from sklearn.metrics import log_loss
import torch
from fsrs_cpp.fsrs6_reference import FSRS6
from fsrs_cpp.fsrs_ops import FSRSCppFunction, RevlogTensors
from utils import load_tensor, parse_toml

def fsrs_fun(params, review_th_batch, revlog_tensors):
    return FSRSCppFunction.apply(params, review_th_batch, revlog_tensors)

def train_eval(params, train_set, revlog_tensors: RevlogTensors):
    with torch.no_grad():
        pred_y = fsrs_fun(params, train_set, revlog_tensors)
        y = (revlog_tensors.packed_rating_T[revlog_tensors.perm_inv_T_tensor[train_set]] > 1).float()
        return log_loss(y_true=y, y_pred=pred_y, labels=[0, 1])

def process(config, user_id):
    print("Process:", user_id)
    fsrs_reference = FSRS6()

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
        y_T = (packed_rating_T[perm_inv_T_tensor] > 1).float()
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

        test_pred_y_all = []
        test_y_all = []
        for split_i in range(5):
            # pretrain_params = load_tensor(txn, f"{user_id}_split_{split_i}_pretrain_params", device)
            print("Using defaul params")
            init_w = [
                0.212,
                1.2931,
                2.3065,
                8.2956,
                6.4133,
                0.8334,
                3.0194,
                0.001,
                1.8722,
                0.1666,
                0.796,
                1.4835,
                0.0614,
                0.2629,
                1.6483,
                0.6014,
                1.8729,
                0.5425,
                0.0912,
                0.0658,
                0.1542,
            ]
            pretrain_params = torch.tensor(init_w)


            epochs = load_tensor(txn, f"{user_id}_split_{split_i}_epochs", device)
            epochs_np = epochs.numpy()
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
            best_loss = np.inf
            for batch_i, review_th_batch in enumerate(review_th_batches):
                is_new_epoch = batch_i == 0 or epochs_np[batch_i - 1] != epochs_np[batch_i]
                if is_new_epoch:
                    eval_loss = train_eval(params.detach().clone(), train_set_review_ths, revlog_tensors)
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        best_params = params.detach().clone()

                pred_y_batch = FSRSCppFunction.apply(params, review_th_batch, revlog_tensors)
                y_batch = y_T[review_th_batch - 1]

                # print("Got pred_y", pred_y_batch)
                loss_fn = torch.nn.BCELoss(reduction='none')
                loss = loss_fn(pred_y_batch, y_batch)
                loss_sum = loss.sum()
                print("loss:", loss.mean())
                loss_sum.backward()
                optim.step()
                optim.zero_grad()
                scheduler.step()

                params = fsrs_reference.clip(params)

            eval_loss = train_eval(params.detach().clone(), train_set_review_ths, revlog_tensors)
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_params = params.detach().clone()

            # TODO eval after every epoch
            print("Best params:", best_params)
            test_pred_y = fsrs_fun(best_params, test_set_review_ths, revlog_tensors)
            test_y = y_T[test_set_review_ths - 1]
            test_pred_y_all.extend(test_pred_y.detach().numpy())
            test_y_all.extend(test_y.numpy())

        logloss = log_loss(y_true=test_y_all, y_pred=test_pred_y_all, labels=[0, 1])
        print("Log loss:", logloss)

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