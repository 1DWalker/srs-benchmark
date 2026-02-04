import time
import lmdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, root_mean_squared_error
import torch
from utils import load_tensor


def load_batch(txn, user, i, device):
    feature_elapsed_days_int = load_tensor(txn, f"{user}_feature_elapsed_days_int_{i}", device)
    feature_elapsed_days_real = load_tensor(txn, f"{user}_feature_elapsed_days_real_{i}", device)
    feature_rating = load_tensor(txn, f"{user}_feature_rating_{i}", device)
    label_elapsed_days_int = load_tensor(txn, f"{user}_label_elapsed_days_int_{i}", device)
    label_elapsed_days_real = load_tensor(txn, f"{user}_label_elapsed_days_real_{i}", device)
    label_y = load_tensor(txn, f"{user}_label_y_{i}", device)
    label_review_th = load_tensor(txn, f"{user}_label_review_th_{i}", device)
    label_is_same_day = load_tensor(txn, f"{user}_label_is_same_day_{i}", device)
    label_is_equalize = load_tensor(txn, f"{user}_label_is_equalize_{i}", device)
    has_label = load_tensor(txn, f"{user}_has_label_{i}", device)
    return feature_elapsed_days_int, feature_elapsed_days_real, feature_rating, label_elapsed_days_int, label_elapsed_days_real, label_y, label_review_th, label_is_same_day, label_is_equalize, has_label

def get_data(txn, user_id, device):
    num_batches = load_tensor(txn, f"{user_id}_batches", torch.device("cpu")).item()
    indices = list(range(num_batches))
    return [load_batch(txn, user_id, i, device) for i in indices]

def evaluate_batched_parameters(txn, model, parameters_hp, user_id, min_review_th_h, max_review_th_h, device, equalize_test_reviews=False):
    H = min_review_th_h.size(0)
    batches = get_data(txn, user_id, device)
    loss_tot_h = 0
    loss_n_h = 0
    for batch in batches:
        features_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_y_bl, label_review_th_bl, label_is_same_day_bl, label_is_equalize_bl, has_label_bl = batch
        out_hbl = model(parameters_hp, features_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl=label_elapsed_days_int_bl)
        label_y_hbl = label_y_bl.unsqueeze(0).expand(H, -1, -1).float()
        label_review_th_hbl = label_review_th_bl.unsqueeze(0).expand(H, -1, -1)
        has_label_hbl = has_label_bl.unsqueeze(0).expand(H, -1, -1)
        label_is_equalize_hbl = label_is_equalize_bl.unsqueeze(0).expand(H, -1, -1)
        label_mask_hbl = has_label_hbl * (min_review_th_h.view(H, 1, 1) <= label_review_th_hbl) * (label_review_th_hbl <= max_review_th_h.view(H, 1, 1))
        if equalize_test_reviews:
            label_mask_hbl = label_mask_hbl * label_is_equalize_hbl

        loss_fn = torch.nn.BCELoss(reduction="none")
        loss_hbl = loss_fn(out_hbl, label_y_hbl)
        loss_tot_h += (loss_hbl * label_mask_hbl).sum(dim=(1, 2))
        loss_n_h += label_mask_hbl.sum(dim=(1, 2))
    
    return loss_tot_h / (1e-7 + loss_n_h), loss_tot_h, loss_n_h

def evaluate_full(txn, model, parameter_list, splits_list, user_id, device=torch.device("cpu"), equalize_test_reviews=False):
    assert len(splits_list) == len(parameter_list) + 1
    H = len(parameter_list)
    batches = get_data(txn, user_id, device)
    equalize_review_ths = load_tensor(txn, f"{user_id}_equalize_review_ths", device).tolist()
    rmse_bins = load_tensor(txn, f"{user_id}_rmse_bins", device).tolist()

    rmse_bins_dict = dict(zip(equalize_review_ths, rmse_bins))
    bin_y_pred = {bin: [] for bin in set(rmse_bins)}
    bin_y = {bin: [] for bin in set(rmse_bins)}
    loss_tot = 0
    loss_n = 0
    y_pred = []
    y = []

    parameters_hp = torch.stack(parameter_list, dim=0)
    min_review_th_list = []
    max_review_th_list = []
    for split_i in range(len(parameter_list)):
        min_review_th_list.append(splits_list[split_i])
        max_review_th_list.append(splits_list[split_i + 1] - 1)
    min_review_th_h = torch.tensor(min_review_th_list, device=device)
    max_review_th_h = torch.tensor(max_review_th_list, device=device)

    for batch in batches:
        features_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_y_hbl, label_review_th_bl, label_is_same_day_bl, label_is_equalize_bl, has_label_bl = batch
        out_hbl = model(parameters_hp, features_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl=label_elapsed_days_int_bl)
        assert not out_hbl.isnan().any()
        label_y_hbl = label_y_hbl.unsqueeze(0).expand(H, -1, -1).float()
        label_review_th_hbl = label_review_th_bl.unsqueeze(0).expand(H, -1, -1)
        has_label_hbl = has_label_bl.unsqueeze(0).expand(H, -1, -1)
        label_is_equalize_hbl = label_is_equalize_bl.unsqueeze(0).expand(H, -1, -1)
        label_mask_hbl = has_label_hbl * (min_review_th_h.view(H, 1, 1) <= label_review_th_hbl) * (label_review_th_hbl <= max_review_th_h.view(H, 1, 1))
        if equalize_test_reviews:
            label_mask_hbl = label_mask_hbl * label_is_equalize_hbl
        
        loss_fn = torch.nn.BCELoss(reduction="none")
        loss_hbl = loss_fn(out_hbl, label_y_hbl)
        loss_tot += (loss_hbl * label_mask_hbl).sum()
        loss_n += label_mask_hbl.sum()

        _, B, L = label_y_hbl.shape
        mask_np = label_mask_hbl.cpu().numpy()
        out_np = out_hbl.detach().cpu().numpy()
        label_review_th_np = label_review_th_hbl.cpu().numpy()
        label_y_np = label_y_hbl.cpu().numpy()
        for h in range(H):
            for b in range(B):
                for l in range(L):
                    if mask_np[h, b, l]:
                        y.append(label_y_np[h, b, l])
                        y_pred.append(out_np[h, b, l])
                        bin = rmse_bins_dict[label_review_th_np[h, b, l]]
                        bin_y[bin].append(label_y_np[h, b, l])
                        bin_y_pred[bin].append(out_np[h, b, l])
        assert len(y) == loss_n.item()

    if loss_n == 0:
        return 0, 0, 0, 0, 0

    rmse_raw = root_mean_squared_error(y_true=y, y_pred=y_pred)
    try:
        auc = round(roc_auc_score(y_true=y, y_score=y_pred), 6)
    except:
        auc = None

    rows = []
    for bin in bin_y_pred.keys():
        for y, pred in zip(bin_y[bin], bin_y_pred[bin]):
            rows.append([bin, y, pred, 1])
    assert len(rows) == len(equalize_review_ths)

    tmp = pd.DataFrame(rows, columns=["bin", "y", "p", "weights"])
    tmp = (
        tmp.groupby("bin")
        .agg({"y": "mean", "p": "mean", "weights": "sum"})
        .reset_index()
    )
    rmse_bins = root_mean_squared_error(
        tmp["y"], tmp["p"], sample_weight=tmp["weights"]
    )

    return loss_tot / loss_n, loss_n.item(), rmse_raw, rmse_bins, auc

def main():
    device = torch.device("cuda")
    fsrs = FSRS6().to(device)
    parameters = torch.tensor([
        0.2172,
        1.1771,
        3.2602,
        16.1507,
        7.0114,
        0.57,
        2.0966,
        0.0069,
        1.5261,
        0.112,
        1.0178,
        1.849,
        0.1133,
        0.3127,
        2.2934,
        0.2191,
        3.0004,
        0.7536,
        0.3332,
        0.1437,
        0.2,
    ], device=device)

    DB_PATH = "fsrs_evaluate_db"
    env = lmdb.open(DB_PATH, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        n = 0
        for user_id in range(1, 101):
            if user_id == 10:
                time_start = time.time()
            with torch.no_grad():
                print(evaluate(txn, model=fsrs, parameters=parameters, user_id=user_id, min_review_th=1, max_review_th=1e9, device=device, equalize_test_reviews=True, include_other_metrics=True, skip_same_day_reviews=True))
                n += 1

        if n > 0:
            print(f"speed: {n / (time.time() - time_start):.3f} users/second")
            
        # evaluate(txn, fsrs=fsrs, parameters=parameters, user_id=4, device=device, equalize_test_reviews=True)
    env.close()

if __name__ == '__main__':
    main()