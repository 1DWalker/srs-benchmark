import time
import lmdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, root_mean_squared_error
import torch
from utils import load_tensor


def load_batch(txn, user, i, device):
    feature_review_th = load_tensor(txn, f"{user}_feature_review_th_{i}", device)
    feature_elapsed_days_int = load_tensor(txn, f"{user}_feature_elapsed_days_int_{i}", device)
    feature_elapsed_days_real = load_tensor(txn, f"{user}_feature_elapsed_days_real_{i}", device)
    feature_rating = load_tensor(txn, f"{user}_feature_rating_{i}", device)
    label_elapsed_days_int = load_tensor(txn, f"{user}_label_elapsed_days_int_{i}", device)
    label_elapsed_days_real = load_tensor(txn, f"{user}_label_elapsed_days_real_{i}", device)
    label_rating = load_tensor(txn, f"{user}_label_rating_{i}", device)
    label_review_th = load_tensor(txn, f"{user}_label_review_th_{i}", device)
    label_is_same_day = load_tensor(txn, f"{user}_label_is_same_day_{i}", device)
    has_label = load_tensor(txn, f"{user}_has_label_{i}", device)
    return feature_review_th, feature_elapsed_days_int, feature_elapsed_days_real, feature_rating, label_elapsed_days_int, label_elapsed_days_real, label_rating, label_review_th, label_is_same_day, has_label

def get_data(txn, user_id, device):
    num_batches = load_tensor(txn, f"{user_id}_batches", torch.device("cpu")).item()
    indices = list(range(num_batches))
    return [load_batch(txn, user_id, i, device) for i in indices]

def extract_num_reviews(batches):
    return max(map(lambda batch: batch[0].max().item(), batches))

def encode(batches, encoder_model, min_review_th, max_review_th):
    review_range = max_review_th - min_review_th + 1
    accum_weighted_value_h = 0
    accum_weight_h = 0
    for batch in batches:
        feature_review_th_bl, feature_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_rating_bl, label_review_th_bl, label_is_same_day_bl, has_label_bl = batch
        weight_blh, value_blh = encoder_model(feature_elapsed_days_real_bl=feature_elapsed_days_real_bl, feature_rating_bl=feature_rating_bl)
        mask_bl = (min_review_th <= feature_review_th_bl) * (feature_review_th_bl <= max_review_th)
        clamped_ord_bl = (feature_review_th_bl - min_review_th).clamp(min=0, max=review_range - 1)
        recency_weights_bl = encoder_model.get_recency_weights(clamped_ord_bl, review_range)
        eff_weight_blh = mask_bl.unsqueeze(-1).float() * recency_weights_bl.unsqueeze(-1) * weight_blh
        accum_weighted_value_h += (eff_weight_blh * value_blh).sum(dim=(0, 1))
        accum_weight_h += eff_weight_blh.sum(dim=(0, 1))
        assert accum_weighted_value_h.size(0) == weight_blh.size(2)

    base_h = accum_weighted_value_h / (accum_weight_h + 1)
    return encoder_model.transform(base_h), base_h

def decode(batches, decoder_model, encoding_h, min_review_th, max_review_th):
    loss_tot = 0
    loss_n = 0
    for batch in batches:
        feature_review_th_bl, feature_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_rating_bl, label_review_th_bl, label_is_same_day_bl, has_label_bl = batch
        logits_bl4 = decoder_model(
            encoding_h=encoding_h,
            feature_elapsed_days_real_bl=feature_elapsed_days_real_bl, 
            feature_rating_bl=feature_rating_bl, 
            label_elapsed_days_real_bl=label_elapsed_days_real_bl,
            label_is_new_anki_day=(label_elapsed_days_int_bl > 0).float(),
        )
        label_mask_bl = has_label_bl * (min_review_th <= label_review_th_bl) * (label_review_th_bl <= max_review_th)

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        loss_bl = loss_fn(logits_bl4.transpose(1, 2), (label_rating_bl - 1).clamp(0).long())
        loss_tot += (loss_bl * label_mask_bl).sum()
        loss_n += label_mask_bl.sum()
    
    return loss_tot / (1e-7 + loss_n), loss_tot, loss_n

def decode_full(batches, model, parameter_list, splits_list, user_id, device=torch.device("cpu"), equalize_test_reviews=False):
    assert len(splits_list) == len(parameter_list) + 1
    H = len(parameter_list)
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
        feature_review_th_bl, feature_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_rating_hbl, label_review_th_bl, label_is_same_day_bl, has_label_bl = batch
        out_hbl = model(parameters_hp, feature_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl=label_elapsed_days_int_bl)
        assert not out_hbl.isnan().any()
        label_rating_hbl = label_rating_hbl.unsqueeze(0).expand(H, -1, -1).float()
        label_review_th_hbl = label_review_th_bl.unsqueeze(0).expand(H, -1, -1)
        has_label_hbl = has_label_bl.unsqueeze(0).expand(H, -1, -1)
        label_is_equalize_hbl = label_is_equalize_bl.unsqueeze(0).expand(H, -1, -1)
        label_mask_hbl = has_label_hbl * (min_review_th_h.view(H, 1, 1) <= label_review_th_hbl) * (label_review_th_hbl <= max_review_th_h.view(H, 1, 1))
        if equalize_test_reviews:
            label_mask_hbl = label_mask_hbl * label_is_equalize_hbl
        
        loss_fn = torch.nn.BCELoss(reduction="none")
        loss_hbl = loss_fn(out_hbl, label_rating_hbl)
        loss_tot += (loss_hbl * label_mask_hbl).sum()
        loss_n += label_mask_hbl.sum()

        _, B, L = label_rating_hbl.shape
        mask_np = label_mask_hbl.cpu().numpy()
        out_np = out_hbl.detach().cpu().numpy()
        label_review_th_np = label_review_th_hbl.cpu().numpy()
        label_y_np = label_rating_hbl.cpu().numpy()
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