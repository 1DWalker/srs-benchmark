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

def encode(batches, encoder_model, min_review_th_s, max_review_th_s):
    review_range_s = max_review_th_s - min_review_th_s + 1
    S = min_review_th_s.size(0)
    accum_weighted_value_sh = 0
    accum_weight_sh = 0
    # mask_sum_s = 0
    # non_norm_recency_weights_sum_s = 0
    for batch in batches:
        feature_review_th_bl, feature_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_rating_bl, label_review_th_bl, label_is_same_day_bl, has_label_bl = batch
        B, L = feature_review_th_bl.shape
        weight_blh, value_blh = encoder_model(feature_elapsed_days_real_bl=feature_elapsed_days_real_bl, feature_rating_bl=feature_rating_bl)
        min_review_sbl = min_review_th_s.view(S, 1, 1).expand(S, B, L)
        max_review_sbl = max_review_th_s.view(S, 1, 1).expand(S, B, L)
        review_range_sbl = review_range_s.view(S, 1, 1).expand(S, B, L)
        feature_review_th_sbl = feature_review_th_bl.unsqueeze(0).expand(S, B, L)
        mask_sbl = (min_review_sbl <= feature_review_th_sbl) * (feature_review_th_sbl <= max_review_sbl)
        # mask_sum_s += mask_sbl.sum(dim=(1, 2))
        clamped_ord_sbl = (feature_review_th_sbl - min_review_sbl).clamp(min=torch.zeros_like(review_range_sbl), max=review_range_sbl - 1)
        non_norm_recency_weights_sbl = encoder_model.get_non_norm_recency_weights(clamped_ord_sbl, review_range_s)
        # non_norm_recency_weights_sum_s += non_norm_recency_weights_sbl.sum(dim=(1, 2))
        eff_weight_sblh = mask_sbl.unsqueeze(-1).float() * non_norm_recency_weights_sbl.unsqueeze(-1) * weight_blh.unsqueeze(0)
        accum_weighted_value_sh += (eff_weight_sblh * value_blh.unsqueeze(0)).sum(dim=(1, 2))
        accum_weight_sh += eff_weight_sblh.sum(dim=(1, 2))
        assert accum_weighted_value_sh.size(1) == weight_blh.size(2)

    base_sh = accum_weighted_value_sh / (accum_weight_sh + 1e-6)
    return encoder_model.transform(base_sh, review_range_s), base_sh

def encode_single(batches, encoder_model, min_review_th, max_review_th):
    device = batches[0][0].device
    encoding, sum_encoding = encode(batches, encoder_model, torch.tensor([min_review_th], device=device), torch.tensor([max_review_th], device=device))
    return encoding[0], sum_encoding[0]

def decode(batches, decoder_model, encoding_h, min_review_th, max_review_th):
    loss_tot = 0
    loss_n = 0
    for batch in batches:
        feature_review_th_bl, feature_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_rating_bl, label_review_th_bl, label_is_same_day_bl, has_label_bl = batch
        B, L = feature_review_th_bl.shape
        H = encoding_h.size(0)
        logits_bl4 = decoder_model(
            encoding_bh=encoding_h.unsqueeze(0).expand(B, H),
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

def decode_full(batches, decoder_model, encoding_hp, splits_list, equalize_review_ths, rmse_bins, device, equalize_test_reviews):
    assert len(splits_list) == encoding_hp.size(0) + 1

    rmse_bins_dict = dict(zip(equalize_review_ths, rmse_bins))
    bin_y_pred = {bin: [] for bin in set(rmse_bins)}
    bin_y = {bin: [] for bin in set(rmse_bins)}
    loss_tot = 0
    loss_n = 0
    y_pred = []
    y = []
    bin_unique = np.unique(rmse_bins)

    H, P = encoding_hp.shape
    min_review_th_list = []
    max_review_th_list = []
    for split_i in range(len(splits_list) - 1):
        min_review_th_list.append(splits_list[split_i])
        max_review_th_list.append(splits_list[split_i + 1] - 1)
    min_review_th_h = torch.tensor(min_review_th_list, device=device)
    max_review_th_h = torch.tensor(max_review_th_list, device=device)


    for batch in batches:
        feature_review_th_bl, feature_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_rating_bl, label_review_th_bl, label_is_same_day_bl, has_label_bl = batch
        B, L = feature_elapsed_days_int_bl.shape
        logits_hbl4 = decoder_model(
            encoding_bh=encoding_hp.view(H, 1, -1).expand(H, B, P).reshape(-1, P),
            feature_elapsed_days_real_bl=feature_elapsed_days_real_bl.expand(H, B, L).reshape(-1, L),
            feature_rating_bl=feature_rating_bl.expand(H, B, L).reshape(-1, L), 
            label_elapsed_days_real_bl=label_elapsed_days_real_bl.expand(H, B, L).reshape(-1, L),
            label_is_new_anki_day=(label_elapsed_days_int_bl > 0).float().expand(H, B, L).reshape(-1, L),
        ).view(H, B, L, 4)
        assert not logits_hbl4.isnan().any()
        probs_hbl4 = torch.softmax(logits_hbl4, dim=-1)
        p_success_hbl = probs_hbl4[..., 1:].sum(dim=-1)
        label_pass_bl = (label_rating_bl >= 2).float()
        label_pass_hbl = label_pass_bl.unsqueeze(0).expand(H, -1, -1).float()
        label_review_th_hbl = label_review_th_bl.unsqueeze(0).expand(H, -1, -1)
        has_label_hbl = has_label_bl.unsqueeze(0).expand(H, -1, -1)
        label_is_equalize_bl = torch.isin(label_review_th_bl.int(), torch.tensor(equalize_review_ths, device=label_review_th_bl.device))
        label_is_equalize_hbl = label_is_equalize_bl.unsqueeze(0).expand(H, -1, -1)
        label_mask_hbl = has_label_hbl * (min_review_th_h.view(H, 1, 1) <= label_review_th_hbl) * (label_review_th_hbl <= max_review_th_h.view(H, 1, 1))
        if equalize_test_reviews:
            label_mask_hbl = label_mask_hbl * label_is_equalize_hbl
        
        loss_fn = torch.nn.BCELoss(reduction="none")
        loss_hbl = loss_fn(p_success_hbl, label_pass_hbl)
        loss_tot += (loss_hbl * label_mask_hbl).sum()
        loss_n += label_mask_hbl.sum()

        _, B, L = label_pass_hbl.shape
        mask_np = label_mask_hbl.cpu().numpy()
        p_success_np = p_success_hbl.detach().cpu().numpy()
        label_review_th_np = label_review_th_hbl.cpu().numpy()
        label_y_np = label_pass_hbl.cpu().numpy()
        mask_np = mask_np.astype(bool)
        y.extend(label_y_np[mask_np])
        y_pred.extend(p_success_np[mask_np])
        for h in range(H):
            for b in range(B):
                for l in range(L):
                    if mask_np[h, b, l]:
                        bin = rmse_bins_dict[label_review_th_np[h, b, l]]
                        bin_y[bin].append(label_y_np[h, b, l])
                        bin_y_pred[bin].append(p_success_np[h, b, l])
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