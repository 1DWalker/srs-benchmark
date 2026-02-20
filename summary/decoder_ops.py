import time
import lmdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from summary.model import EncoderModel
import torch
from utils import load_tensor
import torch.nn.functional as F
from typing import NamedTuple


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
    return sum(map(lambda batch: (batch[0] < int(1e8)).sum().item(), batches))

def encode_single(batches, encoder_model: EncoderModel, min_review_th, max_review_th):
    assert min_review_th > 0
    assert max_review_th < 1e7
    device = batches[0][0].device
    min_review_th = torch.tensor([min_review_th], device=device)
    max_review_th = torch.tensor([max_review_th], device=device)
    review_range = max_review_th - min_review_th + 1

    x_list = []
    mask_list = []
    for batch in batches:
        feature_review_th_bl, feature_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_rating_bl, label_review_th_bl, label_is_same_day_bl, has_label_bl = batch
        assert (feature_review_th_bl > 0).all()
        B, L = feature_review_th_bl.shape
        min_review_bl = min_review_th.view(1, 1).expand(B, L)
        max_review_bl = max_review_th.view(1, 1).expand(B, L)
        mask_bl = (min_review_bl <= feature_review_th_bl) * (feature_review_th_bl <= max_review_bl)
        card_num_reviews_bl = mask_bl.sum(dim=-1, keepdim=True).expand(B, L)
        card_review_th_ratio_bl = feature_review_th_bl / max_review_bl

        idx = torch.arange(L, device=mask_bl.device).view(1, L)
        card_i_ratio_bl = idx / card_num_reviews_bl.clamp_min(1)

        x_sbl = encoder_model.first(
            feature_elapsed_days_real_bl=feature_elapsed_days_real_bl,
            feature_rating_bl=feature_rating_bl, 
            card_review_th_ratio_bl=card_review_th_ratio_bl, 
            global_num_reviews=review_range, 
            card_i_ratio_bl=card_i_ratio_bl, 
            card_num_reviews_bl=card_num_reviews_bl,
        )
        x_list.append(x_sbl)
        mask_list.append(mask_bl)

    return encoder_model.run_core(x_list, mask_list)

def encode(batches, encoder_model, min_review_th_s, max_review_th_s):
    return torch.stack([encode_single(batches, encoder_model, min_review_th_s[i].item(), max_review_th_s[i].item()) for i in range(min_review_th_s.size(0))])

class DecodeResult(NamedTuple):
    ce_loss_sum: torch.Tensor
    bce_loss_sum: torch.Tensor
    loss_n: int

def combine_decode_results(a: DecodeResult, b: DecodeResult) -> DecodeResult:
    return DecodeResult(
        ce_loss_sum = a.ce_loss_sum + b.ce_loss_sum,
        bce_loss_sum = a.bce_loss_sum + b.bce_loss_sum,
        loss_n = a.loss_n + b.loss_n,
    )

def first_logits(batches, first_review_logits_4, min_review_th, max_review_th):
    assert min_review_th > 0
    assert max_review_th < 1e7
    first_ce_tot = 0
    first_bce_tot = 0
    first_n = 0

    for batch in batches:
        # First review
        feature_review_th_bl, feature_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_rating_bl, label_review_th_bl, label_is_same_day_bl, has_label_bl = batch
        first_rating_b = feature_rating_bl[:, 0]
        B = first_rating_b.size(0)
        first_feature_review_th_b = feature_review_th_bl[:, 0]
        first_rating_mask_b = (min_review_th <= first_feature_review_th_b) * (first_feature_review_th_b <= max_review_th)
        first_n += first_rating_mask_b.sum()
        first_ce_loss = F.cross_entropy(first_review_logits_4.unsqueeze(0).expand(B, 4), first_rating_b.long() - 1, reduction='none')
        assert first_ce_loss.shape == first_rating_mask_b.shape
        first_ce_tot += (first_ce_loss * first_rating_mask_b).sum()

        first_fail_logit = first_review_logits_4[0]
        first_pass_logit = torch.logsumexp(first_review_logits_4[1:], dim=-1)
        first_logits_binary = first_pass_logit - first_fail_logit
        first_target_bce = (first_rating_b > 1).float()
        first_bce_loss = F.binary_cross_entropy_with_logits(first_logits_binary.expand(B), first_target_bce, reduction='none')
        assert first_bce_loss.shape == first_rating_mask_b.shape
        first_bce_tot += (first_bce_loss * first_rating_mask_b).sum()
    
    return DecodeResult(
        ce_loss_sum=first_ce_tot,
        bce_loss_sum=first_bce_tot,
        loss_n=first_n,
    )

def extract_first_review_dist_logits(first_review_model, encoding_h):
    return first_review_model(encoding_h.unsqueeze(0)).squeeze(0)

def extract_first_review_dist(first_review_model, encoding_h):
    return F.softmax(first_review_model(encoding_h.unsqueeze(0)).squeeze(0), dim=-1)

def first_decode(batches, first_review_model, encoding_h, min_review_th, max_review_th):
    first_review_logits_4 = first_review_model(encoding_h.unsqueeze(0)).squeeze(0)
    return first_logits(batches, first_review_logits_4, min_review_th, max_review_th)

def decode(batches, decoder_model, encoding_h, min_review_th, max_review_th):
    review_ce_tot = 0
    review_bce_tot = 0
    review_n = 0
    for batch in batches:
        feature_review_th_bl, feature_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_rating_bl, label_review_th_bl, label_is_same_day_bl, has_label_bl = batch
        B, L = feature_review_th_bl.shape
        H = encoding_h.size(0)
        logits_bl4 = decoder_model(
            encoding_h=encoding_h,
            feature_elapsed_days_real_bl=feature_elapsed_days_real_bl, 
            feature_rating_bl=feature_rating_bl, 
            label_elapsed_days_real_bl=label_elapsed_days_real_bl,
        )
        label_mask_bl = has_label_bl * (min_review_th <= label_review_th_bl) * (label_review_th_bl <= max_review_th)

        # CE loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        loss_bl = loss_fn(logits_bl4.transpose(1, 2), (label_rating_bl - 1).clamp(0).long())
        review_ce_tot += (loss_bl * label_mask_bl).sum()
        review_n += label_mask_bl.sum()

        # BCE loss
        label_pass_bl = (label_rating_bl > 1).float()
        logit_pass_bl = (
            torch.logsumexp(logits_bl4[..., 1:], dim=-1)
            - logits_bl4[..., 0]
        )
        loss_binary_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss_binary_bl = loss_binary_fn(logit_pass_bl, label_pass_bl)
        review_bce_tot += (loss_binary_bl * label_mask_bl).sum()

    return DecodeResult(
        ce_loss_sum=review_ce_tot,
        bce_loss_sum=review_bce_tot,
        loss_n=review_n,
    )

def decode_full(batches, decoder_model, encoding_hp, splits_list, equalize_review_ths, rmse_bins, device, equalize_test_reviews):
    assert len(splits_list) == encoding_hp.size(0) + 1

    rmse_bins_dict = dict(zip(equalize_review_ths, rmse_bins))
    bin_y_pred = {bin: [] for bin in set(rmse_bins)}
    bin_y = {bin: [] for bin in set(rmse_bins)}
    loss_tot = 0
    loss_n = 0
    y_pred = []
    y = []
    cond_loss_tot = 0
    cond_n = 0

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
        logits_bl4_list = []
        for h in range(H):
            logits_bl4 = decoder_model(
                encoding_h=encoding_hp[h],
                feature_elapsed_days_real_bl=feature_elapsed_days_real_bl, 
                feature_rating_bl=feature_rating_bl, 
                label_elapsed_days_real_bl=label_elapsed_days_real_bl,
            )
            logits_bl4_list.append(logits_bl4)
        logits_hbl4 = torch.stack(logits_bl4_list, dim=0)

        assert not logits_hbl4.isnan().any()
        probs_hbl4 = torch.softmax(logits_hbl4, dim=-1)
        p_success_hbl = probs_hbl4[..., 1:].sum(dim=-1)
        label_pass_bl = (label_rating_bl > 1).float()
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

        label_cond_mask_hbl = label_mask_hbl * label_pass_hbl
        cond_prob_hbl3 = probs_hbl4[..., 1:] / p_success_hbl.unsqueeze(-1)
        label_rating_hbl = label_rating_bl.unsqueeze(0).expand(H, -1, -1)
        loss_flat = torch.nn.functional.cross_entropy(
            cond_prob_hbl3.log().reshape(-1, 3),   # (H*B*L, 3)
            (label_rating_hbl.long() - 2).clamp_min(0).reshape(-1),
            reduction="none",
        )
        cond_loss_hbl = loss_flat.view(H, B, L)
        assert cond_loss_hbl.shape == label_cond_mask_hbl.shape
        cond_loss_tot += (cond_loss_hbl * label_cond_mask_hbl).sum()
        cond_n += label_cond_mask_hbl.sum()

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

    return loss_tot / loss_n, loss_n.item(), cond_loss_tot / (1e-7 + cond_n), cond_n.item(), rmse_raw, rmse_bins, auc