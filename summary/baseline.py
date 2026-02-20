import json
from pathlib import Path
from summary import decoder_ops
import torch
import lmdb
from tqdm import tqdm
from utils import load_tensor, parse_toml, sort_jsonl

DEVICE = torch.device("cpu")
RESULT_FILE = Path("result/average-baseline.jsonl")

def run_split_first_rating(test_l, test_r, batches):
    freqs = 0
    for batch in batches:
        feature_review_th_bl, feature_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_rating_bl, label_review_th_bl, label_is_same_day_bl, has_label_bl = batch
        first_rating_b = feature_rating_bl[:, 0]
        first_feature_review_th_b = feature_review_th_bl[:, 0]
        first_rating_mask_b = (first_feature_review_th_b < test_l)
        rating_b4 = torch.nn.functional.one_hot((first_rating_b.long() - 1).clamp(min=0), num_classes=4).float()
        freqs += (rating_b4 * first_rating_mask_b.view(-1, 1)).sum(dim=0)

    pred_p = (1 + freqs[1:].sum()) / (2 + freqs.sum())
    print("pred p:", pred_p)
    pred_4 = (1 + freqs) / (1 + freqs).sum()
    print("pred 4:", pred_4)

    bce_tot = 0
    bce_n = 0
    ce_tot = 0
    ce_n = 0
    for batch in batches:
        feature_review_th_bl, feature_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_rating_bl, label_review_th_bl, label_is_same_day_bl, has_label_bl = batch
        first_rating_b = feature_rating_bl[:, 0]
        first_feature_review_th_b = feature_review_th_bl[:, 0]
        first_rating_mask_b = (test_l <= first_feature_review_th_b) * (first_feature_review_th_b <= test_r)

        y = (feature_rating_bl[:, 0] >= 2).float()
        bce_tot += (first_rating_mask_b * torch.nn.functional.binary_cross_entropy(pred_p.expand_as(y), y, reduction='none')).sum()
        bce_n += first_rating_mask_b.sum()

        target = (feature_rating_bl[:, 0].long() - 1)
        ce_loss = torch.nn.functional.cross_entropy(pred_4.log().unsqueeze(0).expand(target.shape[0], -1), target, reduction='none')
        ce_tot += (first_rating_mask_b * ce_loss).sum()
        ce_n += first_rating_mask_b.sum()

    return bce_tot.item(), bce_n.item(), ce_tot.item(), ce_n.item()

def run_first_rating(batches, splits, T):
    bce_tot = 0
    bce_n = 0
    ce_tot = 0
    ce_n = 0
    for split_i in range(len(splits) - 1):
        test_l = splits[split_i]
        test_r = min(T, splits[split_i + 1] - 1)
        split_bce_tot, split_bce_n, split_ce_tot, split_ce_n = run_split_first_rating(test_l, test_r, batches)
        bce_tot += split_bce_tot
        bce_n += split_bce_n
        ce_tot += split_ce_tot
        ce_n += split_ce_n

    print("BCE avg:", bce_tot / bce_n if bce_n else float("nan"))
    print("CE avg:", ce_tot / ce_n if ce_n else float("nan"))
    print("n:", bce_n)
    assert ce_n == bce_n
    return bce_tot / (1e-7 + bce_n), ce_tot / (1e-7 + ce_n), bce_n

def run_split_conditional_rating(test_l, test_r, batches, equalize_review_ths):
    freqs = 0
    for batch in batches:
        feature_review_th_bl, feature_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_rating_bl, label_review_th_bl, label_is_same_day_bl, has_label_bl = batch
        mask_bl = (feature_review_th_bl < test_l) * (feature_rating_bl > 1)
        mask_bl[:, 0] = 0  # Ignore first rating

        rating_bl4 = torch.nn.functional.one_hot((feature_rating_bl.long() - 1).clamp(min=0), num_classes=4).float()
        freqs += (rating_bl4 * mask_bl.unsqueeze(-1)).sum(dim=(0, 1))

    freqs = freqs[1:]
    pred_3 = (1 + freqs) / (1 + freqs).sum()
    print("cond pred 3:", pred_3, test_l, freqs)

    ce_tot = 0
    ce_n = 0
    for batch in batches:
        feature_review_th_bl, feature_elapsed_days_int_bl, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_int_bl, label_elapsed_days_real_bl, label_rating_bl, label_review_th_bl, label_is_same_day_bl, has_label_bl = batch
        label_is_equalize_bl = torch.isin(label_review_th_bl.int(), torch.tensor(equalize_review_ths, device=label_review_th_bl.device))
        mask_bl = has_label_bl * label_is_equalize_bl * (test_l <= label_review_th_bl) * (label_review_th_bl <= test_r) * (label_rating_bl > 1)

        target = (label_rating_bl.long() - 2).clamp_min(0)
        B, L = label_elapsed_days_int_bl.shape
        ce_loss = torch.nn.functional.cross_entropy(pred_3.log().unsqueeze(0).expand(B * L, 3), target.view(-1), reduction='none').view(B, L)
        ce_tot += (mask_bl * ce_loss).sum()
        ce_n += mask_bl.sum()

    return ce_tot.item(), ce_n.item()


def run_conditional_rating(batches, splits, equalize_review_ths, T):
    ce_tot = 0
    ce_n = 0
    for split_i in range(len(splits) - 1):
        test_l = splits[split_i]
        test_r = min(T, splits[split_i + 1] - 1)
        split_ce_tot, split_ce_n = run_split_conditional_rating(test_l, test_r, batches, equalize_review_ths)
        ce_tot += split_ce_tot
        ce_n += split_ce_n
    print("Cond CE avg:", ce_tot / ce_n if ce_n else float("nan"), ce_n)
    return ce_tot / (1e-7 + ce_n), ce_n

def to_json(user_id, cond_loss, cond_n, first_bce, first_ce, first_n):
    try:
        auc = round(auc, 6)
    except:
        auc = None
    stats = {
        "metrics": {
            "CE | passed review": round(cond_loss, 6),
            "First BCE": round(first_bce, 6),
            "First CE": round(first_ce, 6),
        },
        "user": int(user_id),
        "passed review size": int(cond_n),
        "first rating size": int(first_n),
    }
    return stats

@torch.inference_mode()
def run(user, config):
    compressed_db_env = lmdb.open(config.SUMMARY_DB_PATH, readonly=True, lock=False)
    label_filter_env = lmdb.open(config.LABEL_FILTER_DB_PATH, readonly=True, lock=False)
    with compressed_db_env.begin(write=False) as summary_txn:
        with label_filter_env.begin(write=False) as label_filter_txn:
            batches = decoder_ops.get_data(summary_txn, user, DEVICE)
            T = decoder_ops.extract_num_reviews(batches)
            splits = load_tensor(label_filter_txn, f"{user}_split", device=DEVICE).tolist()
            equalize_review_ths = load_tensor(label_filter_txn, f"{user}_review_ths", DEVICE).tolist()
            assert len(splits) == 6
            bce, ce, first_n = run_first_rating(batches, splits, T)
            cond_ce, cond_n = run_conditional_rating(batches, splits, equalize_review_ths, T)

            stats = to_json(user, cond_ce, cond_n, bce, ce, first_n)
            with open(RESULT_FILE, "a") as f:
                f.write(json.dumps(stats, ensure_ascii=False) + "\n")

def main(config): 
    unprocessed_users = []
    Path("result").mkdir(parents=True, exist_ok=True)
    if RESULT_FILE.exists():
        data = sort_jsonl(RESULT_FILE)
        processed_user = set(map(lambda x: x["user"], data))
    else:
        processed_user = set()

    users = list(range(1, 1001))
    for user_id in users:
        if user_id not in processed_user:
            unprocessed_users.append(user_id)

    unprocessed_users.sort()
    print("Unprocessed users length:", len(unprocessed_users))
    for user in tqdm(unprocessed_users, smoothing=1e-2):
        run(user, config)

if __name__ == '__main__':
    config = parse_toml()
    main(config)