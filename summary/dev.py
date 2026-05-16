from summary import fsrs_7, fsrs_7_curve_nn, fsrs_7_custom, fsrs_7_s_nn, fsrs_7_truth, model
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def find_streak(mask):
    succ_cumsum = mask.cumsum(dim=-1)
    succ_cumsum_at_fail = succ_cumsum * (1 - mask)
    last_succ_cumsum_at_fail = torch.cummax(succ_cumsum_at_fail, dim=-1).values
    return succ_cumsum - last_succ_cumsum_at_fail

class CardModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.n_feature = 0
        self.feature_lin = nn.Linear(37, self.n_feature)
        # self.act = CustomAct(self.n_feature)
        self.feat_deg = nn.Parameter(torch.ones(self.n_feature))
        # self.n_input = 32
        self.n_hidden = 39
        self.enc_scale = nn.Linear(n_encoding, self.n_hidden + self.n_feature)
        self.enc_curve = nn.Linear(n_encoding, 8)
        self.enc_deg = nn.Linear(n_encoding, 1)
        self.scale = nn.Linear(self.n_hidden + self.n_feature, 1, bias=False)
        self.elapsed_nn_last = nn.Linear(8, 1)
        torch.nn.init.zeros_(self.elapsed_nn_last.weight)
        torch.nn.init.zeros_(self.elapsed_nn_last.bias)
        self.elapsed_nn = model.ResBlock(
            nn.Sequential(
                nn.Linear(1, 8, bias=False),
                *[model.FFBlock(n_hidden=8, use_timeshift=False, dropout=0, use_checkpoint=True) for _ in range(4)],
                nn.LayerNorm(8),
                self.elapsed_nn_last,
            )
        )

    def forward(self, encoding_h, feature_elapsed_days_real_bl, feature_elapsed_days_int_bl, feature_rating_bl, label_elapsed_days_real_bl, label_elapsed_days_int_bl, log):
        if len(encoding_h.shape) == 2:
            encoding_h = encoding_h.mean(dim=0)
        """
        - last rating
        - first rating
        - longest succ
        - longest succ since last fail
        - time since last fail (feature pov)
        - # same-day fail
        - # non-same-day fail
        - # same-day pass
        - # non-same-day pass
        - # same-day reviews
        - # non-same-day reviews
        """
        B, L = feature_rating_bl.shape

        same_day = (feature_elapsed_days_real_bl < 1).float()
        feature_rating_is_hard = (feature_rating_bl == 2).float()
        feature_rating_is_better_than_hard = (feature_rating_bl > 2).float()
        feature_rating_onehot_bl4 = torch.nn.functional.one_hot((feature_rating_bl.long() - 1).clamp(min=0), num_classes=4).float()
        feature_rating_onehot_bl3 = (feature_rating_bl > 1).float().unsqueeze(-1) * torch.nn.functional.one_hot((feature_rating_bl.long() - 2).clamp(min=0), num_classes=3).float()
        success_bl = (feature_rating_bl > 1).float()
        is_first_review_bl = torch.zeros_like(feature_rating_bl)
        is_first_review_bl[:, 0] = 1
        first_rating_bl = feature_rating_bl[:, 0].unsqueeze(-1).expand(B, L)
        first_rating_onehot_bl4 = torch.nn.functional.one_hot((first_rating_bl.long() - 1).clamp(min=0), num_classes=4).float()
        first_rating_onehot_bl3 = (first_rating_bl > 1).float().unsqueeze(-1) * torch.nn.functional.one_hot((first_rating_bl.long() - 2).clamp(min=0), num_classes=3).float()
        first_rating_is_better_than_hard = (first_rating_bl > 2).float()

        # _first_idx = success_bl.int().argmax(dim=-1)
        # has_true = success_bl.any(dim=-1)
        # _first_idx = torch.where(has_true, _first_idx, torch.full_like(_first_idx, int(1e9)))
        # _first_idx = _first_idx.unsqueeze(-1).expand_as(success_bl)
        # first_success_idx = torch.minimum(_first_idx, 1 + torch.arange(L, device=encoding_h.device).expand(B, L))

        pass_streak = find_streak(success_bl)
        fail_streak = find_streak(1 - success_bl)
        streak_score = torch.sign(pass_streak - fail_streak) * torch.log(1 + torch.abs(pass_streak - fail_streak))
        # print()
        # print(pass_streak)
        # print(fail_streak)
        # print(streak_score)
        # fail_streak

        same_day_fail = same_day * (1 - is_first_review_bl) * (feature_rating_bl == 1.0).float()
        non_same_day_fail = (1 - same_day) * (1 - is_first_review_bl) * (feature_rating_bl == 1.0).float()
        num_same_day_fail = torch.cumsum(same_day_fail, dim=-1)
        num_non_same_day_fail = torch.cumsum(non_same_day_fail, dim=-1)
        same_day_pass = same_day * (1 - is_first_review_bl) * (feature_rating_bl > 1).float()
        non_same_day_pass = (1 - same_day) * (1 - is_first_review_bl) * (feature_rating_bl > 1).float()
        num_same_day_pass = torch.cumsum(same_day_pass, dim=-1)
        num_non_same_day_pass = torch.cumsum(non_same_day_pass, dim=-1)
        num_same_day = torch.cumsum(same_day, dim=-1)
        num_non_same_day = torch.cumsum(1.0 - same_day, dim=-1)
        num_pass = torch.cumsum((feature_rating_bl > 1).float(), dim=-1)
        num_fail = torch.cumsum((feature_rating_bl == 1).float(), dim=-1)
        has_passed = (num_pass > 0).float()

        elapsed_given_succ = torch.where(success_bl.bool(), feature_elapsed_days_real_bl, torch.zeros_like(success_bl))
        elapsed_given_succ_premax = torch.cummax(elapsed_given_succ, dim=-1).values

        cumulative_time = feature_elapsed_days_real_bl.cumsum(dim=-1)
        cumulative_time_only_succ = (feature_elapsed_days_real_bl * success_bl).cumsum(dim=-1)

        first_or_lapse_time = torch.where((is_first_review_bl.bool() | (1 - success_bl).bool()), cumulative_time, torch.zeros_like(cumulative_time))
        time_since_first_or_lapse = cumulative_time - torch.cummax(first_or_lapse_time, dim=-1).values

        label_is_same_day = (label_elapsed_days_int_bl == 0).float()
        
        deg_0_in_blh = torch.concat(
            (
                torch.ones_like(feature_rating_bl).unsqueeze(-1),
                feature_rating_onehot_bl3,
                first_rating_onehot_bl3,
                first_rating_onehot_bl3 * is_first_review_bl.unsqueeze(-1),  
                model.transform_elapsed_days_real(feature_elapsed_days_real_bl).unsqueeze(-1),
                has_passed.unsqueeze(-1),
                model.transform_elapsed_days_real(time_since_first_or_lapse).unsqueeze(-1),
                model.transform_elapsed_days_real(cumulative_time).unsqueeze(-1),
                model.transform_elapsed_days_real(cumulative_time_only_succ).unsqueeze(-1),
                ((first_rating_bl > 1) * (1 + num_fail).log()).unsqueeze(-1),
                ((first_rating_bl > 1) * (1 + num_pass).log()).unsqueeze(-1),
                (1 + num_fail).log().unsqueeze(-1),
                (1 + num_pass).log().unsqueeze(-1),
                streak_score.unsqueeze(-1),
                (1 + num_same_day_fail).log().unsqueeze(-1),
                (1 + num_non_same_day_fail).log().unsqueeze(-1),
                (1 + num_same_day_fail * label_is_same_day).log().unsqueeze(-1),
                (1 + num_non_same_day_fail * (1 - label_is_same_day)).log().unsqueeze(-1),
                (1 + num_same_day_pass).log().unsqueeze(-1),
                (1 + num_non_same_day_pass).log().unsqueeze(-1),
                (1 + num_same_day_pass * label_is_same_day).log().unsqueeze(-1),
                (1 + num_non_same_day_pass * (1 - label_is_same_day)).log().unsqueeze(-1),
                (1 + num_same_day).log().unsqueeze(-1),
                (1 + num_non_same_day).log().unsqueeze(-1),
                (1 + num_same_day * label_is_same_day).log().unsqueeze(-1),
                (1 + num_non_same_day * (1 - label_is_same_day)).log().unsqueeze(-1),
                label_is_same_day.unsqueeze(-1),
                ((first_rating_bl > 1) * (1 + num_same_day_fail).log()).unsqueeze(-1),
                ((first_rating_bl > 1) * (1 + num_non_same_day_fail).log()).unsqueeze(-1),
                ((first_rating_bl > 1) * (1 + num_same_day_pass).log()).unsqueeze(-1),
                ((first_rating_bl > 1) * (1 + num_non_same_day_pass).log()).unsqueeze(-1),
                ((first_rating_bl > 1) * (1 + num_same_day).log()).unsqueeze(-1),
                ((first_rating_bl > 1) * (1 + num_non_same_day).log()).unsqueeze(-1),
            ),
            dim=-1
        )
        # all_blh = torch.concat(
        #     (
        #         # torch.ones_like(feature_rating_bl).unsqueeze(-1),
        #         feature_rating_onehot_bl4,
        #         first_rating_onehot_bl4,
        #         first_rating_onehot_bl4 * is_first_review_bl.unsqueeze(-1),
        #         model.transform_elapsed_days_real(feature_elapsed_days_real_bl).unsqueeze(-1),
        #         is_first_review_bl.unsqueeze(-1),
        #         (1 + num_same_day_fail).log().unsqueeze(-1),
        #         (1 + num_non_same_day_fail).log().unsqueeze(-1),
        #         (1 + num_same_day_fail * label_is_same_day).log().unsqueeze(-1),
        #         (1 + num_non_same_day_fail * (1 - label_is_same_day)).log().unsqueeze(-1),
        #         (1 + num_same_day_pass).log().unsqueeze(-1),
        #         (1 + num_non_same_day_pass).log().unsqueeze(-1),
        #         (1 + num_same_day_pass * label_is_same_day).log().unsqueeze(-1),
        #         (1 + num_non_same_day_pass * (1 - label_is_same_day)).log().unsqueeze(-1),
        #         (1 + num_same_day).log().unsqueeze(-1),
        #         (1 + num_non_same_day).log().unsqueeze(-1),
        #         (1 + num_same_day * label_is_same_day).log().unsqueeze(-1),
        #         (1 + num_non_same_day * (1 - label_is_same_day)).log().unsqueeze(-1),
        #         has_passed.unsqueeze(-1),
        #         model.transform_elapsed_days_real(elapsed_given_succ_premax).unsqueeze(-1),
        #         model.transform_elapsed_days_real(time_since_first_or_lapse).unsqueeze(-1),
        #         label_is_same_day.unsqueeze(-1),
        #         model.transform_elapsed_days_real(label_elapsed_days_real_bl).unsqueeze(-1),
        #         model.transform_elapsed_days_real(times).unsqueeze(-1),
        #         ((first_rating_bl > 1) * (1 + num_same_day_fail).log()).unsqueeze(-1),
        #         ((first_rating_bl > 1) * (1 + num_non_same_day_fail).log()).unsqueeze(-1),
        #         ((first_rating_bl > 1) * (1 + num_same_day).log()).unsqueeze(-1),
        #         ((first_rating_bl > 1) * (1 + num_non_same_day).log()).unsqueeze(-1),
        #         streak_score.unsqueeze(-1),
        #         # (1 + pass_streak).log().unsqueeze(-1),
        #         # (1 + fail_streak).log().unsqueeze(-1),
        #         # (1 + first_success_idx).log().unsqueeze(-1)
        #     ),
        #     dim=-1
        # )
        v_bl = model.transform_elapsed_days_real(label_elapsed_days_real_bl)
        # v_bl = self.elapsed_nn(v_bl.unsqueeze(-1)).squeeze(-1)
        factor_H = self.enc_scale(encoding_h) * self.scale.weight.squeeze(0)
        if "factor" in log:
            log["factor"].append(factor_H)
        y_blH = torch.cat(
            (
                deg_0_in_blh,
                # self.feature_lin(all_blh).clamp(min=0, max=1),
            )
            , dim=-1
        )
        s_bl = torch.einsum('blh,h->bl', y_blH, factor_H).exp().clamp(1e-8)
        curve_mins = torch.tensor([
            0.01,    # 27
            0.01,    # 28 (depends on w27)
            0.5,     # 29
            0.5,     # 30 (depends on w29)
            0.01,    # 31
            0.01,     # 32
            0.1,     # 33
            0.1      # 34
        ], device=s_bl.device)

        curve_maxs = torch.tensor([
            0.95,   # 27
            0.95,   # 28
            0.99,   # 29
            0.99,   # 30
            1.0,    # 31
            1.0,    # 32
            1.0,    # 33
            1.0     # 34
        ], device=s_bl.device)

        def _bounded(x, lo, hi, default):
            mid = torch.log((default - lo) / (hi - default))
            return lo + (hi - lo) * torch.sigmoid(x + mid)

        curve_h = _bounded(self.enc_curve(encoding_h), curve_mins, curve_maxs, (curve_mins + curve_maxs) / 2)
        decay, base, base_weight, s_exp = curve_h.chunk(4, dim=-1)

        t_over_s = (label_elapsed_days_real_bl / s_bl).unsqueeze(-1)
        factor = base ** (1 / -decay) - 1
        R = (1 + factor * t_over_s) ** -decay
        weight = base_weight * s_bl.unsqueeze(-1) ** s_exp
        return self.to_four_pred(1e-5 + (1 - 2e-5) * ((weight * R).sum(-1) / weight.sum(-1)))



        return self.to_four_pred(torch.sigmoid(out_bl))

        in_blh = torch.concat(
            (
                torch.ones_like(feature_rating_bl).unsqueeze(-1),
                # 1
                feature_rating_onehot_bl4,
                # 5
                first_rating_onehot_bl4,
                # 9
                first_rating_onehot_bl4 * is_first_review_bl.unsqueeze(-1),
                # 13
                model.transform_elapsed_days_real(feature_elapsed_days_real_bl).unsqueeze(-1),
                is_first_review_bl.unsqueeze(-1),
                (1 + num_same_day_fail).log().unsqueeze(-1),
                # 16
                (1 + num_non_same_day_fail).log().unsqueeze(-1),
                (1 + num_same_day_fail * label_is_same_day).log().unsqueeze(-1),
                (1 + num_non_same_day_fail * (1 - label_is_same_day)).log().unsqueeze(-1),
                # 19
                (1 + num_same_day_pass).log().unsqueeze(-1),
                (1 + num_non_same_day_pass).log().unsqueeze(-1),
                (1 + num_same_day_pass * label_is_same_day).log().unsqueeze(-1),
                # 22
                (1 + num_non_same_day_pass * (1 - label_is_same_day)).log().unsqueeze(-1),
                (1 + num_same_day).log().unsqueeze(-1),
                (1 + num_non_same_day).log().unsqueeze(-1),
                # 25
                (1 + num_same_day * label_is_same_day).log().unsqueeze(-1),
                (1 + num_non_same_day * (1 - label_is_same_day)).log().unsqueeze(-1),
                has_passed.unsqueeze(-1),
                # 28
                model.transform_elapsed_days_real(elapsed_given_succ_premax).unsqueeze(-1),
                model.transform_elapsed_days_real(time_since_first_or_lapse).unsqueeze(-1),
                label_is_same_day.unsqueeze(-1),
                # 31
                model.transform_elapsed_days_real(label_elapsed_days_real_bl).unsqueeze(-1),
            ),
            dim=-1
        )

    def to_four_pred(self, x_bl):
        B, L = x_bl.shape
        eps = 1e-6
        x_bl4 = torch.full((B, L, 4), float("-inf"),
                        device=x_bl.device, dtype=x_bl.dtype)
        x_bl4[..., 0] = torch.log((1 - x_bl).clamp_min(eps))
        x_bl4[..., 2] = torch.log(x_bl.clamp_min(eps))
        return x_bl4


class FirstReviewModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.pred = nn.Parameter(torch.tensor([0.25, 0.25, 0.25, 0.25]))

    def forward(self, encoding_bh):
        return self.pred

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_encoding = model.N_ENCODING
        self.encoder_model = model.EncoderModel(n_encoding=self.n_encoding)
        self.card_model = CardModel(n_encoding=self.n_encoding)
        self.first_review_model = FirstReviewModel(n_encoding=self.n_encoding)

    def get_excluded_params(self):
        return [] 

    def is_copy_exclude_param(self, x):
        start_layer = 6
        if "global_encoder" not in x:
            return False
        if "global_encoder.norm" in x or "value_linear" in x or "weight_linear" in x or "in_norm" in x:
            return False
        vote = 0
        for ex_l in range(start_layer):
            if f"core.layers.{ex_l}." in x:
                vote += 1
        for ex_l in range(start_layer - 1):
            if x.endswith(f"core.queries.{ex_l}"):
                vote += 1
            if f"core.norms.{ex_l}." in x:
                vote += 1
        return vote == 0
    
    def is_frozen_param(self, x):
        exclude = ["fsrs_linear", "curve_nn_encode", "fcnn", "card_model"]
        for s in exclude:
            if s in x:
                return False
        return not self.is_copy_exclude_param(x)


    