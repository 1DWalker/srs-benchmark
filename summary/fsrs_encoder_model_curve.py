from summary import fsrs_7, fsrs_7_curve_nn, fsrs_7_custom, fsrs_7_truth, model
import torch
import torch.nn as nn
import numpy as np

class Config:
    def __init__(self):
        self.s_min = 0.0001
        self.init_s_max = 100
        self.use_secs_intervals = True
        self.device = torch.device("cuda")

FSRS = fsrs_7_curve_nn

class CardModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.fsrs_linear = nn.Linear(n_encoding, 35)
        torch.nn.init.zeros_(self.fsrs_linear.weight)
        torch.nn.init.zeros_(self.fsrs_linear.bias)
        with torch.no_grad():
            self.fsrs_linear.bias[1:4].copy_(torch.tensor([-2.2, -0.5, -0.5]))
        self.curve_n_hidden = 18
        self.curve_nn_encode = nn.Linear(n_encoding, self.curve_n_hidden, bias=False)
        self.fcnn = FCNN(self.curve_n_hidden)

    def print_curve(self, encoding_h):
        if len(encoding_h.shape) == 2:
            encoding_h = encoding_h.mean(dim=0)
        curve_encoding_h = self.curve_nn_encode(encoding_h)
        MINUTE = 1 / 24 / 60
        S = torch.tensor([60 * MINUTE, 1.0, 30], device=encoding_h.device)
        D = torch.tensor([1.0, 7.0, 10.0], device=encoding_h.device)
        grid_s, grid_d = torch.meshgrid(S, D, indexing='ij')
        decay, base, weight, swp = self.fcnn.get_curve(curve_encoding_h, grid_s, grid_d)
        for j, d in enumerate(D):
            print(f"\n=== D = {d.item():.3f} ===")
            for i, s in enumerate(S):
                print(f"  S = {s.item():.3f}")
                print(f"    Decay : {[f'{x:.3f}' for x in decay[i, j].tolist()]}")
                print(f"    Base  : {[f'{x:.3f}' for x in base[i, j].tolist()]}")
                print(f"    Weight: {[f'{x:.3f}' for x in weight[i, j].tolist()]}")
                print(f"    Swp   : {[f'{x:.3f}' for x in swp[i, j].tolist()]}")

    def encoding_to_fsrs(self, encoding_h):
        if len(encoding_h.shape) == 2:
            encoding_h = encoding_h.mean(dim=0)
        x = self.fsrs_linear(encoding_h)
        fsrs_params = FSRS.nn_vec_to_fsrs7_params(x)
        return fsrs_params

    def forward(self, encoding_h, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_real_bl):
        if len(encoding_h.shape) == 2:
            encoding_h = encoding_h.mean(dim=0)
        fsrs_params = self.encoding_to_fsrs(encoding_h)
        B, L = feature_elapsed_days_real_bl.shape
        assert len(encoding_h.shape) == 1
        curve_encoding_h = self.curve_nn_encode(encoding_h)

        def nn_curve(
            elapsed_b,
            s_b,
            d_b,
        ):
            return self.fcnn(curve_encoding_h, elapsed_b, s_b, d_b)

        x_bl, state_bl2 = FSRS.forward(
            parameters_p=fsrs_params, 
            feature_elapsed_days_real_bl=feature_elapsed_days_real_bl,
            feature_rating_bl=feature_rating_bl,   
            label_elapsed_days_real_bl=label_elapsed_days_real_bl,
            curve_nn = nn_curve
        )

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

class FCNN(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.n_curves = 2
        self.n_hidden = n_hidden
        self.encode = nn.Linear(2, self.n_hidden, bias=False)
        self.n_blocks = 3
        self.core = nn.Sequential(
            *[model.FFBlock(n_hidden=self.n_hidden, use_timeshift=False, dropout=0, use_checkpoint=False) for _ in range(self.n_blocks)],
            nn.LayerNorm(self.n_hidden),
        )
        self.proj = nn.Linear(self.n_hidden, 4 * self.n_curves)
        torch.nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.weight.multiply_(0.1)

    def bounded_exp(self, x, lo, hi, default):
        lo_log = np.log(lo)
        hi_log = np.log(hi)
        def_log = np.log(default)
        mid = np.log((def_log - lo_log) / (hi_log - def_log))
        out_log = lo_log + (hi_log - lo_log) * torch.sigmoid(x + mid)
        return torch.exp(out_log)
    # def bounded_exp(self, x, lo, hi, default):
    #     lo_log = np.log(lo)
    #     hi_log = np.log(hi)
    #     def_log = torch.log(default)
    #     mid = torch.log((def_log - lo_log) / (hi_log - def_log))
    #     out_log = lo_log + (hi_log - lo_log) * torch.sigmoid(x + mid)
    #     return torch.exp(out_log)

    def get_curve(
            self, 
            encoding_bh,
            s_b,
            d_b,
        ):
        x_b2 = torch.cat(
            (
                model.transform_elapsed_days_real(s_b).unsqueeze(-1),
                torch.log(11 - d_b).unsqueeze(-1),
            ), 
            dim=-1,
        )
        x_bh = self.encode(x_b2) + encoding_bh
        x_bh = self.core(x_bh)
        out = self.proj(x_bh)
        decay, base, weight, swp = out.chunk(4, dim=-1)
        """
            w[27] = w[27].clamp(0.01, 0.25)  # decay 1
            w[28] = w[28].clamp(w[27], 0.95)  # decay 2
            w[29] = w[29].clamp(0.5, 0.85)  # base 1
            w[30] = w[30].clamp(w[29], 0.99)  # base 2
            w[31] = w[31].clamp(0.01, 1)  # weight 1
            w[32] = w[32].clamp(0.1, 1)  # weight 2
            w[33] = w[33].clamp(0, 0.9)  # S weight power 1
            w[34] = w[34].clamp(0.1, 1.1)  # S weight power 2
        """
        device = x_bh.device
        # decay = self.bounded_exp(decay, 0.01, 0.95, torch.tensor([0.0723, 0.1634], device=device))
        # base = self.bounded_exp(base, 0.1, 0.99, torch.tensor([0.5, 0.95], device=device))
        # weight = self.bounded_exp(weight, 0.01, 1, torch.tensor([0.22, 0.62], device=device))
        # swp = self.bounded_exp(swp, 0.1, 1.1, torch.tensor([0.13, 0.38], device=device))
        decay = self.bounded_exp(decay, 0.01, 0.95, 0.2)
        base = self.bounded_exp(base, 0.1, 0.99, 0.9)
        weight = self.bounded_exp(weight, 0.01, 1, 0.5)
        swp = self.bounded_exp(swp, 0.1, 1.1, 0.5)
        return decay, base, weight, swp

    def forward(
            self, 
            encoding_bh,
            t_b,
            s_b,
            d_b,
        ):
        decay, base, base_weight, swp = self.get_curve(encoding_bh, s_b, d_b)
        t_over_s = (t_b.clamp_min(1e-5) / s_b).unsqueeze(-1)
        factor = base ** (1 / -decay) - 1
        R = (1 + factor * t_over_s) ** -decay
        weight = base_weight * s_b.unsqueeze(-1) ** swp
        return 1e-5 + (1 - 2e-5) * ((weight * R).sum(-1) / weight.sum(-1))



# class SDRatingElapsedModel(torch.nn.Module):
#     def __init__(self, n_encoding):
#         super().__init__()
#         self.n_hidden = 8
#         self.n_blocks = 4
#         self.encode = nn.Linear(7, self.n_hidden)
#         self.blocks = nn.ModuleList(
#             [model.FFBlockWithEncoder(n_hidden=self.n_hidden, n_encoding=n_encoding, use_timeshift=False, dropout=0) for _ in range(self.n_blocks)]
#         )
#         self.last = nn.Sequential(
#             nn.LayerNorm(self.n_hidden),
#             nn.Linear(self.n_hidden, 1),
#         )
    
#     def forward(
#             self, 
#             encoding_h,
#             s_b,
#             d_b,
#             feature_elapsed_days_real_b, 
#             feature_rating_b, 
#             ):
#         B = feature_elapsed_days_real_b.size(0)
#         feature_rating_onehot_b4 = torch.nn.functional.one_hot((feature_rating_b.long() - 1).clamp(min=0), num_classes=4).float()
#         x = torch.cat(
#             (
#                 transform_elapsed_days_real(s_b).unsqueeze(-1),
#                 torch.log(11 - d_b).unsqueeze(-1),
#                 transform_elapsed_days_real(feature_elapsed_days_real_b).unsqueeze(-1), 
#                 feature_rating_onehot_b4,
#             ), 
#             dim=-1,
#         )
#         x = self.encode(x)
#         for block in self.blocks:
#             x = block(x)
#         return self.last(x)

# class DTransitionModel(torch.nn.Module):
#     def __init__(self, n_encoding):
#         super().__init__()
#         self.core = SDRatingElapsedModel(n_encoding=n_encoding)

#     def set_encoding(self, encoding_h):
#         self.core.set_encoding(encoding_h)

#     def forward(
#             self, 
#             encoding_h,
#             s_b,
#             d_b,
#             feature_elapsed_days_real_b, 
#             feature_rating_b, 
#             ):
#         x = self.core(encoding_h, s_b, d_b, feature_elapsed_days_real_b, feature_rating_b)
#         return 1 + 9 * torch.sigmoid(x).squeeze(-1)

# class DInitModel(torch.nn.Module):
#     def __init__(self, n_encoding):
#         super().__init__()
#         self.linear = nn.Linear(n_encoding, 4)

#     def forward(
#             self, 
#             s_b,
#             d_b,
#             feature_elapsed_days_real_b, 
#             feature_rating_b, 
#             ):
#         x = self.core(s_b, d_b, feature_elapsed_days_real_b, feature_rating_b)
#         return 1 + 9 * torch.sigmoid(x)

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


    