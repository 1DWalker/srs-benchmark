from summary import fsrs_7, fsrs_7_custom, fsrs_7_truth, model
import torch
import torch.nn as nn
import numpy as np

class Config:
    def __init__(self):
        self.s_min = 0.0001
        self.init_s_max = 100
        self.use_secs_intervals = True
        self.device = torch.device("cuda")

FSRS = fsrs_7

class CardModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.d_transition_model = DTransitionModel(n_encoding=n_encoding)
        self.fsrs_linear = nn.Linear(n_encoding, 35)
        torch.nn.init.zeros_(self.fsrs_linear.weight)
        torch.nn.init.zeros_(self.fsrs_linear.bias)
        with torch.no_grad():
            self.fsrs_linear.bias[1:4].copy_(torch.tensor([-2.2, -0.5, -0.5]))

    def encoding_to_fsrs(self, encoding_h):
        x = self.fsrs_linear(encoding_h)
        print(x)
        fsrs_params = FSRS.nn_vec_to_fsrs7_params(x)
        print(fsrs_params, fsrs_params.shape)
        return fsrs_params

    def forward(self, encoding_h, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_real_bl):
        if len(encoding_h.shape) == 2:
            encoding_h = encoding_h.mean(dim=0)
        fsrs_params = self.encoding_to_fsrs(encoding_h)
        B, L = feature_elapsed_days_real_bl.shape

        # self.d_transition_model.set_encoding(encoding_h=encoding_h)
        # def d_transition(
        #     s_b,
        #     d_b,
        #     feature_elapsed_days_real_b, 
        #     feature_rating_b
        # ):
        #     return self.d_transition_model(None, s_b, d_b, feature_elapsed_days_real_b, feature_rating_b)

        x_bl, state_bl2 = FSRS.forward(
            parameters_p=fsrs_params, 
            feature_elapsed_days_real_bl=feature_elapsed_days_real_bl,
            feature_rating_bl=feature_rating_bl,   
            label_elapsed_days_real_bl=label_elapsed_days_real_bl,
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

class SDRatingElapsedModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.n_hidden = 8
        self.n_blocks = 4
        self.encode = nn.Linear(7, self.n_hidden)
        self.blocks = nn.ModuleList(
            [model.FFBlockWithEncoder(n_hidden=self.n_hidden, n_encoding=n_encoding, use_timeshift=False, dropout=0) for _ in range(self.n_blocks)]
        )
        self.last = nn.Sequential(
            nn.LayerNorm(self.n_hidden),
            nn.Linear(self.n_hidden, 1),
        )
    
    def set_encoding(self, encoding_h):
        for block in self.blocks:
            block.set_encoding(encoding_h)

    def forward(
            self, 
            encoding_h,
            s_b,
            d_b,
            feature_elapsed_days_real_b, 
            feature_rating_b, 
            ):
        B = feature_elapsed_days_real_b.size(0)
        feature_rating_onehot_b4 = torch.nn.functional.one_hot((feature_rating_b.long() - 1).clamp(min=0), num_classes=4).float()
        x = torch.cat(
            (
                transform_elapsed_days_real(s_b).unsqueeze(-1),
                torch.log(11 - d_b).unsqueeze(-1),
                transform_elapsed_days_real(feature_elapsed_days_real_b).unsqueeze(-1), 
                feature_rating_onehot_b4,
            ), 
            dim=-1,
        )
        x = self.encode(x)
        for block in self.blocks:
            x = block(x, encoding_h)
        return self.last(x)

class DTransitionModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.core = SDRatingElapsedModel(n_encoding=n_encoding)

    def set_encoding(self, encoding_h):
        self.core.set_encoding(encoding_h)

    def forward(
            self, 
            encoding_h,
            s_b,
            d_b,
            feature_elapsed_days_real_b, 
            feature_rating_b, 
            ):
        x = self.core(encoding_h, s_b, d_b, feature_elapsed_days_real_b, feature_rating_b)
        return 1 + 9 * torch.sigmoid(x).squeeze(-1)

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