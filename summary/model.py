import torch
import torch.nn as nn
import numpy as np

BASE_DROPOUT = 0
FORGETTING_CURVE_DROPOUT = 1 - (1 - BASE_DROPOUT) ** 2
FIRST_REVIEW_DROPOUT = 0.5
GLOBAL_ENCODER_DROPOUT = 0.5

ENCODER_N_HIDDEN = 16
N_ENCODING = 24
FORGETTING_CURVE_N_LAYERS = 4

def transform_elapsed_days_real(x):
    return ((x + 1e-5).log() + 1.3) / 5

def transform_card_num_reviews(x):
    return (x + 1).clamp(max=64).log()

def transform_global_n_reviews(x):
    return (x + 1).clamp(max=100000).log()

class TimeShiftLerp(torch.nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))
        self.lerp = torch.nn.Parameter(torch.ones((1, 1, n_hidden)))
    
    def forward(self, x_blh):
        x_shift = self.time_shift(x_blh)
        return torch.lerp(x_shift, x_blh, self.lerp)

class ResBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class RNNWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs)[0]

class RNNBlock(torch.nn.Module):
    def __init__(self, n_hidden, dropout=0):
        super().__init__()

        self.seq = ResBlock(nn.Sequential(
            nn.LayerNorm(n_hidden),
            TimeShiftLerp(n_hidden=n_hidden),
            RNNWrapper(nn.LSTM(input_size=n_hidden, hidden_size=n_hidden, batch_first=True)),
            nn.Linear(n_hidden, n_hidden),
            *[nn.Dropout(p=dropout) for _ in range(1 if dropout > 0 else 0)],
        ))
        for name, param in self.named_parameters():
            if "weight_ih" in name:  # Input-to-hidden weights
                nn.init.orthogonal_(param.data)
            elif "weight_hh" in name:  # Hidden-to-hidden weights
                nn.init.orthogonal_(param.data)
            elif "bias_ih" in name:  # Biases
                start_index = len(param.data) // 4
                end_index = len(param.data) // 2
                param.data[start_index:end_index].fill_(1.0)

    def forward(self, x):
        return self.seq(x)

class FFBlock(torch.nn.Module):
    def __init__(self, n_hidden, use_timeshift, dropout=0):
        super().__init__()
        self.seq = ResBlock(nn.Sequential(
            nn.LayerNorm(n_hidden),
            *[TimeShiftLerp(n_hidden=n_hidden) for _ in range(1 if use_timeshift else 0)],
            nn.Linear(n_hidden, n_hidden),
            nn.Mish(),
            nn.Linear(n_hidden, n_hidden),
            *[nn.Dropout(p=dropout) for _ in range(1 if dropout > 0 else 0)],
        ))

    def forward(self, x):
        return self.seq(x)

class Block(torch.nn.Module):
    def __init__(self, n_hidden, dropout=0):
        super().__init__()
        self.seq = nn.Sequential(
            RNNBlock(n_hidden=n_hidden, dropout=dropout),
            FFBlock(n_hidden=n_hidden, use_timeshift=True, dropout=dropout),
        )

    def forward(self, x):
        return self.seq(x)

class GlobalEncoder(torch.nn.Module):
    def __init__(self, n_in, n_out, return_accept_weights=False):
        super().__init__()
        self.n_hidden = 4 * n_in
        self.norm = nn.LayerNorm(n_in)
        self.value_linear = nn.Linear(n_in, self.n_hidden, bias=False)
        self.weight_linear = nn.Linear(n_in, self.n_hidden)
        self.return_accept_weights = return_accept_weights
        if self.return_accept_weights:
            self.accept_linear = nn.Linear(n_in, n_out)
            torch.nn.init.zeros_(self.accept_linear.weight)
            torch.nn.init.zeros_(self.accept_linear.bias)

        self.core = nn.Sequential(
            *[FFBlock(self.n_hidden, use_timeshift=False, dropout=GLOBAL_ENCODER_DROPOUT) for _ in range(3)],
            nn.LayerNorm(self.n_hidden),
            nn.Linear(self.n_hidden, n_out),
        )
    
    def forward(self, x_list, mask_list):
        accum_weighted_value_h = 0
        accum_weight_h = 0
        accept_weights_list = []
        for x_bl, mask_bl in zip(x_list, mask_list):
            assert len(x_bl.shape) == 3
            x_bl = self.norm(x_bl)
            value_blh = self.value_linear(x_bl)
            weight_blh = torch.nn.functional.softplus(self.weight_linear(x_bl))
            if self.return_accept_weights:
                accept_weights_list.append(self.accept_linear(x_bl))

            eff_weight_blh = mask_bl.unsqueeze(-1).float() * weight_blh
            accum_weighted_value_h += (eff_weight_blh * value_blh).sum(dim=(0, 1))
            accum_weight_h += eff_weight_blh.sum(dim=(0, 1))

        sum_encoding_h = accum_weighted_value_h / (accum_weight_h + 1e-6)
        if self.return_accept_weights:
            return self.core(sum_encoding_h), accept_weights_list
        else:
            return self.core(sum_encoding_h)


class EncoderModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.n_features = 9
        self.n_hidden = ENCODER_N_HIDDEN
        self.n_curves = 3
        self.n_encoding = n_encoding
        self.dropout = BASE_DROPOUT

        self.encode_block = nn.Sequential(
            nn.Linear(self.n_features, self.n_hidden),
            Block(self.n_hidden, dropout=self.dropout),
        )
        self.full_blocks = 2  # 1 full block consists of GLOBAL - FF - LSTM - FF
        self.intermediate_global_encoders = nn.ModuleList([GlobalEncoder(n_in=self.n_hidden, n_out=self.n_hidden, return_accept_weights=True) for _ in range(self.full_blocks)])
        self.intermediate_sequentials = nn.ModuleList([
            nn.Sequential(
                FFBlock(self.n_hidden, use_timeshift=True, dropout=self.dropout),
                Block(self.n_hidden, dropout=self.dropout),
            )
            for _ in range(self.full_blocks)
        ])
        self.last_global_encoder = GlobalEncoder(n_in=self.n_hidden, n_out=self.n_encoding, return_accept_weights=False)

    def first(
            self, 
            feature_elapsed_days_real_bl, 
            feature_rating_bl, 
            card_review_th_ratio_bl, 
            global_num_reviews, 
            card_i_ratio_bl, 
            card_num_reviews_bl):
        B, L = feature_elapsed_days_real_bl.shape
        feature_rating_onehot_bl4 = torch.nn.functional.one_hot((feature_rating_bl.long() - 1).clamp(min=0), num_classes=4).float()
        x = torch.cat(
            (
                transform_elapsed_days_real(feature_elapsed_days_real_bl).unsqueeze(-1), 
                feature_rating_onehot_bl4,
                card_review_th_ratio_bl.unsqueeze(-1), 
                transform_global_n_reviews(global_num_reviews).view(1, 1, 1).expand(B, L, 1),
                card_i_ratio_bl.unsqueeze(-1), 
                transform_card_num_reviews(card_num_reviews_bl).unsqueeze(-1),
            ), 
            dim=-1,
        )
        return self.encode_block(x)

    def run_core(self, x_list, mask_list):
        for global_encoder, sequential in zip(self.intermediate_global_encoders, self.intermediate_sequentials):
            new_x_list = []
            global_encoding, accept_weights_list = global_encoder(x_list, mask_list)
            for x, accept in zip(x_list, accept_weights_list):
                new_x_list.append(sequential(x + accept * global_encoding))
            x_list = new_x_list
        
        return self.last_global_encoder(x_list, mask_list)

class ForgettingCurveNN(torch.nn.Module):
    def __init__(self, n_input, dropout):
        super().__init__()
        self.n_hidden = n_input        
        self.n_layers = FORGETTING_CURVE_N_LAYERS
        self.core = nn.Sequential(
            *[FFBlock(self.n_hidden, use_timeshift=False, dropout=FORGETTING_CURVE_DROPOUT) for _ in range(self.n_layers)],
            nn.LayerNorm(self.n_hidden),
        )

        self.forgetting_curve_last_linear = nn.Linear(self.n_hidden, 4)
        with torch.no_grad():
            self.forgetting_curve_last_linear.bias.data.copy_(torch.tensor([-0.1645, -0.0393,  0.3989, -0.2395]))

    def forward(self, x, label_elapsed_days_real_bl):
        x = torch.cat([x, transform_elapsed_days_real(label_elapsed_days_real_bl).unsqueeze(-1)], dim=-1)
        x = self.core(x)
        x = self.forgetting_curve_last_linear(x)
        return x


class CardModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.n_features = 5 + n_encoding
        self.n_hidden = n_encoding
        self.n_blocks = 2
        self.n_encoding = n_encoding
        self.dropout = BASE_DROPOUT
        self.encoder = nn.Linear(self.n_features, self.n_hidden)
        self.blocks = nn.ModuleList([Block(self.n_hidden, dropout=self.dropout) for _ in range(self.n_blocks)])
        self.last_rnn_block = RNNBlock(self.n_hidden, dropout=self.dropout)
        self.transition = nn.Sequential(
            nn.LayerNorm(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden - 1),
        )
        self.forgetting_curve_nn = ForgettingCurveNN(self.n_hidden, self.dropout)

    def forward(self, encoding_bh, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_real_bl):
        B, L = feature_elapsed_days_real_bl.shape
        H = encoding_bh.size(1)
        feature_rating_onehot_bl4 = torch.nn.functional.one_hot((feature_rating_bl.long() - 1).clamp(min=0), num_classes=4).float()
        x = torch.cat(
            (
                transform_elapsed_days_real(feature_elapsed_days_real_bl).unsqueeze(-1),  # [B, L, 1]
                feature_rating_onehot_bl4,                                                        # [B, L, 4]
                encoding_bh.view(B, 1, H).expand(B, L, H),                         # [B, L, H]
            ),
            dim=-1,
        )
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.last_rnn_block(x)
        x = self.transition(x)
        return self.forgetting_curve_nn(x, label_elapsed_days_real_bl)

class FirstReviewModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.n_layers = 2
        self.n_hidden = n_encoding
        self.first_review_last_linear = nn.Linear(self.n_hidden, 4)
        self.core = nn.Sequential(
            *[FFBlock(self.n_hidden, use_timeshift=False, dropout=FIRST_REVIEW_DROPOUT) for _ in range(self.n_layers)],
            nn.LayerNorm(self.n_hidden),
        )

    def forward(self, encoding_bh):
        return self.first_review_last_linear(self.core(encoding_bh))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_encoding = N_ENCODING
        self.encoder_model = EncoderModel(n_encoding=self.n_encoding)
        self.card_model = CardModel(n_encoding=self.n_encoding)
        self.first_review_model = FirstReviewModel(n_encoding=self.n_encoding)