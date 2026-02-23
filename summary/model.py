import torch
import torch.nn as nn
import numpy as np

BASE_DROPOUT = 0
FORGETTING_CURVE_DROPOUT = 1 - (1 - BASE_DROPOUT) ** 2
FIRST_REVIEW_DROPOUT = 0
GLOBAL_ENCODER_DROPOUT = 0
GLOBAL_ENCODER_CHANNEL_DROPOUT = 0.01
LAST_GLOBAL_NN_DROPOUT = 0

ENCODER_N_HIDDEN = 16
DECODER_N_HIDDEN = 24
FORGETTING_CURVE_N_LAYERS = 4
GLOBAL_FACTOR = 4
ENCODER_FULL_BLOCKS = 2
N_ENCODING = ENCODER_N_HIDDEN * GLOBAL_FACTOR * (ENCODER_FULL_BLOCKS + 1)

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
        self.lerp = torch.nn.Parameter(torch.ones((n_hidden,)))
    
    def forward(self, x_blh):
        x_shift = self.time_shift(x_blh)
        return torch.lerp(x_shift, x_blh, self.lerp.view(1, 1, -1))

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
    def __init__(self, n_hidden, use_timeshift, dropout=0, dropout_channel=0):
        super().__init__()
        self.seq = ResBlock(nn.Sequential(
            nn.LayerNorm(n_hidden),
            *[TimeShiftLerp(n_hidden=n_hidden) for _ in range(1 if use_timeshift else 0)],
            nn.Linear(n_hidden, n_hidden),
            nn.Mish(),
            nn.Linear(n_hidden, n_hidden),
            *[nn.Dropout(p=dropout) for _ in range(1 if dropout > 0 else 0)],
        ))
        self.dropout_channel = nn.Dropout(p=dropout_channel)

    def forward(self, x):
        return self.dropout_channel(self.seq(x))

class Block(torch.nn.Module):
    def __init__(self, n_hidden, dropout=0):
        super().__init__()
        self.seq = nn.Sequential(
            RNNBlock(n_hidden=n_hidden, dropout=dropout),
            FFBlock(n_hidden=n_hidden, use_timeshift=True, dropout=dropout),
        )

    def forward(self, x):
        return self.seq(x)

class FFBlockWithEncoder(torch.nn.Module):
    def __init__(self, n_hidden, n_encoding, use_timeshift, dropout=0):
        super().__init__()
        self.encoding_linear = nn.Linear(n_encoding, n_hidden, bias=False)
        torch.nn.init.zeros_(self.encoding_linear.weight)

        self.A = nn.Sequential(
            nn.LayerNorm(n_hidden),
            *[TimeShiftLerp(n_hidden=n_hidden) for _ in range(1 if use_timeshift else 0)],
            nn.Linear(n_hidden, n_hidden),
        )
        self.act = nn.Mish()
        self.B = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            *[nn.Dropout(p=dropout) for _ in range(1 if dropout > 0 else 0)],
        )

    def forward(self, x, encoding_h):
        x_start = x
        x = self.A(x)
        x = self.act(x + self.encoding_linear(encoding_h))
        x = self.B(x)
        return x_start + x

class BlockWithEncoder(torch.nn.Module):
    def __init__(self, n_hidden, n_encoding, dropout=0):
        super().__init__()
        self.rnn = RNNBlock(n_hidden=n_hidden, dropout=dropout)
        self.ff = FFBlockWithEncoder(n_hidden=n_hidden, n_encoding=n_encoding, use_timeshift=True, dropout=dropout)

    def forward(self, x, encoding_h):
        x = self.rnn(x)
        x = self.ff(x, encoding_h=encoding_h)
        return x


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
        self.full_blocks = ENCODER_FULL_BLOCKS # 1 full block consists of GLOBAL - FF - LSTM - FF
        self.intermediate_global_encoders = nn.ModuleList(
            [GlobalEncoder(n_in=self.n_hidden, n_out=self.n_hidden, n_hidden_in=GLOBAL_FACTOR * i * self.n_hidden, intermediate=True) for i in range(self.full_blocks)]
        )
        self.intermediate_ffs = nn.ModuleList([
            FFBlockWithEncoder(self.n_hidden, n_encoding=GLOBAL_FACTOR * (i + 1) * self.n_hidden, use_timeshift=True, dropout=self.dropout)
            for i in range(self.full_blocks)
        ])
        self.intermediate_blocks = nn.ModuleList([
            BlockWithEncoder(self.n_hidden, n_encoding=GLOBAL_FACTOR * (i + 1) * self.n_hidden, dropout=self.dropout)
            for i in range(self.full_blocks)
        ])
        self.last_global_encoder = GlobalEncoder(n_in=self.n_hidden, n_out=self.n_hidden, n_hidden_in=GLOBAL_FACTOR * self.full_blocks * self.n_hidden, num_blocks=10, intermediate=False)

        self.n_full_global = (self.full_blocks + 1) * self.n_hidden * GLOBAL_FACTOR

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
        global_hidden = torch.tensor([], device=x_list[0].device)
        for global_encoder, ff, block in zip(self.intermediate_global_encoders, self.intermediate_ffs, self.intermediate_blocks):
            new_x_list = []
            new_global_hidden, inter_contrib, accept_weights_list = global_encoder(x_list, mask_list, h_in=global_hidden)
            global_hidden = new_global_hidden
            for x, accept in zip(x_list, accept_weights_list):
                x = x + accept * inter_contrib
                x = ff(x, encoding_h=global_hidden)
                x = block(x, encoding_h=global_hidden)
                new_x_list.append(x)
            x_list = new_x_list
        
        return self.last_global_encoder(x_list, mask_list, h_in=global_hidden)

class ForgettingCurveNN(torch.nn.Module):
    def __init__(self, n_input, n_encoding, dropout):
        super().__init__()
        self.n_hidden = n_input        
        self.n_layers = FORGETTING_CURVE_N_LAYERS
        self.blocks = nn.ModuleList(
            [FFBlockWithEncoder(self.n_hidden, n_encoding=n_encoding, use_timeshift=False, dropout=FORGETTING_CURVE_DROPOUT) for _ in range(self.n_layers)],
        )
        self.norm = nn.LayerNorm(self.n_hidden)
        self.forgetting_curve_last_linear = nn.Linear(self.n_hidden, 4)
        with torch.no_grad():
            self.forgetting_curve_last_linear.bias.data.copy_(torch.tensor([-0.1645, -0.0393,  0.3989, -0.2395]))

    def forward(self, x, label_elapsed_days_real_bl, encoding_h):
        x = torch.cat([x, transform_elapsed_days_real(label_elapsed_days_real_bl).unsqueeze(-1)], dim=-1)
        for block in self.blocks:
            x = block(x, encoding_h=encoding_h)
        x = self.norm(x)
        x = self.forgetting_curve_last_linear(x)
        return x

class CardModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.n_features = 5
        self.n_hidden = DECODER_N_HIDDEN
        self.n_blocks = 2
        self.n_encoding = n_encoding
        self.dropout = BASE_DROPOUT
        self.encoder_linear = nn.Linear(n_encoding, self.n_hidden, bias=False)
        torch.nn.init.zeros_(self.encoder_linear.weight)

        self.feature_linear = nn.Linear(self.n_features, self.n_hidden)
        self.blocks = nn.ModuleList([BlockWithEncoder(self.n_hidden, n_encoding=n_encoding, dropout=self.dropout) for _ in range(self.n_blocks)])
        self.last_rnn_block = RNNBlock(self.n_hidden, dropout=self.dropout)
        self.transition = nn.Sequential(
            nn.LayerNorm(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden - 1),
        )
        self.forgetting_curve_nn = ForgettingCurveNN(self.n_hidden, n_encoding=n_encoding, dropout=self.dropout)

    def forward(self, encoding_h, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_real_bl):
        B, L = feature_elapsed_days_real_bl.shape
        feature_rating_onehot_bl4 = torch.nn.functional.one_hot((feature_rating_bl.long() - 1).clamp(min=0), num_classes=4).float()
        x = torch.cat(
            (
                transform_elapsed_days_real(feature_elapsed_days_real_bl).unsqueeze(-1),  # [B, L, 1]
                feature_rating_onehot_bl4,                                                        # [B, L, 4]
            ),
            dim=-1,
        )
        x = self.feature_linear(x) + self.encoder_linear(encoding_h)
        for block in self.blocks:
            x = block(x, encoding_h=encoding_h)
        x = self.last_rnn_block(x)
        x = self.transition(x)
        return self.forgetting_curve_nn(x, label_elapsed_days_real_bl, encoding_h=encoding_h)

class FirstReviewModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.first_review_last_linear = nn.Linear(n_encoding, 4)
        torch.nn.init.zeros_(self.first_review_last_linear.weight)
        torch.nn.init.zeros_(self.first_review_last_linear.bias)

    def forward(self, encoding_bh):
        return self.first_review_last_linear(encoding_bh)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_encoding = N_ENCODING
        self.encoder_model = EncoderModel(n_encoding=self.n_encoding)
        self.card_model = CardModel(n_encoding=self.n_encoding)
        self.first_review_model = FirstReviewModel(n_encoding=self.n_encoding)

    def get_excluded_params():
        return ["forgetting_curve_last_linear", "first_review_last_linear"] 