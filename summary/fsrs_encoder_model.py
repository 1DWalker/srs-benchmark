from summary import fsrs_7, fsrs_7_truth
import torch
import torch.nn as nn
import numpy as np

BASE_DROPOUT = 0
FORGETTING_CURVE_DROPOUT = 1 - (1 - BASE_DROPOUT) ** 2
FIRST_REVIEW_DROPOUT = 0
GLOBAL_ENCODER_DROPOUT = 0
GLOBAL_ENCODER_CHANNEL_DROPOUT = 0.01
LAST_GLOBAL_NN_DROPOUT = 0

ENCODER_N_HIDDEN = 24
GLOBAL_FACTOR = 4
ENCODER_FULL_BLOCKS = 3
N_ENCODING = ENCODER_N_HIDDEN * GLOBAL_FACTOR * (ENCODER_FULL_BLOCKS + 1)

INTERMEDIATE_GLOBAL_LAYERS = 5
LAST_GLOBAL_LAYERS = 20

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

class GlobalEncoder(torch.nn.Module):
    def __init__(self, n_in, n_out, n_hidden_in, num_blocks=3, intermediate=False):
        super().__init__()
        self.n_proj = GLOBAL_FACTOR * n_in
        self.n_hidden = self.n_proj + n_hidden_in
        self.norm = nn.LayerNorm(n_in)
        self.value_linear = nn.Linear(n_in, self.n_proj, bias=False)
        self.weight_linear = nn.Linear(n_in, self.n_proj)
        self.intermediate = intermediate
        if self.intermediate:
            self.accept_linear = nn.Linear(n_in, n_out)
            torch.nn.init.orthogonal_(self.accept_linear.weight)
            self.intermediate_out = nn.Linear(self.n_hidden, n_out)
            torch.nn.init.zeros_(self.intermediate_out.weight)

        self.in_norm = nn.LayerNorm(self.n_proj)
        self.core = nn.Sequential(
            *[FFBlock(self.n_hidden, use_timeshift=False, dropout=GLOBAL_ENCODER_DROPOUT, dropout_channel=GLOBAL_ENCODER_CHANNEL_DROPOUT) for _ in range(num_blocks)],
            nn.LayerNorm(self.n_hidden),
        )
    
    def forward(self, x_list, mask_list, h_in):
        accum_weighted_value_h = 0
        accum_weight_h = 0
        accept_weights_list = []
        for x_bl, mask_bl in zip(x_list, mask_list):
            assert len(x_bl.shape) == 3
            x_bl = self.norm(x_bl)
            value_blh = self.value_linear(x_bl)
            weight_blh = torch.nn.functional.softplus(self.weight_linear(x_bl))
            if self.intermediate:
                accept_weights_list.append(self.accept_linear(x_bl))

            eff_weight_blh = mask_bl.unsqueeze(-1).float() * weight_blh
            accum_weighted_value_h += (eff_weight_blh * value_blh).sum(dim=(0, 1))
            accum_weight_h += eff_weight_blh.sum(dim=(0, 1))

        sum_encoding_h = accum_weighted_value_h / (accum_weight_h + 1e-6)
        x_h = self.in_norm(sum_encoding_h)
        x_h = torch.cat((x_h, h_in), dim=-1)
        x_h = self.core(x_h)
        if self.intermediate:
            return x_h, self.intermediate_out(x_h), accept_weights_list
        else:
            return x_h


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
            [GlobalEncoder(n_in=self.n_hidden, n_out=self.n_hidden, n_hidden_in=GLOBAL_FACTOR * i * self.n_hidden, num_blocks=INTERMEDIATE_GLOBAL_LAYERS, intermediate=True) for i in range(self.full_blocks)]
        )
        self.intermediate_ffs = nn.ModuleList([
            FFBlockWithEncoder(self.n_hidden, n_encoding=GLOBAL_FACTOR * (i + 1) * self.n_hidden, use_timeshift=True, dropout=self.dropout)
            for i in range(self.full_blocks)
        ])
        self.intermediate_blocks = nn.ModuleList([
            BlockWithEncoder(self.n_hidden, n_encoding=GLOBAL_FACTOR * (i + 1) * self.n_hidden, dropout=self.dropout)
            for i in range(self.full_blocks)
        ])
        self.last_global_encoder = GlobalEncoder(n_in=self.n_hidden, n_out=self.n_hidden, n_hidden_in=GLOBAL_FACTOR * self.full_blocks * self.n_hidden, num_blocks=LAST_GLOBAL_LAYERS, intermediate=False)
        self.n_full_global = (self.full_blocks + 1) * self.n_hidden * GLOBAL_FACTOR

        self.fsrs_linear = nn.Linear(self.n_full_global, 35, bias=True)
        torch.nn.init.zeros_(self.fsrs_linear.weight)
        torch.nn.init.zeros_(self.fsrs_linear.bias)

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
        
        x = self.fsrs_linear(self.last_global_encoder(x_list, mask_list, h_in=global_hidden))
        x = fsrs_7.nn_vec_to_fsrs7_params(x)
        return x

# class Config:
#     def __init__(self):
#         self.s_min = 0.0001
#         self.init_s_max = 100
#         self.use_secs_intervals = True
#         self.device = torch.device("cuda")


class CardModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()

    def forward(self, encoding_h, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_real_bl):
        B, L = feature_elapsed_days_real_bl.shape
        x_bl, state_bl2 = fsrs_7.forward(
            parameters_p=encoding_h, 
            feature_elapsed_days_real_bl=feature_elapsed_days_real_bl,
            feature_rating_bl=feature_rating_bl,   
            label_elapsed_days_real_bl=label_elapsed_days_real_bl,
        )
        # fsrs = fsrs_7_truth.FSRS7(config=Config(), w=encoding_h.tolist()).to(torch.device("cuda"))
        # sequences_lb2 = torch.stack((feature_elapsed_days_real_bl, feature_rating_bl), dim=-1).transpose(0, 1)
        # out2 = fsrs.batch_process(sequences_lb2, 0, 0, real_batch_size=B)
        # out1 = state_bl2.transpose(0, 1)
        # error = (out1 - out2).max()
        # print("done, error", error)

        eps = 1e-6
        x_bl4 = torch.full((*x_bl.shape, 4), float("-inf"),
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
        self.n_encoding = N_ENCODING
        self.encoder_model = EncoderModel(n_encoding=self.n_encoding)
        self.card_model = CardModel(n_encoding=self.n_encoding)
        self.first_review_model = FirstReviewModel(n_encoding=self.n_encoding)

    def get_excluded_params(self):
        return [] 