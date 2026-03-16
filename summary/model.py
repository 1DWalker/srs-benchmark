import torch
import torch.nn as nn
import numpy as np

BASE_DROPOUT = 0
FORGETTING_CURVE_DROPOUT = 1 - (1 - BASE_DROPOUT) ** 2
FIRST_REVIEW_DROPOUT = 0
GLOBAL_ENCODER_DROPOUT = 0.1
GLOBAL_ENCODER_CHANNEL_DROPOUT = 0.01
LAST_GLOBAL_NN_DROPOUT = 0

# ENCODER_N_HIDDEN = 16
# DECODER_N_HIDDEN = 24
# FORGETTING_CURVE_N_LAYERS = 4
# GLOBAL_FACTOR = 4
# ENCODER_FULL_BLOCKS = 2
# N_ENCODING = ENCODER_N_HIDDEN * GLOBAL_FACTOR * (ENCODER_FULL_BLOCKS + 1)

ENCODER_N_HIDDEN = 24
DECODER_N_HIDDEN = 16
FORGETTING_CURVE_N_LAYERS = 3
GLOBAL_FACTOR = 3
ENCODER_FULL_BLOCKS = 4
FF_PER_BLOCK = 2
N_ENCODING = int((FF_PER_BLOCK * ENCODER_FULL_BLOCKS + 1) * ENCODER_N_HIDDEN * GLOBAL_FACTOR)

INTERMEDIATE_GLOBAL_LAYERS = 8
LAST_GLOBAL_LAYERS = 40
EXCLUDE_LAST_DROPOUT = 4  # number of suffix layers to exclude from dropout

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
            nn.Linear(n_hidden, n_hidden, bias=False),
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
        self.n_act = int(n_hidden * 2 / 3)
        # self.n_act = n_hidden
        self.start = nn.Sequential(
            nn.LayerNorm(n_hidden),
            *[TimeShiftLerp(n_hidden=n_hidden) for _ in range(1 if use_timeshift else 0)],
        )
        self.A = nn.Linear(n_hidden, self.n_act)
        self.act = nn.SiLU()
        self.gate_linear = nn.Linear(n_hidden, self.n_act)
        self.B = nn.Linear(self.n_act, n_hidden, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_channel = nn.Dropout(p=dropout_channel)

    def forward(self, x_in):
        assert not x_in.isnan().any()
        x = self.start(x_in)
        gate = self.gate_linear(x)
        x = self.act(self.A(x))
        x = x * gate
        assert not x.isnan().any()
        x = self.B(x)
        return self.dropout_channel(self.dropout(x) + x_in)

# class FFBlock(torch.nn.Module):
#     def __init__(self, n_hidden, use_timeshift, dropout=0, dropout_channel=0):
#         super().__init__()
#         self.seq = ResBlock(nn.Sequential(
#             nn.LayerNorm(n_hidden),
#             *[TimeShiftLerp(n_hidden=n_hidden) for _ in range(1 if use_timeshift else 0)],
#             nn.Linear(n_hidden, n_hidden),
#             nn.Mish(),
#             nn.Linear(n_hidden, n_hidden),
#             *[nn.Dropout(p=dropout) for _ in range(1 if dropout > 0 else 0)],
#         ))
#         self.dropout_channel = nn.Dropout(p=dropout_channel)

#     def forward(self, x):
#         return self.dropout_channel(self.seq(x))

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
        self.n_act = int(n_hidden * 2 / 3)
        self.start = nn.Sequential(
            nn.LayerNorm(n_hidden),
            *[TimeShiftLerp(n_hidden=n_hidden) for _ in range(1 if use_timeshift else 0)],
        )
        self.gate_linear_combine = nn.Linear(n_hidden, self.n_act)
        self.global_gate_linear_combine = nn.Linear(n_encoding, self.n_act, bias=False)
        with torch.no_grad():
            self.gate_linear_combine.weight.mul_(0.5)
            self.global_gate_linear_combine.weight.mul_(0.5)

        self.A_combine = nn.Linear(n_hidden, self.n_act)
        self.global_linear_combine = nn.Linear(n_encoding, self.n_act, bias=False)
        with torch.no_grad():
            self.A_combine.weight.mul_(0.5)
            self.global_linear_combine.weight.mul_(0.5)
        self.act = nn.SiLU()
        self.B = nn.Sequential(
            nn.Linear(self.n_act, n_hidden, bias=False),
            *[nn.Dropout(p=dropout) for _ in range(1 if dropout > 0 else 0)],
        )

        # cached values
        self.register_buffer("_enc_gate", None, persistent=False)
        self.register_buffer("_enc_lin", None, persistent=False)

    def set_encoding(self, encoding_h):
        self._enc_gate = self.global_gate_linear_combine(encoding_h)
        self._enc_lin = self.global_linear_combine(encoding_h)

    def _core(self, x, enc_gate, enc_lin):
        gate = self.gate_linear_combine(x) + enc_gate
        x = self.A_combine(x) + enc_lin
        x = self.act(x)
        x = x * gate
        assert not x.isnan().any()
        x = self.B(x)
        return x


    def forward(self, x_in, encoding_h):
        x = self.start(x_in)
        if encoding_h is not None:
            self.set_encoding(encoding_h)

        enc_gate = self._enc_gate
        enc_lin = self._enc_lin

        x = torch.utils.checkpoint.checkpoint(
            self._core,
            x,
            enc_gate,
            enc_lin,
        )
        # x = self._core(x, enc_gate, enc_lin)

        return x_in + x


# class FFBlockWithEncoder(torch.nn.Module):
#     def __init__(self, n_hidden, n_encoding, use_timeshift, dropout=0):
#         super().__init__()
#         self.start = nn.Sequential(
#             nn.LayerNorm(n_hidden),
#             *[TimeShiftLerp(n_hidden=n_hidden) for _ in range(1 if use_timeshift else 0)],
#         )

#         self.encoding_linear_combine = nn.Linear(n_encoding, n_hidden, bias=False)
#         self.A_combine = nn.Linear(n_hidden, n_hidden)
#         with torch.no_grad():
#             self.encoding_linear_combine.weight.mul_(0.5)
#             self.A_combine.weight.mul_(0.5)

#         self.act = nn.Mish()
#         self.B = nn.Sequential(
#             nn.Linear(n_hidden, n_hidden),
#             *[nn.Dropout(p=dropout) for _ in range(1 if dropout > 0 else 0)],
#         )

#     def _core(self, x_in, encoding_h):
#         x = self.start(x_in)
#         x = self.act(self.A_combine(x) + self.encoding_linear_combine(encoding_h))
#         x = self.B(x)
#         return x

#     def forward(self, x_in, encoding_h):
#         x = torch.utils.checkpoint.checkpoint(self._core, x_in, encoding_h)
#         return x_in + x

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
    def __init__(self, n_in, n_out, n_hidden_in, num_blocks=3, exclude_dropout=0):
        super().__init__()
        self.n_proj = int(GLOBAL_FACTOR * n_in)
        self.n_hidden = self.n_proj + n_hidden_in
        self.norm = nn.LayerNorm(n_in)
        self.value_linear = nn.Linear(n_in, self.n_proj, bias=False)
        self.weight_linear = nn.Linear(n_in, self.n_proj, bias=False)
        torch.nn.init.zeros_(self.weight_linear.weight)

        self.in_norm = nn.GroupNorm(
                num_groups = 3,
                num_channels = 3 * self.n_proj,
                affine=True
            )
        self.in_proj = nn.Sequential(
            nn.Linear(3 * self.n_proj, self.n_proj, bias=False),
            nn.LayerNorm(self.n_proj),
        )
        def dropout_multi(layer):
            if layer + exclude_dropout < num_blocks:
                return 1.0
            else:
                return 0.0

        self.core = nn.Sequential(
            *[FFBlock(self.n_hidden, use_timeshift=False, dropout=GLOBAL_ENCODER_DROPOUT * dropout_multi(i), dropout_channel=GLOBAL_ENCODER_CHANNEL_DROPOUT * dropout_multi(i)) for i in range(num_blocks)],
        )
        self.out_norm = nn.LayerNorm(self.n_hidden)

    def forward(self, x_in_list, mask_list, h_in, log=None):
        x_norm_list = [self.norm(x_bl) for x_bl in x_in_list]
    
        def _moments_fn(*inputs):
            n = len(inputs) // 2
            xs = inputs[:n]
            ms = inputs[n:]

            global_max = torch.full((self.n_proj,), -float("inf"), device=inputs[0][0].device)
            ws = []
            for x_bl, mask_bl in zip(xs, ms):
                w = self.weight_linear(x_bl)
                w = w.masked_fill(~mask_bl.unsqueeze(-1), float('-inf'))
                ws.append(w)
                global_max = torch.maximum(global_max, w.amax((0,1)))

            sum_w = sum_wx = sum_wx2 = sum_wx3 = 0
            for x_bl, mask_bl, w in zip(xs, ms, ws):
                v = self.value_linear(x_bl)
                eff_blh = torch.exp(w - global_max)

                sum_w  += eff_blh.sum((0,1))
                sum_wx  += (eff_blh * v).sum((0,1))
                sum_wx2 += (eff_blh * v**2).sum((0,1))
                sum_wx3 += (eff_blh * v**3).sum((0,1))

            den = sum_w + 1e-7
            mean = sum_wx / den
            var = sum_wx2 / den - mean**2
            m3 = sum_wx3 / den - 3*mean*var - mean**3
            std = torch.sqrt(var + 1e-7)
            skew = m3 / (std**3 + 1e-7)
            return mean, std, skew

        inputs = tuple(x_norm_list) + tuple(mask_list)
        mean_h, std_h, skew_h = torch.utils.checkpoint.checkpoint(
            _moments_fn,
            *inputs,
            use_reentrant=False
        )

        x_3h = torch.cat((mean_h, std_h, skew_h), dim=-1)
        x_3h = self.in_norm(x_3h.unsqueeze(0)).squeeze(0)
        x_3h = self.in_proj(x_3h)
        x_p = torch.cat((x_3h, h_in), dim=-1)
        x_p = self.core(x_p)
        if log is not None:
            log["hidden/hidden_out_std"] = x_p.std()
            log["hidden/hidden_out_mean"] = x_p.mean()
            log["hidden/hidden_out_max"] = x_p.max()
            log["hidden/hidden_out_min"] = x_p.min()
        x_p = self.out_norm(x_p)
        return x_p

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
        )
        self.full_blocks = ENCODER_FULL_BLOCKS # 1 full block consists of LSTM - GLOBAL_FF - GLOBAL_FF 
        self.FF_PER_BLOCK = FF_PER_BLOCK
        self.intermediate_global_encoders = nn.ModuleList(
            [GlobalEncoder(n_in=self.n_hidden, n_out=self.n_hidden, n_hidden_in=int(GLOBAL_FACTOR * i * self.n_hidden), num_blocks=INTERMEDIATE_GLOBAL_LAYERS, exclude_dropout=1) for i in range(self.FF_PER_BLOCK * self.full_blocks)]
        )
        self.intermediate_ffs = nn.ModuleList([
            FFBlockWithEncoder(self.n_hidden, n_encoding=int(GLOBAL_FACTOR * (i + 1) * self.n_hidden), use_timeshift=True, dropout=self.dropout)
            for i in range(self.FF_PER_BLOCK * self.full_blocks)
        ])
        self.intermediate_lstm = nn.ModuleList([
            RNNBlock(n_hidden=self.n_hidden, dropout=self.dropout)
            for i in range(self.full_blocks)
        ])
        self.last_global_encoder = GlobalEncoder(n_in=self.n_hidden, n_out=self.n_hidden, n_hidden_in=int(GLOBAL_FACTOR * self.FF_PER_BLOCK * self.full_blocks * self.n_hidden), num_blocks=LAST_GLOBAL_LAYERS, exclude_dropout=EXCLUDE_LAST_DROPOUT)


    def forward(
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
                card_review_th_ratio_bl.unsqueeze(-1).clamp(min=0, max=1), 
                transform_global_n_reviews(global_num_reviews).view(1, 1, 1).expand(B, L, 1),
                card_i_ratio_bl.unsqueeze(-1).clamp(min=0, max=1), 
                transform_card_num_reviews(card_num_reviews_bl).unsqueeze(-1),
            ), 
            dim=-1,
        )
        return self.encode_block(x)

    def run_intermediate_lstm(self, i, x_list):
        new_x_list = []
        for x in x_list:
            new_x_list.append(self.intermediate_lstm[i](x))
        return new_x_list

    def run_ff(self, i, j, global_hidden, x_list, mask_list):
        new_x_list = []
        global_hidden = self.intermediate_global_encoders[self.FF_PER_BLOCK * i + j](x_list, mask_list, h_in=global_hidden)
        intermediate_ff = self.intermediate_ffs[self.FF_PER_BLOCK * i + j]
        intermediate_ff.set_encoding(encoding_h=global_hidden)
        for x in x_list:
            new_x_list.append(intermediate_ff(x, encoding_h=None))
        return global_hidden, new_x_list

    def run_core(self, x_list, mask_list, log=None):
        global_hidden = torch.tensor([], device=x_list[0].device)
        card_max_stds = []
        for i in range(self.full_blocks):
            x_list = self.run_intermediate_lstm(i, x_list)

            for j in range(self.FF_PER_BLOCK):
                global_hidden, x_list = self.run_ff(i, j, global_hidden, x_list, mask_list)

            with torch.no_grad():
                for x in x_list:
                    card_max_stds.append(x.std(dim=-1).max().item())

        if log is not None:
            log["hidden/max_card_std"] = max(card_max_stds)
        return self.last_global_encoder(x_list, mask_list, h_in=global_hidden, log=log)


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
                feature_rating_onehot_bl4,                                                # [B, L, 4]
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

    def get_excluded_params(self):
        return ["forgetting_curve_last_linear", "first_review_last_linear"] 