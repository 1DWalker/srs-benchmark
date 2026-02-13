import torch
import torch.nn as nn
import numpy as np

BASE_DROPOUT = 0.3
FORGETTING_CURVE_DROPOUT = 1 - (1 - BASE_DROPOUT) ** 2
FIRST_REVIEW_DROPOUT = 1 - (1 - BASE_DROPOUT) ** 4
ENCODE_TRANSFORM_DROPOUT = 0.7
RECENCY_NN_DROPOUT = 0.5

def transform_elapsed_days_real(x):
    return ((x + 1e-5).log() + 1.3) / 5

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
            nn.Dropout(p=dropout),
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
            nn.Dropout(p=dropout),
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

class EncoderModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.n_features = 5
        self.n_hidden = n_encoding
        self.n_layers = 3
        self.n_curves = 3
        self.n_encoding = n_encoding
        self.dropout = BASE_DROPOUT

        self.encoder = nn.Linear(self.n_features, self.n_hidden)
        self.blocks = nn.ModuleList([Block(self.n_hidden, dropout=self.dropout) for _ in range(self.n_layers)])
        self.out_norm = nn.LayerNorm(self.n_hidden)
        self.value_linear = nn.Linear(self.n_hidden, self.n_encoding, bias=False)
        self.weight_linear = nn.Linear(self.n_hidden, self.n_encoding)
        self.encode_transform_nn = nn.Sequential(
            nn.Linear(self.n_encoding + 1, self.n_encoding),
            *[FFBlock(self.n_encoding, use_timeshift=False, dropout=ENCODE_TRANSFORM_DROPOUT) for _ in range(6)],
            nn.LayerNorm(self.n_encoding),
        )
        self.recency_nn_hidden = 8
        self.recency_nn_last_linear = nn.Linear(self.recency_nn_hidden, 2)
        nn.init.zeros_(self.recency_nn_last_linear.weight)
        with torch.no_grad():
            self.recency_nn_last_linear.bias.copy_(torch.tensor([np.log(0.05), np.log(15)]))
        self.recency_nn = nn.Sequential(
            nn.Linear(1, self.recency_nn_hidden),
            *[FFBlock(self.recency_nn_hidden, use_timeshift=False, dropout=RECENCY_NN_DROPOUT) for _ in range(2)],
            nn.LayerNorm(self.recency_nn_hidden),
        )

    def forward(self, feature_elapsed_days_real_bl, feature_rating_bl):
        feature_rating_onehot_bl4 = torch.nn.functional.one_hot((feature_rating_bl.long() - 1).clamp(min=0), num_classes=4).float()
        x = torch.cat((transform_elapsed_days_real(feature_elapsed_days_real_bl).unsqueeze(-1), feature_rating_onehot_bl4), dim=-1)
        assert not x.isnan().any()
        x = self.encoder(x)
        assert not x.isnan().any()
        for block in self.blocks:
            x = block(x)
            assert not x.isnan().any()
        x = self.out_norm(x)
        value_blh = self.value_linear(x)
        weight_blh = torch.sigmoid(self.weight_linear(x))
        return weight_blh, value_blh

    def train_n_transform(self, train_n_s):
        return (train_n_s.clamp(min=1, max=1e5).log() - 9) / 2

    def transform(self, encoding_sh, review_range_s):
        return self.encode_transform_nn(torch.cat((encoding_sh, self.train_n_transform(review_range_s).unsqueeze(-1)), dim=-1))

    def train_size_to_recency_poly(self, train_n_s):
        x = self.recency_nn_last_linear(self.recency_nn(self.train_n_transform(train_n_s).unsqueeze(-1)))
        return x.exp().unbind(dim=-1)

    def get_non_norm_recency_weights(self, ord_sbl, n_s):
        # ord_bl contains values from 0 to n-1
        S = n_s.size(0)
        x_sbl = ord_sbl / n_s.view(S, 1, 1)

        recency_const_s, recency_degree_s = self.train_size_to_recency_poly(n_s)
        x_sbl = recency_const_s.view(S, 1, 1) + torch.pow(x_sbl, recency_degree_s.view(S, 1, 1))
        return x_sbl


class ForgettingCurveNN(torch.nn.Module):
    def __init__(self, n_input, dropout):
        super().__init__()
        self.n_hidden = n_input        
        self.n_layers = 4
        self.core = nn.Sequential(
            *[FFBlock(self.n_hidden, use_timeshift=False, dropout=FORGETTING_CURVE_DROPOUT) for _ in range(self.n_layers)],
            nn.LayerNorm(self.n_hidden),
        )

        self.forgetting_curve_last_linear = nn.Linear(self.n_hidden, 4)
        with torch.no_grad():
            self.forgetting_curve_last_linear.bias.data.copy_(torch.tensor([-0.1645, -0.0393,  0.3989, -0.2395]))

    def forward(self, x, label_elapsed_days_real_bl, label_is_new_anki_day_bl):
        x = torch.cat([x, transform_elapsed_days_real(label_elapsed_days_real_bl).unsqueeze(-1), label_is_new_anki_day_bl.float().unsqueeze(-1)], dim=-1)
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
            nn.Linear(self.n_hidden, self.n_hidden - 2),
        )
        self.forgetting_curve_nn = ForgettingCurveNN(self.n_hidden, self.dropout)

    def forward(self, encoding_bh, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_real_bl, label_is_new_anki_day):
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
        return self.forgetting_curve_nn(x, label_elapsed_days_real_bl, label_is_new_anki_day)

class FirstReviewModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.n_layers = 3
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
        self.n_encoding = 24
        self.encoder_model = EncoderModel(n_encoding=self.n_encoding)
        self.card_model = CardModel(n_encoding=self.n_encoding)
        self.first_review_model = FirstReviewModel(n_encoding=self.n_encoding)