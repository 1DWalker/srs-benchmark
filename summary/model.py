import torch
import torch.nn as nn
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class RNNWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs)[0]

class RNNBlock(nn.Module):
    def __init__(self, n_hidden, dropout=0):
        super().__init__()

        zero_init_linear = nn.Linear(n_hidden, n_hidden)
        # nn.init.zeros_(zero_init_linear.weight)
        # nn.init.zeros_(zero_init_linear.bias)

        self.seq = ResBlock(nn.Sequential(
            nn.LayerNorm(n_hidden),
            RNNWrapper(nn.LSTM(input_size=n_hidden, hidden_size=n_hidden, batch_first=True)),
            zero_init_linear,
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

class FFBlock(nn.Module):
    def __init__(self, n_hidden, dropout=0):
        super().__init__()
        self.n_hidden = n_hidden

        zero_init_linear = nn.Linear(n_hidden, n_hidden)
        # nn.init.zeros_(zero_init_linear.weight)
        # nn.init.zeros_(zero_init_linear.bias)

        self.seq = ResBlock(nn.Sequential(
            nn.LayerNorm(n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Mish(),
            zero_init_linear,
            nn.Dropout(p=dropout),
        ))

    def forward(self, x):
        return self.seq(x)

class Block(nn.Module):
    def __init__(self, n_hidden, dropout=0):
        super().__init__()
        self.seq = nn.Sequential(
            RNNBlock(n_hidden=n_hidden, dropout=dropout),
            FFBlock(n_hidden=n_hidden, dropout=dropout),
        )

    def forward(self, x):
        return self.seq(x)

class EncoderModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.n_features = 5
        self.n_hidden = 16
        self.n_layers = 3
        self.n_curves = 3
        self.n_encoding = n_encoding
        self.dropout = 0.1

        self.encoder = nn.Linear(self.n_features, self.n_hidden)
        self.blocks = nn.ModuleList([Block(self.n_hidden, dropout=self.dropout) for _ in range(self.n_layers)])
        self.out_norm = nn.LayerNorm(self.n_hidden)
        self.value_linear = nn.Linear(self.n_hidden, self.n_encoding, bias=False)
        self.weight_linear = nn.Linear(self.n_hidden, self.n_encoding, bias=False)
        self.encode_transform_nn = nn.Sequential(
            nn.Linear(self.n_encoding + 1, self.n_encoding),
            *[FFBlock(self.n_encoding, dropout=self.dropout) for _ in range(2)],
            nn.LayerNorm(self.n_encoding),
            nn.Linear(self.n_encoding, self.n_encoding),
        )
        self.recency_nn_hidden = 8
        self.recency_nn_last_linear = nn.Linear(self.recency_nn_hidden, 2)
        nn.init.zeros_(self.recency_nn_last_linear.weight)
        with torch.no_grad():
            self.recency_nn_last_linear.bias.copy_(torch.tensor([np.log(0.2), np.log(4)]))
        self.recency_nn = nn.Sequential(
            nn.Linear(1, self.recency_nn_hidden),
            *[FFBlock(self.recency_nn_hidden, dropout=0.5) for _ in range(2)],
            nn.LayerNorm(self.recency_nn_hidden),
        )

    def forward(self, feature_elapsed_days_real_bl, feature_rating_bl):
        feature_rating_onehot_bl4 = torch.nn.functional.one_hot((feature_rating_bl.long() - 1).clamp(min=0), num_classes=4).float()
        x = torch.cat((feature_elapsed_days_real_bl.clamp(min=1e-5).log().unsqueeze(-1), feature_rating_onehot_bl4), dim=-1)
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        value_blh = self.value_linear(x)
        weight_blh = torch.sigmoid(self.weight_linear(x))
        return weight_blh, value_blh

    def transform(self, encoding_sh, review_range_s1):
        return self.encode_transform_nn(torch.cat((encoding_sh, review_range_s1.clamp(min=1, max=1e5).log()), dim=-1))

    def train_size_to_recency_poly(self, train_s):
        x = self.recency_nn_last_linear(self.recency_nn(train_s.unsqueeze(-1).clamp(min=1, max=1e5).log()))
        return x.exp().unbind(dim=-1)

    def get_recency_weights(self, ord_sbl, mask_sbl, n_sbl):
        # ord_bl contains values from 0 to n-1
        x_sbl = ord_sbl / n_sbl
        weight_s = mask_sbl.sum(dim=(1, 2))

        recency_const_s, recency_degree_s = self.train_size_to_recency_poly(weight_s)
        S = weight_s.size(0)
        x_sbl = recency_const_s.view(S, 1, 1) + torch.pow(x_sbl, recency_degree_s.view(S, 1, 1))

        # normalize the weights
        sum_s = (x_sbl * mask_sbl).sum(dim=(1, 2))
        return x_sbl * weight_s.view(S, 1, 1) / (1e-5 + sum_s.view(S, 1, 1))


class ForgettingCurveNN(torch.nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.n_hidden = n_input        
        self.n_layers = 4
        self.blocks = nn.ModuleList([FFBlock(self.n_hidden) for _ in range(self.n_layers)])

        zero_init_linear = nn.Linear(self.n_hidden, 4)
        # nn.init.zeros_(zero_init_linear.weight)
        # nn.init.zeros_(zero_init_linear.bias)

        self.out_head = nn.Sequential(
            nn.LayerNorm(self.n_hidden),
            zero_init_linear,
        )

    def forward(self, x, label_elapsed_days_real_bl, label_is_new_anki_day_bl):
        log_label_elapsed_days_real_bl = label_elapsed_days_real_bl.clamp(min=1e-5).log()
        x = torch.cat([x, log_label_elapsed_days_real_bl.unsqueeze(-1), label_is_new_anki_day_bl.float().unsqueeze(-1)], dim=-1)
        for block in self.blocks:
            x = block(x)
        x = self.out_head(x)
        return x


class CardModel(torch.nn.Module):
    def __init__(self, n_encoding):
        super().__init__()
        self.n_features = 5 + n_encoding
        self.n_hidden = 16
        self.n_blocks = 2
        self.n_encoding = n_encoding
        self.dropout = 0.1
        self.encoder = nn.Linear(self.n_features, self.n_hidden)
        self.blocks = nn.ModuleList([Block(self.n_hidden, dropout=self.dropout) for _ in range(self.n_blocks)])
        self.last_rnn_block = RNNBlock(self.n_hidden, dropout=self.dropout)
        self.transition = nn.Sequential(
            nn.LayerNorm(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden - 2),
        )
        self.forgetting_curve_nn = ForgettingCurveNN(self.n_hidden)

    def forward(self, encoding_bh, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_real_bl, label_is_new_anki_day):
        B, L = feature_elapsed_days_real_bl.shape
        H = encoding_bh.size(1)
        feature_rating_onehot_bl4 = torch.nn.functional.one_hot((feature_rating_bl.long() - 1).clamp(min=0), num_classes=4).float()
        x = torch.cat(
            (
                feature_elapsed_days_real_bl.clamp(min=1e-5).log().unsqueeze(-1),  # [B, L, 1]
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

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_encoding = 16
        self.encoder_model = EncoderModel(n_encoding=self.n_encoding)
        self.card_model = CardModel(n_encoding=self.n_encoding)