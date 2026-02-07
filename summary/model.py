import torch
import torch.nn as nn


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
        nn.init.zeros_(zero_init_linear.weight)
        nn.init.zeros_(zero_init_linear.bias)

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
        nn.init.zeros_(zero_init_linear.weight)
        nn.init.zeros_(zero_init_linear.bias)

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
        self.value_linear = nn.Linear(self.n_hidden, self.n_encoding + 2, bias=False)
        self.weight_linear = nn.Linear(self.n_hidden, self.n_encoding + 2, bias=False)
        self.encode_transform_nn = nn.Sequential(
            *[FFBlock(self.n_encoding + 2, dropout=self.dropout) for _ in range(2)],
            nn.LayerNorm(self.n_encoding + 2),
            nn.Linear(self.n_encoding + 2, self.n_encoding),
        )
        self.recency_const_log = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.recency_degree_log = torch.nn.parameter.Parameter(torch.tensor(0.0))

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
        return self.encode_transform_nn(torch.cat((encoding_sh, review_range_s1.clamp(max=1e5).log()), dim=-1))

    def get_recency_weights(self, ord_sbl, mask_sbl, n_sbl):
        # ord_bl contains values from 0 to n-1
        # print("TODO recency")
        x_sbl = ord_sbl / n_sbl
        recency_const = torch.exp(self.recency_const_log)
        recency_degree = torch.exp(self.recency_degree_log)
        x_sbl = recency_const + torch.pow(x_sbl, recency_degree)

        # normalize the weights
        sum_s = (x_sbl * mask_sbl).sum(dim=(1, 2))
        weight_s = mask_sbl.sum(dim=(1, 2))
        S = sum_s.size(0)
        return x_sbl * weight_s.view(S, 1, 1) / (1e-5 + sum_s.view(S, 1, 1))


class ForgettingCurveNN(torch.nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.n_hidden = n_input        
        self.n_layers = 4
        self.blocks = nn.ModuleList([FFBlock(self.n_hidden) for _ in range(self.n_layers)])

        zero_init_linear = nn.Linear(self.n_hidden, 4)
        nn.init.zeros_(zero_init_linear.weight)
        nn.init.zeros_(zero_init_linear.bias)

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
        self.n_encoding = 14
        self.encoder_model = EncoderModel(n_encoding=self.n_encoding)
        self.card_model = CardModel(n_encoding=self.n_encoding)