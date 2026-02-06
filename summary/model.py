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
        outputs, _ = self.module(inputs)
        return outputs

class RNNBlock(nn.Module):
    def __init__(self, n_hidden, dropout=0):
        super().__init__()
        self.seq = ResBlock(nn.Sequential(
            nn.LayerNorm(n_hidden),
            RNNWrapper(nn.LSTM(input_size=n_hidden, hidden_size=n_hidden, batch_first=True)),
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
        self.seq = ResBlock(nn.Sequential(
            nn.LayerNorm(n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Mish(),
            nn.Linear(self.n_hidden, self.n_hidden),
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
        self.n_features = 2
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

    def forward(self, feature_elapsed_days_real_bl, feature_rating_bl):
        x = torch.stack((feature_elapsed_days_real_bl, feature_rating_bl), dim=-1)
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        value_blh = self.value_linear(x)
        weight_blh = torch.sigmoid(self.weight_linear(x))
        return weight_blh, value_blh

    def transform(self, encoding_h):
        return self.encode_transform_nn(encoding_h)

    def get_recency_weights(self, ord_bl, n):
        # ord_bl contains values from 0 to n-1
        print("TODO recency")
        return torch.ones_like(ord_bl).float()


class ForgettingCurveNN(torch.nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.n_hidden = n_input        
        self.n_layers = 4
        self.blocks = nn.ModuleList([FFBlock(self.n_hidden) for _ in range(self.n_layers)])
        self.out_head = nn.Sequential(
            nn.LayerNorm(self.n_hidden),
            nn.Linear(self.n_hidden, 4),
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
        self.n_features = 2 + n_encoding
        self.n_hidden = 16
        self.n_blocks = 2
        self.n_encoding = n_encoding
        self.dropout = 0.1
        self.encoder = nn.Linear(self.n_features, self.n_hidden)
        self.rnn_blocks = nn.ModuleList([RNNBlock(self.n_hidden, dropout=self.dropout) for _ in range(self.n_blocks)])
        self.transition = nn.Sequential(
            nn.LayerNorm(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden - 2),
        )
        self.forgetting_curve_nn = ForgettingCurveNN(self.n_hidden)

        for name, param in self.named_parameters():
            if "weight_ih" in name:  # Input-to-hidden weights
                nn.init.orthogonal_(param.data)
            elif "weight_hh" in name:  # Hidden-to-hidden weights
                nn.init.orthogonal_(param.data)
            elif "bias_ih" in name:  # Biases
                start_index = len(param.data) // 4
                end_index = len(param.data) // 2
                param.data[start_index:end_index].fill_(1.0)

    def forward(self, encoding_h, feature_elapsed_days_real_bl, feature_rating_bl, label_elapsed_days_real_bl, label_is_new_anki_day):
        B, L = feature_elapsed_days_real_bl.shape
        H = encoding_h.size(0)
        x = torch.cat((feature_elapsed_days_real_bl.unsqueeze(-1), feature_rating_bl.unsqueeze(-1), encoding_h.view(1, 1, H).expand(B, L, H)), dim=-1)
        x = self.encoder(x)
        for block in self.rnn_blocks:
            x = block(x)
        x = self.transition(x)
        return self.forgetting_curve_nn(x, label_elapsed_days_real_bl, label_is_new_anki_day)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_encoding = 14
        self.encoder_model = EncoderModel(n_encoding=self.n_encoding)
        self.card_model = CardModel(n_encoding=self.n_encoding)