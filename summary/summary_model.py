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
    def __init__(self, n_hidden):
        super().__init__()
        self.seq = ResBlock(nn.Sequential([
            nn.LayerNorm(n_hidden),
            RNNWrapper(nn.LSTM(input_size=n_hidden, hidden_size=n_hidden, batch_first=True)),
        ]))
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
    def __init__(self, n_hidden):
        super().__init__()
        self.seq = ResBlock(nn.Sequential([
            nn.LayerNorm(n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Mish(),
            nn.Linear(self.n_hidden, self.n_hidden),
        ]))

    def forward(self, x):
        return self.seq(x)

class Block(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.seq = nn.Sequential([
            RNNBlock(n_hidden=n_hidden),
            FFBlock(n_hidden=n_hidden),
        ])

    def forward(self, x):
        return self.seq(x)

class SummaryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_features = 2
        self.n_hidden = 16
        self.n_layers = 3
        self.n_curves = 3
        self.n_slots = 16

        self.encoder = nn.Linear(self.n_features, self.n_hidden),
        self.blocks = nn.ModuleList([Block(self.n_hidden) for _ in range(self.n_layers)])
        self.out_norm = nn.LayerNorm(self.n_hidden)
        self.value_linear = nn.Linear(self.n_hidden, self.n_slots, bias=False)
        self.weight_linear = nn.Linear(self.n_hidden, self.n_slots, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        value_blh = self.value_linear(x)
        weight_blh = torch.sigmoid(self.weight_linear(x))
        return weight_blh, value_blh

class ForgettingCurveNN(torch.nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.n_hidden = n_input        
        self.n_layers = 3
        self.blocks = nn.ModuleList([FFBlock(self.n_hidden) for _ in range(self.n_layers)])
        self.out_head = nn.Sequential([
            nn.LayerNorm(self.n_hidden),
            nn.Linear(self.n_hidden, 4),
        ])

    def forward(self, x, label_elapsed_days_real_bl, label_is_new_anki_day_bl):
        log_label_elapsed_days_real_bl = label_elapsed_days_real_bl.clamp(min=1e-5).log()
        x = torch.cat([x[..., :(self.n_hidden - 2)], log_label_elapsed_days_real_bl.unsqueeze(-1), label_is_new_anki_day_bl.float().unsqueeze(-1)], dim=-1)
        for block in self.blocks:
            x = block(x)
        x = self.out_head(x)
        return x


class CardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_features = 2
        self.n_hidden = 16
        self.n_blocks = 1
        self.n_curves = 3
        self.encoder = nn.Linear(self.n_features, self.n_hidden),
        self.blocks = nn.ModuleList([Block(self.n_hidden) for _ in range(self.n_blocks)])
        self.last_rnn = nn.LSTM(input_size=self.n_hidden, hidden_size=self.n_hidden, batch_first=True)
        self.out_norm = nn.LayerNorm(self.n_hidden)
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

    def forward(self, x, label_elapsed_days_real, label_is_new_anki_day):
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        x, _ = self.last_rnn(x)
        x = self.out_norm(x)
        return self.forgetting_curve_nn(x, label_elapsed_days_real, label_is_new_anki_day)