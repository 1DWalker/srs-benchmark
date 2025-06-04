

import random
import time
import torch

from rwkv.model.rwkv_model import RWKV7, RWKV7Config
from rwkv.model.rwkv_packed_sequence_model import RWKV7Packed, RWKV7PackedConfig
from rwkv.utils import get_number_of_trainable_parameters

def __nop(ob):
    return ob

ModuleType = torch.nn.Module
FunctionType = __nop

# ModuleType = torch.jit.ScriptModule
# FunctionType = torch.jit.script_method

class SummaryModel(ModuleType):
    def __init__(self):
        super().__init__()
        self.initial_encoding = torch.nn.Sequential(
            torch.nn.Linear(9, 32),
            torch.nn.Mish(),
        )
        rwkv7packed_config = RWKV7PackedConfig(d_model=32, n_heads=1, n_layers=2, channel_mixer_factor=1.5, decay_lora=8, a_lora=8, v0_mix_amt_lora=4, gate_lora=8)
        self.card_encoder = RWKV7Packed(rwkv7packed_config)
        rwkv7_config = RWKV7Config(d_model=32, n_heads=1, n_layers=3, channel_mixer_factor=2.0, layer_offset=0, total_layers=3, decay_lora=8, a_lora=8, v0_mix_amt_lora=4, gate_lora=8, dropout=0.01, dropout_layer=0.01)
        self.global_encoder = RWKV7(rwkv7_config)

    def forward(self, in_TC, indices_I, perm_T, perm_inv_T):
        T, C = in_TC.shape
        in_TC = self.initial_encoding(in_TC)
        card_in_TC = in_TC[perm_T]
        card_encoding_TC = self.card_encoder(card_in_TC, indices_I) 

        timeshift_select_1T = torch.cat((torch.zeros(1, dtype=torch.long, device=in_TC.device), torch.arange(T - 1, dtype=torch.long, device=in_TC.device))).unsqueeze(0)
        skip_1T = torch.full((T,), fill_value=0, dtype=torch.bool, device=in_TC.device).unsqueeze(0)
        global_in_TC = card_encoding_TC[perm_inv_T]
        global_out_TC = self.global_encoder(global_in_TC.unsqueeze(0), timeshift_select_1T, skip_1T).squeeze(0)
        return global_out_TC


class ToFSRSParamsLayer(ModuleType):
    def __init__(self, in_dim):
        super().__init__()
        self.ln = torch.nn.LayerNorm(in_dim)
        self.linear = torch.nn.Linear(in_dim, 21)

    def forward(self, x):
        w = self.linear(self.ln(x))
        pass