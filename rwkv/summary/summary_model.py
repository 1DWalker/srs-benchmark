

import random
import time
import numpy as np
import torch

from rwkv.model.rwkv_model import RWKV7, RWKV7Config
from rwkv.model.rwkv_packed_sequence_model import RWKV7Packed, RWKV7PackedConfig
from rwkv.utils import get_number_of_trainable_parameters

def __nop(ob):
    return ob

# ModuleType = torch.nn.Module
# FunctionType = __nop

ModuleType = torch.jit.ScriptModule
FunctionType = torch.jit.script_method

class SummaryCore(ModuleType):
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

    @FunctionType
    def forward(self, in_TC, indices_I, perm_T, perm_inv_T):
        T, _ = in_TC.shape
        in_TC = self.initial_encoding(in_TC)
        card_in_TC = in_TC[perm_T]
        card_encoding_TC = self.card_encoder(card_in_TC, indices_I) 
        timeshift_select_1T = torch.cat((torch.zeros(1, dtype=torch.long, device=in_TC.device), torch.arange(start=0, end=T - 1, dtype=torch.long, device=in_TC.device))).unsqueeze(0)
        skip_1T = torch.full((T,), fill_value=0, dtype=torch.bool, device=in_TC.device).unsqueeze(0)
        global_in_TC = card_encoding_TC[perm_inv_T]
        global_out_TC = self.global_encoder(global_in_TC.unsqueeze(0), timeshift_select_1T, skip_1T).squeeze(0)
        return global_out_TC

class ToFSRSParams(ModuleType):
    def __init__(self, in_dim):
        super().__init__()
        self.fsrs_ln = torch.nn.LayerNorm(in_dim)
        self.fsrs_linear = torch.nn.Linear(in_dim, 21)
        torch.nn.init.zeros_(self.fsrs_linear.weight)
        torch.nn.init.zeros_(self.fsrs_linear.bias)

    @FunctionType
    def transform_bounded(self, param, min: float, max: float, middle: float):
        return min + (max - min) * torch.sigmoid(param + torch.logit(torch.tensor((middle - min) / (max - min), device=param.device)))

    @FunctionType
    def transform_bounded_exp(self, param, min: float, max: float, middle: float):
        device = param.device
        return torch.exp(self.transform_bounded(param, torch.log(torch.tensor(min, device=device)), torch.log(torch.tensor(max, device=device)), torch.log(torch.tensor(middle, device=device))))

    @FunctionType
    def forward(self, x):
        w = self.fsrs_linear(self.fsrs_ln(x.float()))
        w[:, 0] = self.transform_bounded_exp(w[:, 0], 0.001, 100.0, 0.22)
        w[:, 1] = self.transform_bounded_exp(w[:, 1], 0.001, 100.0, 1.17)
        w[:, 2] = self.transform_bounded_exp(w[:, 2], 0.001, 100.0, 3.26)
        w[:, 3] = self.transform_bounded_exp(w[:, 3], 0.001, 100.0, 16.15)
        w[:, 4] = self.transform_bounded(w[:, 4], 1.0, 10.0, 7.0)
        w[:, 5] = self.transform_bounded(w[:, 5], 0.001, 4.0, 0.57)
        w[:, 6] = self.transform_bounded(w[:, 6], 0.001, 4.0, 2.10)
        w[:, 7] = self.transform_bounded_exp(w[:, 7], 0.001, 0.75, 0.0069)
        w[:, 8] = self.transform_bounded(w[:, 8], 0.0, 4.5, 1.52)
        w[:, 9] = self.transform_bounded(w[:, 9], 0.0, 0.8, 0.11)
        w[:, 10] = self.transform_bounded(w[:, 10], 0.001, 3.5, 1.02)
        w[:, 11] = self.transform_bounded(w[:, 11], 0.001, 5.0, 1.85)
        w[:, 12] = self.transform_bounded(w[:, 12], 0.001, 0.25, 0.11)
        w[:, 13] = self.transform_bounded(w[:, 13], 0.001, 0.9, 0.31)
        w[:, 14] = self.transform_bounded(w[:, 14], 0.0, 4.0, 2.29)
        w[:, 15] = self.transform_bounded(w[:, 15], 0.0, 1.0, 0.22)
        w[:, 16] = self.transform_bounded(w[:, 16], 1.0, 6.0, 3.0)
        w[:, 17] = self.transform_bounded(w[:, 17], 0.0, 2.0, 0.75)
        w[:, 18] = self.transform_bounded(w[:, 18], 0.0, 2.0, 0.33)
        w[:, 19] = self.transform_bounded(w[:, 19], 0.0, 0.8, 0.14)
        w[:, 20] = self.transform_bounded_exp(w[:, 20], 0.1, 0.8, 0.2)
        return w


class FSRSSummaryModel(ModuleType):
    def __init__(self):
        super().__init__()
        self.core = SummaryCore()
        self.fsrs_layer = ToFSRSParams(in_dim=21)
    
    @FunctionType
    def forward(self, in_TC, indices_I, perm_T, perm_inv_T):
        return self.fsrs_layer(self.core(in_TC, indices_I, perm_T, perm_inv_T))

    def is_excluded(self, name):
        DTYPE_EXCLUDE = [
            "fsrs_ln",
            "fsrs_linear",
        ]
        for query in DTYPE_EXCLUDE:
            if query in name:
                return True
        return False

    def copy_downcast_(self, master_model, dtype):
        master_params = dict(master_model.named_parameters())
        with torch.no_grad():
            for name, param in self.named_parameters():
                target_dtype = torch.float32 if self.is_excluded(name) else dtype
                assert param.dtype == target_dtype
                param.data.copy_(master_params[name].to(target_dtype))
                assert param.dtype == target_dtype

    def selective_cast(self, dtype):
        for name, module in self.named_modules():
            if len(name) == 0:
                # Skip the root module
                continue
            if not self.is_excluded(name):
                if dtype == torch.bfloat16:
                    module = module.to(dtype)
                elif dtype == torch.half:
                    raise ValueError("not tested.")
                elif dtype == torch.float32:
                    pass
        return self
