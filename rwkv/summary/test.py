import random
import time
import torch

from rwkv.model.rwkv_packed_sequence_model import RWKV7PackedConfig, RWKVPacked
from rwkv.utils import get_number_of_trainable_parameters
# from rwkv.model.rwkv_model import RWKV7, RWKV7Config
# from rwkv.utils import get_number_of_trainable_parameters


def main():
    DEVICE = torch.device("cuda")
    config = RWKV7PackedConfig(d_model=64, n_heads=2, n_layers=2, channel_mixer_factor=1.0, decay_lora=4, a_lora=4, v0_mix_amt_lora=4, gate_lora=8)
    model = RWKVPacked(config).to(DEVICE)
    print("params", get_number_of_trainable_parameters(model))

    in_list = [torch.randn((32, 64), device=DEVICE), torch.randn((8800, 64), device=DEVICE)]
    in_tensor = torch.cat(in_list, dim=0)
    indices_list = []
    t = 0
    for tensor in in_list:
        indices_list.append(t)
        t += tensor.size(0)
    indices = torch.tensor(indices_list, dtype=torch.long, device=DEVICE)
    out = model(in_tensor, indices)
    print(out)
    print("out stats", out.min(), out.mean(), out.max())
    print(out.shape)
    print("done.")

if __name__ == '__main__':
    main()