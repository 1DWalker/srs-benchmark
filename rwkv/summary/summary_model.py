

import random
import time
import torch

from rwkv.model.rwkv_model import RWKV7, RWKV7Config
from rwkv.utils import get_number_of_trainable_parameters


class LSTMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = torch.nn.LSTM(input_size=3, hidden_size=32, num_layers=2, batch_first=True)

    def forward(self, x):
        return self.nn(x)

class RWKVModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = RWKV7Config(d_model=32*1, n_heads=1, n_layers=2, channel_mixer_factor=1, layer_offset=0, total_layers=2, decay_lora=4, a_lora=4, v0_mix_amt_lora=4, gate_lora=8, dropout=0.0, dropout_layer=0.0)
        self.encode = torch.nn.Linear(3, 32)
        self.nn = RWKV7(config)

    def forward(self, x, timeshift, skip):
        return self.nn(self.encode(x), timeshift, skip)


def main():
    # DEVICE = torch.device("cuda")
    DEVICE = torch.device("cpu")
    torch.set_num_threads(1)
    random.seed(123)
    torch.manual_seed(123)
    model = RWKVModel().to(DEVICE)
    # model = LSTMModel().to(DEVICE)
    inp = []
    n = 0
    while n < 1e6:
        m = random.randint(1, 20)
        inp.append(torch.randn((m, 3), requires_grad=False, device="cpu").to(DEVICE))
        n += m

    time_start = time.time()
    for iter in range(1):
        with torch.inference_mode():
            # packed = torch.nn.utils.rnn.pack_sequence(inp, enforce_sorted=False)
            # # print(packed)
            # for x in inp:
            #     out = model(x)
            # out = model(packed)
            # print(out)

            cat = torch.cat(inp, dim=0).unsqueeze(0)
            assert n == cat.size(1)
            arr = [0]
            for i in range(n-1):
                arr.append(i)
            timeshift_select = torch.tensor(arr, dtype=torch.long, device="cpu").unsqueeze(0)
            skip = torch.full_like(timeshift_select, 0, dtype=torch.bool)
            timeshift_select = timeshift_select.to(DEVICE)
            skip = skip.to(DEVICE)
            print(cat.shape)
            out = model(cat, timeshift_select, skip)
            print(out)
            # print(cat)

        
        print(iter, "elapsed", time.time() - time_start)

    num_params = get_number_of_trainable_parameters(model)
    print("num params:", num_params)

if __name__ == '__main__':
    main()