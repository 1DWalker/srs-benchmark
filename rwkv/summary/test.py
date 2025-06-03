import math
import random
import time
import torch

from rwkv.model.rwkv_ops import RWKV7_Packed_WKV, reference_rwkv7_packed
from rwkv.model.rwkv_packed_sequence_model import RWKV7PackedConfig, RWKVPacked
from rwkv.utils import get_number_of_trainable_parameters
# from rwkv.model.rwkv_model import RWKV7, RWKV7Config
# from rwkv.utils import get_number_of_trainable_parameters


def main():
    random.seed(123)
    torch.manual_seed(123)
    device = torch.device("cpu")
    # config = RWKV7PackedConfig(d_model=64, n_heads=2, n_layers=2, channel_mixer_factor=1.0, decay_lora=4, a_lora=4, v0_mix_amt_lora=4, gate_lora=8)
    # model = RWKVPacked(config).to(DEVICE)
    # print("params", get_number_of_trainable_parameters(model))

    # in_list = [torch.randn((32, 64), device=DEVICE), torch.randn((100, 64), device=DEVICE), torch.randn((200, 64), device=DEVICE)]
    # in_tensor = torch.cat(in_list, dim=0)
    # indices_list = []
    # t = 0
    # for tensor in in_list:
    #     indices_list.append(t)
    #     t += tensor.size(0)
    # indices = torch.tensor(indices_list, dtype=torch.long, device=DEVICE)
    # out = model(in_tensor, indices)
    # print(out)
    # print("out stats", out.min(), out.mean(), out.max())
    # print(out.shape)
    # print("done.")
    T = 2000
    H = 2
    K = 32
    indices_list = [0]
    for i in range(1, T):
        if random.randint(0, 32) == 0:
            indices_list.append(i)
    print("indices", indices_list)
    indices_I = torch.tensor(indices_list, dtype=torch.long, device=device)

    # dtype = torch.float16
    dtype = torch.float
    r_THK = torch.randn(T, H, K, dtype=dtype, device=device, requires_grad=True) / math.sqrt(K)
    k_THK = torch.randn(T, H, K, dtype=dtype, device=device, requires_grad=True) / K
    v_THK = torch.randn(T, H, K, dtype=dtype, device=device, requires_grad=True) / math.sqrt(K)
    w_THK = torch.rand(T, H, K, dtype=torch.float32, device=device, requires_grad=True)
    a_THK = torch.rand(T, H, K, dtype=dtype, device=device, requires_grad=True)
    k_deformed_THK = torch.randn(T, H, K, dtype=dtype, device=device, requires_grad=True) / K
    params = [r_THK, k_THK, v_THK, w_THK, a_THK, k_deformed_THK]
    out_reference = reference_rwkv7_packed(indices_I, r_THK, k_THK, v_THK, w_THK, a_THK, k_deformed_THK)
    out_THK = RWKV7_Packed_WKV.apply(
        indices_I, r_THK, k_THK, v_THK, w_THK, a_THK, k_deformed_THK
    )
    torch.testing.assert_close(out_reference, out_THK)
    if device == torch.device("cpu"):
        print("passed forward for cpu. exiting...")
        exit()

    grad_out = 1 * torch.randn_like(out_reference)
    grad_out_copy = grad_out.clone()
    grad_reference = torch.autograd.grad(out_reference, params, grad_out)
    grad = torch.autograd.grad(out_THK, params, grad_out)
    for i in range(6):
        print("param", i)
        print(grad[i])
        print(grad_reference[i])
        print(i, "max error: ", (grad[i] - grad_reference[i]).abs().max())
        print(i, "max rel: ", (grad[i] / grad_reference[i]).abs().max())
    print("indices", indices_list)
    torch.testing.assert_close(grad_reference, grad)
    print("all good!", device)


if __name__ == '__main__':
    main()