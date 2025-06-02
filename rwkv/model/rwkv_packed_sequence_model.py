"""
To handle card parallelism, we implement a version of RWKV that can handle a PackedSequence-like data structure.
"""

from dataclasses import dataclass
import math
import torch

from rwkv.model.rwkv_ops import RWKV7_Packed_WKV, reference_rwkv7_packed


torch.manual_seed(2025)

def __nop(ob):
    return ob


ModuleType = torch.nn.Module
FunctionType = __nop

# ModuleType = torch.jit.ScriptModule
# FunctionType = torch.jit.script_method

@dataclass
class RWKV7PackedConfig:
    d_model: int
    n_heads: int
    n_layers: int
    channel_mixer_factor: float
    decay_lora: int
    a_lora: int  # a = in-context learning rate
    v0_mix_amt_lora: int
    gate_lora: int

class RWKVPacked(torch.nn.Module):
    def __init__(self, config: RWKV7PackedConfig):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                RWKV7PackedLayer(config, layer_id)
                for layer_id in range(config.n_layers)
            ]
        )

    @FunctionType
    def forward(self, in_TC, indices_I):
        T, C = in_TC.shape
        x_TC, v0_TC = in_TC, torch.empty_like(in_TC)
        time_shift_select_T = torch.arange(T, device=in_TC.device, requires_grad=False) - 1
        time_shift_select_T[indices_I] = indices_I
        for _, block in enumerate(self.blocks):
            x_TC, v0_TC = block(
                in_TC=x_TC,
                indices_I=indices_I,
                v0_TC=v0_TC,
                time_shift_select_T = time_shift_select_T,
            )
        return x_TC


class RWKV7PackedLayer(ModuleType):
    def __init__(self, config: RWKV7PackedConfig, layer_id):
        super().__init__()
        self.time_mixer = RWKV7PackedTimeMixer(config, layer_id)
        self.channel_mixer = RWKV7PackedChannelMixer(config, layer_id)

    @FunctionType
    def forward(self, in_TC, indices_I, v0_TC, time_shift_select_T):
        x_TC, v0_TC = self.time_mixer(
            in_TC=in_TC,
            indices_I=indices_I,
            v0_TC=v0_TC,
            time_shift_select_T=time_shift_select_T,
        )
        return (
            self.channel_mixer(x_TC, time_shift_select_T),
            v0_TC,
        )

class RWKV7PackedChannelMixer(ModuleType):
    def __init__(self, config: RWKV7PackedConfig, layer_id):
        super().__init__()
        assert config.d_model // config.n_heads == 32
        self.d_model = config.d_model
        with torch.no_grad():
            ratio_1_to_almost_0 = 1.0 - (layer_id / config.n_layers)
            self.layer_norm = torch.nn.LayerNorm(config.d_model)
            self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))

            channel_ratio = torch.ones(1, config.d_model)
            for i in range(config.d_model):
                channel_ratio[0, i] = i / config.d_model

            self.lerp_k = torch.nn.Parameter(
                1 - torch.pow(channel_ratio, ratio_1_to_almost_0**4)
            )

            k_dim = int(config.channel_mixer_factor * config.d_model)
            self.W_k = torch.nn.Linear(config.d_model, k_dim, bias=False)
            self.W_v = torch.nn.Linear(k_dim, config.d_model, bias=False)

            self.W_k.weight.data.uniform_(
                -0.5 / (config.d_model**0.5), 0.5 / (config.d_model**0.5)
            )
            self.W_v.weight.data.zero_()

    @FunctionType
    def forward(self, in_TC, time_shift_select_T):
        x_TC = self.layer_norm(in_TC)
        x_shift_TC = torch.gather(
            x_TC,
            dim=0,
            index=time_shift_select_T.unsqueeze(-1).expand(-1, self.d_model),
        )
        k_TK = self.W_k(torch.lerp(x_TC, x_shift_TC, self.lerp_k))
        o_TC = self.W_v(torch.square(torch.nn.functional.relu(k_TK)))
        return in_TC + o_TC


def ortho_init(x, scale):
    with torch.no_grad():
        shape = x.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
            torch.nn.init.orthogonal_(x, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
            for i in range(shape[0]):
                torch.nn.init.orthogonal_(x[i], gain=gain * scale)
        else:
            assert False
        return x


class LoraSimple(ModuleType):
    def __init__(self, name, d_model, d_lora, layer_id):
        super().__init__()
        with torch.no_grad():
            # The lambda term can be written out as a linear layer that includes a bias
            self.A = torch.nn.Linear(d_model, d_lora, bias=False)
            torch.nn.init.zeros_(self.A.weight)
            self.B_and_lamb = torch.nn.Linear(d_lora, d_model, bias=True)
            ortho_init(self.B_and_lamb.weight, scale=0.1)
            if name == "v":
                # Bias with ones to let the first layer's value flow directly
                torch.nn.init.ones_(self.B_and_lamb.bias)
            else:
                torch.nn.init.zeros_(self.B_and_lamb.bias)

    @FunctionType
    def forward(self, in_BTC):
        return self.B_and_lamb(self.A(in_BTC))


class LoraMLP(ModuleType):
    def __init__(self, name, config: RWKV7PackedConfig, d_lora, out_dim, layer_id):
        super().__init__()
        C = out_dim
        ratio_0_to_1 = layer_id / (config.n_layers - 1)

        with torch.no_grad():
            self.A = torch.nn.Linear(config.d_model, d_lora, bias=False)
            torch.nn.init.zeros_(self.A.weight)
            self.B_and_lamb = torch.nn.Linear(d_lora, out_dim, bias=True)
            ortho_init(self.B_and_lamb.weight, scale=0.1)
            if name == "d":
                decay_speed = torch.ones(C)
                for i in range(C):
                    decay_speed[i] = -7 + 5 * (i / (C - 1)) ** (
                        0.85 + 1.0 * ratio_0_to_1**0.5
                    )
                self.B_and_lamb.bias.copy_(decay_speed + 0.5)
            else:
                torch.nn.init.zeros_(self.B_and_lamb.bias)

    @FunctionType
    def forward(self, in_BTC):
        return self.B_and_lamb(torch.nn.functional.tanh(self.A(in_BTC)))


class RWKV7PackedTimeMixer(ModuleType):
    def __init__(self, config: RWKV7PackedConfig, layer_id):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.layer_id = layer_id
        C = config.d_model
        self.d_model = C
        self.H = config.n_heads
        self.K = C // config.n_heads

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layers - 1)
            ratio_1_to_almost_0 = 1.0 - (layer_id / config.n_layers)
            channel_ratio = torch.ones(1, C)
            for i in range(C):
                channel_ratio[0, i] = i / C

            self.layer_norm = torch.nn.LayerNorm(config.d_model)
            self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))

            self.rkvdag_lerp = torch.nn.Parameter(torch.empty(8, 1, config.d_model))

            # Overall, the earlier the layer the more that we care about the shifted input.
            self.rkvdag_lerp[0] = 1.0 - torch.pow(
                channel_ratio, 0.2 * ratio_1_to_almost_0
            )  # r
            # The weight for k, v, can become negative and are roughly centered around 0 for the later layers.
            self.rkvdag_lerp[1] = 1.0 - (
                torch.pow(channel_ratio, 0.9 * ratio_1_to_almost_0) + 0.4 * ratio_0_to_1
            )  # k
            self.rkvdag_lerp[2] = 1.0 - (
                torch.pow(channel_ratio, 0.2 * ratio_1_to_almost_0) + 0.6 * ratio_0_to_1
            )  # v
            self.rkvdag_lerp[3] = 1.0 - torch.pow(
                channel_ratio, 0.9 * ratio_1_to_almost_0
            )  # d (aka w)
            self.rkvdag_lerp[4] = 1.0 - torch.pow(
                channel_ratio, 0.9 * ratio_1_to_almost_0
            )  # a
            self.rkvdag_lerp[5] = 1.0 - torch.pow(
                channel_ratio, 0.2 * ratio_1_to_almost_0
            )  # g
            self.rkvdag_lerp[6] = 1.0 - torch.pow(
                channel_ratio, 0.9 * ratio_1_to_almost_0
            )
            self.rkvdag_lerp[7] = 1.0 - torch.pow(
                channel_ratio, 0.9 * ratio_1_to_almost_0
            )

            self.bonus = torch.nn.Parameter(
                torch.zeros(1, config.n_heads, config.d_model // config.n_heads)
            )  # r_k

            self.W_r = torch.nn.Linear(config.d_model, config.d_model, bias=False)
            self.W_k = torch.nn.Linear(config.d_model, config.d_model, bias=False)
            self.W_v = torch.nn.Linear(config.d_model, config.d_model, bias=False)
            self.W_o = torch.nn.Linear(config.d_model, config.d_model, bias=False)

            self.W_r.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.W_k.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
            self.W_v.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.W_o.weight.data.zero_()

            self.k_scale_linear = torch.nn.Linear(config.d_model, self.H, bias=True)
            self.v_scale_linear = torch.nn.Linear(config.d_model, self.H, bias=True)
            torch.nn.init.zeros_(self.k_scale_linear.weight)
            torch.nn.init.zeros_(self.k_scale_linear.bias)
            torch.nn.init.zeros_(self.v_scale_linear.weight)
            torch.nn.init.zeros_(self.v_scale_linear.bias)

            self.v_lora_simple = LoraSimple(
                name="v",
                d_model=config.d_model,
                d_lora=config.v0_mix_amt_lora,
                layer_id=layer_id,
            )
            self.a_lora_simple = LoraSimple(
                name="a",
                d_model=config.d_model,
                d_lora=config.a_lora,
                layer_id=layer_id,
            )
            self.d_lora_mlp = LoraMLP(
                name="d",
                config=config,
                d_lora=config.decay_lora,
                out_dim=config.d_model,
                layer_id=layer_id,
            )

            self.lora_A_g = torch.nn.Linear(
                config.d_model, config.gate_lora, bias=False
            )
            torch.nn.init.zeros_(self.lora_A_g.weight)
            self.lora_B_g = torch.nn.Linear(
                config.gate_lora, config.d_model, bias=False
            )
            ortho_init(self.lora_B_g.weight, 0.1)

            self.out_group_norm = torch.nn.GroupNorm(
                config.n_heads, config.d_model, eps=64e-5
            )

    @FunctionType
    def forward(self, in_TC, indices_I, v0_TC, time_shift_select_T):
        T, C = in_TC.shape
        H, K = self.H, self.K

        x_TC = self.layer_norm(in_TC)
        x_shift_TC = torch.gather(
            x_TC,
            dim=0,
            index=time_shift_select_T.unsqueeze(-1).expand(-1, self.d_model),
        )

        rkvdag_8TC = torch.lerp(
            x_TC.unsqueeze(0), x_shift_TC.unsqueeze(0), self.rkvdag_lerp
        )
        r_TC, k_TC, v_TC, d_TC, a_TC, g_TC, k_scale_TC, v_scale_TC = (
            rkvdag_8TC.unbind(dim=0)
        )
        r_TC = self.W_r(r_TC)
        k_TC = self.W_k(k_TC)
        k_scale_TH = torch.nn.functional.sigmoid(self.k_scale_linear(k_scale_TC))
        v_scale_TH = torch.nn.functional.sigmoid(self.v_scale_linear(v_scale_TC))

        if self.layer_id == 0:
            v_TC = self.W_v(v_TC)
            v0_TC = v_TC
        else:
            v_lerp_TC = torch.nn.functional.sigmoid(self.v_lora_simple(v_TC))
            v_TC = torch.lerp(self.W_v(v_TC), v0_TC, v_lerp_TC)

        a_TC = torch.nn.functional.sigmoid(self.a_lora_simple(a_TC))
        g_TC = self.lora_B_g(torch.nn.functional.sigmoid(self.lora_A_g(g_TC)))

        _d_TC = -0.5 - torch.nn.functional.softplus(-self.d_lora_mlp(d_TC))
        w_TC = torch.exp(-torch.exp(_d_TC.float()))

        k_THK = k_scale_TH.unsqueeze(-1) * torch.nn.functional.normalize(
            k_TC.view(T, H, K), dim=-1, p=2.0
        )
        r_THK = r_TC.view(T, H, K)
        v_THK = v_scale_TH.unsqueeze(-1) * torch.nn.functional.normalize(
            v_TC.view(T, H, K), dim=-1, p=2.0
        )
        w_THK = w_TC.view(T, H, K)
        a_THK = a_TC.view(T, H, K)
        k_deformed_THK = k_THK
        k_THK = k_THK * a_THK

        # TODO
        out_THK = RWKV7_Packed_WKV.apply(
            indices_I, r_THK, k_THK, v_THK, w_THK, a_THK, k_deformed_THK
        )
        # out_THK = reference_rwkv7_packed(
        #     indices_I, r_THK, k_THK, v_THK, w_THK, a_THK, k_deformed_THK
        # )

        out_TC = self.out_group_norm(out_THK.view(T, C)).view(T, C)
        bonus_TC = (
            (r_THK * self.bonus * k_THK).sum(dim=-1, keepdim=True) * v_THK
        ).view(T, C)
        out_TC = self.W_o(g_TC * (out_TC + bonus_TC))
        return in_TC + out_TC, v0_TC