"""FSRS in terms of mostly pure functions. Gradients are still tracked."""

from typing import Optional, Union
from torch import Tensor
import torch

def __nop(ob):
    return ob

ModuleType = torch.nn.Module
FunctionType = __nop

# ModuleType = torch.jit.ScriptModule
# FunctionType = torch.jit.script_method

class FSRS6(ModuleType):
    def __init__(self):
        super().__init__()
        self.initial_params = torch.tensor([
            0.212,
            1.2931,
            2.3065,
            8.2956,
            6.4133,
            0.8334,
            3.0194,
            0.001,
            1.8722,
            0.1666,
            0.796,
            1.4835,
            0.0614,
            0.2629,
            1.6483,
            0.6014,
            1.8729,
            0.5425,
            0.0912,
            0.0658,
            0.1542,
        ])
        self.default_params_stddev_tensor = torch.tensor(
            [
                6.43,
                9.66,
                17.58,
                27.85,
                0.57,
                0.28,
                0.6,
                0.12,
                0.39,
                0.18,
                0.33,
                0.3,
                0.09,
                0.16,
                0.57,
                0.25,
                1.03,
                0.31,
                0.32,
                0.14,
                0.27,
            ]
        )
        self.S_MIN = 0.001
        self.INIT_S_MAX = 100.0

    @FunctionType
    def forgetting_curve(self, t, s, decay):
        factor = 0.9 ** (1 / decay) - 1
        return (1 + factor * t / s) ** decay

    @FunctionType
    def stability_after_success(
        self, w: Tensor, state: Tensor, r: Tensor, rating: Tensor
    ) -> Tensor:
        hard_penalty = torch.where(rating == 2, w[15], 1)
        easy_bonus = torch.where(rating == 4, w[16], 1)
        new_s = state[:, 0] * (
            1
            + torch.exp(w[8])
            * (11 - state[:, 1])
            * torch.pow(state[:, 0], -w[9])
            * (torch.exp((1 - r) * w[10]) - 1)
            * hard_penalty
            * easy_bonus
        )
        return new_s

    @FunctionType
    def stability_after_failure(self, w: Tensor, state: Tensor, r: Tensor) -> Tensor:
        old_s = state[:, 0]
        new_s = (
            w[11]
            * torch.pow(state[:, 1], -w[12])
            * (torch.pow(old_s + 1, w[13]) - 1)
            * torch.exp((1 - r) * w[14])
        )
        new_minimum_s = old_s / torch.exp(w[17] * w[18])
        return torch.minimum(new_s, new_minimum_s)

    @FunctionType
    def stability_short_term(self, w: Tensor, state: Tensor, rating: Tensor) -> Tensor:
        sinc = torch.exp(w[17] * (rating - 3 + w[18])) * torch.pow(
            state[:, 0], -w[19]
        )
        new_s = state[:, 0] * torch.where(rating >= 3, sinc.clamp(min=1), sinc)
        return new_s

    @FunctionType
    def init_d(self, w: Tensor, rating: Tensor) -> Tensor:
        new_d = w[4] - torch.exp(w[5] * (rating - 1)) + 1
        return new_d

    @FunctionType
    def linear_damping(self, delta_d: Tensor, old_d: Tensor) -> Tensor:
        return delta_d * (10 - old_d) / 9

    @FunctionType
    def next_d(self, w: Tensor, state: Tensor, rating: Tensor) -> Tensor:
        delta_d = -w[6] * (rating - 3)
        new_d = state[:, 1] + self.linear_damping(delta_d, state[:, 1])
        new_d = self.mean_reversion(w, self.init_d(w, torch.tensor(4, device=state.device)), new_d)
        return new_d

    @FunctionType
    def step(self, w: Tensor, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            # first learn, init memory states
            new_s = torch.ones_like(state[:, 0])
            new_s = w[X[:, 1].long() - 1]
            new_d = self.init_d(w, X[:, 1])
        else:
            r = self.forgetting_curve(X[:, 0], state[:, 0], -w[20])
            short_term = X[:, 0] < 1
            success = X[:, 1] > 1
            new_s = torch.where(
                short_term,
                self.stability_short_term(w, state, X[:, 1]),
                torch.where(
                    success,
                    self.stability_after_success(w, state, r, X[:, 1]),
                    self.stability_after_failure(w, state, r),
                ),
            )
            new_d = self.next_d(w, state, X[:, 1])
        new_s = new_s.clamp(self.S_MIN, 36500)
        new_d = new_d.clamp(1, 10)
        return torch.stack([new_s, new_d], dim=1)

    @FunctionType
    def mean_reversion(self, w, init: Tensor, current: Tensor) -> Tensor:
        return w[7] * init + (1 - w[7]) * current

    @FunctionType
    def forward(
        self, 
        parameters: Tensor, 
        feature_elapsed_days_real_bl: Tensor,
        feature_elapsed_days_int_bl: Tensor,
        feature_rating: Tensor,
        label_elapsed_days_real_bl: Tensor,
        label_elapsed_days_int_bl: Tensor, 
        state_b2: Optional[Tensor] = None,
    ) -> Tensor:
        """
        :param inputs: shape[seq_len, batch_size, 2]
        """
        inputs_bl2 = torch.stack((feature_elapsed_days_int_bl, feature_rating), dim=-1)
        if state_b2 is None:
            state_b2 = torch.zeros((inputs_bl2.size(0), 2), device=inputs_bl2.device)
        outputs_b2_list = []
        for X in inputs_bl2.transpose(0, 1):
            state_b2 = self.step(parameters, X, state_b2)
            outputs_b2_list.append(state_b2)
        output_tensor_bl2 = torch.stack(outputs_b2_list).transpose(0, 1)
        output_s_bl, output_d_bl = output_tensor_bl2.unbind(dim=-1)
        return self.forgetting_curve(label_elapsed_days_int_bl, output_s_bl, -parameters[20])

    def clip(self, w: Tensor):
        with torch.no_grad():
            w[0] = w[0].clamp(self.S_MIN, self.INIT_S_MAX)
            w[1] = w[1].clamp(self.S_MIN, self.INIT_S_MAX)
            w[2] = w[2].clamp(self.S_MIN, self.INIT_S_MAX)
            w[3] = w[3].clamp(self.S_MIN, self.INIT_S_MAX)
            w[4] = w[4].clamp(1, 10)
            w[5] = w[5].clamp(0.001, 4)
            w[6] = w[6].clamp(0.001, 4)
            w[7] = w[7].clamp(0.001, 0.75)
            w[8] = w[8].clamp(0, 4.5)
            w[9] = w[9].clamp(0, 0.8)
            w[10] = w[10].clamp(0.001, 3.5)
            w[11] = w[11].clamp(0.001, 5)
            w[12] = w[12].clamp(0.001, 0.25)
            w[13] = w[13].clamp(0.001, 0.9)
            w[14] = w[14].clamp(0, 4)
            w[15] = w[15].clamp(0, 1)
            w[16] = w[16].clamp(1, 6)
            w[17] = w[17].clamp(0, 2)
            w[18] = w[18].clamp(0, 2)
            w[19] = w[19].clamp(0, 0.8)
            w[20] = w[20].clamp(0.1, 0.8)
            return w

    def get_regularization_loss(self, params: Tensor, real_batch_size) -> Tensor:
        return (
            torch.sum(
                torch.square(params - self.initial_params)
                / torch.square(self.default_params_stddev_tensor)
            )
            * real_batch_size
        )
