"""FSRS in terms of pure functions. Not pure in terms of gradient tracking."""

from typing import Optional, Union
from torch import Tensor
import torch

def __nop(ob):
    return ob

# ModuleType = torch.nn.Module
# FunctionType = __nop

ModuleType = torch.jit.ScriptModule
FunctionType = torch.jit.script_method

class FSRS6(ModuleType):
    def __init__(self):
        super().__init__()
        self.S_MIN = 0.001

    @FunctionType
    def forgetting_curve(self, t_bl, s_hbl, decay_h):
        factor_h = 0.9 ** (1 / decay_h) - 1
        return (1 + factor_h.view(-1, 1, 1) * t_bl.unsqueeze(0) / s_hbl) ** decay_h.view(-1, 1, 1)

    @FunctionType
    def stability_after_success(
        self, w_hp: Tensor, state_hb2: Tensor, r_hb: Tensor, rating_b: Tensor
    ) -> Tensor:
        H = w_hp.size(0)
        rating_hb = rating_b.unsqueeze(0).expand(H, -1)
        hard_penalty_hb = torch.where(rating_hb == 2, w_hp[:, 15].unsqueeze(-1), 1)
        easy_bonus_hb = torch.where(rating_hb == 4, w_hp[:, 16].unsqueeze(-1), 1)
        new_s_hb = state_hb2[:, :, 0] * (
            1
            + torch.exp(w_hp[:, 8].unsqueeze(-1))
            * (11 - state_hb2[:, :, 1])
            * torch.pow(state_hb2[:, :, 0], -w_hp[:, 9].unsqueeze(-1))
            * (torch.exp((1 - r_hb) * w_hp[:, 10].unsqueeze(-1)) - 1)
            * hard_penalty_hb
            * easy_bonus_hb
        )
        return new_s_hb

    @FunctionType
    def stability_after_failure(self, w_hp: Tensor, state_hb2: Tensor, r_hb: Tensor) -> Tensor:
        old_s_hb = state_hb2[:, :, 0]
        new_s_hb = (
            w_hp[:, 11].unsqueeze(-1)
            * torch.pow(state_hb2[:, :, 1], -w_hp[:, 12].unsqueeze(-1))
            * (torch.pow(old_s_hb + 1, w_hp[:, 13].unsqueeze(-1)) - 1)
            * torch.exp((1 - r_hb) * w_hp[:, 14].unsqueeze(-1))
        )
        new_minimum_s_hb = old_s_hb / torch.exp(w_hp[:, 17] * w_hp[:, 18]).unsqueeze(-1)
        return torch.minimum(new_s_hb, new_minimum_s_hb)

    @FunctionType
    def stability_short_term(self, w_hp: Tensor, state_hb2: Tensor, rating_b: Tensor) -> Tensor:
        sinc_hb = torch.exp(w_hp[:, 17].unsqueeze(-1) * (rating_b.unsqueeze(0) - 3 + w_hp[:, 18].unsqueeze(-1))) * torch.pow(
            state_hb2[:, :, 0], -w_hp[:, 19].unsqueeze(-1)
        )
        H = w_hp.size(0)
        rating_hb = rating_b.unsqueeze(0).expand(H, -1)
        new_s_hb = state_hb2[:, :, 0] * torch.where(rating_hb >= 3, sinc_hb.clamp(min=1), sinc_hb)
        return new_s_hb

    @FunctionType
    def init_d(self, w_hp: Tensor, rating_b: Tensor) -> Tensor:
        new_d_hb = w_hp[:, 4].unsqueeze(1) - torch.exp(w_hp[:, 5].unsqueeze(1) * (rating_b.unsqueeze(0) - 1)) + 1
        return new_d_hb

    @FunctionType
    def linear_damping(self, delta_d_hb: Tensor, old_d_hb: Tensor) -> Tensor:
        return delta_d_hb * (10 - old_d_hb) / 9

    @FunctionType
    def next_d(self, w_hp: Tensor, state_hb2: Tensor, rating_b: Tensor) -> Tensor:
        B = rating_b.size(0)
        delta_d_hb = -w_hp[:, 6].unsqueeze(-1) * (rating_b.unsqueeze(0) - 3)
        new_d_hb = state_hb2[:, :, 1] + self.linear_damping(delta_d_hb, state_hb2[:, :, 1])
        new_d_hb = self.mean_reversion(w_hp, self.init_d(w_hp, torch.tensor([4], device=state_hb2.device)).expand(-1, B), new_d_hb)
        return new_d_hb

    @FunctionType
    def step(self, w_hp: Tensor, X_b2: Tensor, state_hb2: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty
        :return state:
        """
        H, P = w_hp.shape
        B = X_b2.size(0)
        if torch.equal(state_hb2, torch.zeros_like(state_hb2)):
            # first learn, init memory states
            new_s_hb = torch.ones_like(state_hb2[:, :, 0])
            new_s_hb = w_hp.gather(dim=1, index=(X_b2[:, 1].long() - 1).unsqueeze(0).expand(H, -1))
            new_d_hb = self.init_d(w_hp, X_b2[:, 1])
            new_d_hb = new_d_hb.clamp(1, 10)
        else:
            r_hb = self.forgetting_curve(X_b2[:, 0].unsqueeze(-1), state_hb2[:, :, 0].unsqueeze(-1), -w_hp[:, 20]).squeeze(-1)
            short_term_b = X_b2[:, 0] < 1
            success_b = X_b2[:, 1] > 1
            new_s_hb = torch.where(
                short_term_b,
                self.stability_short_term(w_hp, state_hb2, X_b2[:, 1]),
                torch.where(
                    success_b.unsqueeze(0).expand(H, -1),
                    self.stability_after_success(w_hp, state_hb2, r_hb, X_b2[:, 1]),
                    self.stability_after_failure(w_hp, state_hb2, r_hb),
                ),
            )
            new_d_hb = self.next_d(w_hp, state_hb2, X_b2[:, 1])
            new_d_hb = new_d_hb.clamp(1, 10)
        new_s_hb = new_s_hb.clamp(self.S_MIN, 36500)
        return torch.stack([new_s_hb, new_d_hb], dim=-1)

    @FunctionType
    def mean_reversion(self, w_hp, init_hb: Tensor, current_hb: Tensor) -> Tensor:
        return w_hp[:, 7].unsqueeze(-1) * init_hb + (1 - w_hp[:, 7].unsqueeze(-1)) * current_hb

    @FunctionType
    def forward(
        self, 
        parameters_hp: Tensor, 
        feature_elapsed_days_int_bl: Tensor,
        feature_elapsed_days_real_bl: Tensor,
        feature_rating_bl: Tensor,
        label_elapsed_days_int_bl: Tensor, 
    ) -> Tensor:
        """
        parameters_hp: h is the number of parameters sets to evaluate, p is the number of parameters (e.g. 21 for FSRS-6)
        returns: h x b x l tensor of probabilities
        """
        assert len(parameters_hp.shape) == 2
        inputs_bl2 = torch.stack((feature_elapsed_days_int_bl, feature_rating_bl), dim=-1)
        state_hb2 = torch.zeros((parameters_hp.size(0), inputs_bl2.size(0), 2), device=inputs_bl2.device)
        outputs_list = []
        for X in inputs_bl2.transpose(0, 1):
            state_hb2 = self.step(parameters_hp, X, state_hb2)
            outputs_list.append(state_hb2)
        output_tensor_hbl2 = torch.stack(outputs_list).permute(1, 2, 0, 3)
        output_s_hbl, _ = output_tensor_hbl2.unbind(dim=-1)
        return self.forgetting_curve(label_elapsed_days_int_bl, output_s_hbl, -parameters_hp[:, 20])
