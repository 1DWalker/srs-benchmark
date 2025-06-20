import torch
from dataclasses import dataclass

class RevlogTensors:
    packed_review_th_T: torch.Tensor
    packed_rating_T: torch.Tensor
    packed_elapsed_days_real_T: torch.Tensor
    packed_elapsed_days_int_T: torch.Tensor
    perm_T_tensor: torch.Tensor
    perm_inv_T_tensor: torch.Tensor
    card_locs_T: torch.Tensor


class FSRSCppFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        pass

class FSRSCppAdapter:
    def __init__(self):
        pass

    def run(review_th_B: torch.Tensor, revlog: RevlogTensors):
        pass

class FSRSReferenceAdapter:
    def __init__(self, fsrs):
        self.fsrs = fsrs
    
    def run(review_th_B: torch.Tensor, revlog: RevlogTensors):
        pass