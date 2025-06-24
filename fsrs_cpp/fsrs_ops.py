import torch
from dataclasses import dataclass
from fsrs_cpp import _FSRS_CPP

@dataclass
class RevlogTensors:
    packed_review_th_T: torch.Tensor
    packed_rating_T: torch.Tensor
    packed_elapsed_days_real_T: torch.Tensor
    packed_elapsed_days_int_T: torch.Tensor
    packed_label_elapsed_days_real_T: torch.Tensor
    packed_label_elapsed_days_int_T: torch.Tensor
    perm_T_tensor: torch.Tensor
    perm_inv_T_tensor: torch.Tensor
    card_locs_T: torch.Tensor

    def flatten(self):
        return (
            self.packed_review_th_T,
            self.packed_rating_T,
            self.packed_elapsed_days_real_T,
            self.packed_elapsed_days_int_T,
            self.packed_label_elapsed_days_real_T,
            self.packed_label_elapsed_days_int_T,
            self.perm_T_tensor,
            self.perm_inv_T_tensor,
            self.card_locs_T,
        )

class FSRSCppFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, review_ths_B, revlog_tensors):
        print(params)
        print(review_ths_B)
        out_B, checkpoint = torch.ops.fsrs.fsrs_batch_forward(params, review_ths_B, *revlog_tensors.flatten())
        ctx.save_for_backward(params, review_ths_B, checkpoint, *revlog_tensors.flatten())
        return out_B

    @staticmethod
    def backward(ctx):
        (
            params,
            review_ths,
            checkpoint,
            packed_review_th_T,
            packed_rating_T,
            packed_elapsed_days_real_T,
            packed_elapsed_days_int_T,
            packed_label_elapsed_days_real_T,
            packed_label_elapsed_days_int_T,
            perm_T_tensor,
            perm_inv_T_tensor,
            card_locs_T,
        ) = ctx.saved_tensors