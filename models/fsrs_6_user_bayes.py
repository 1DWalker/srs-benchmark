import json
import logging
import pathlib
import torch
from torch import nn, Tensor
import numpy as np
from config import Config
from models.base import BaseModel
from models.fsrs_6_batched import FSRS6Batched

try:
    from fsrs_optimizer import BatchDataset, BatchLoader # type: ignore
except Exception as e:
    logging.exception("Failed to import fsrs_optimizer: %s", e)

class FSRS6UserBayes(BaseModel):

    def __init__(self, config: Config, state_dict=None):
        super().__init__(config)
        self.fsrs = FSRS6Batched()

        # read data
        result_file = pathlib.Path("./result/FSRS-6-recency.jsonl")
        with open(result_file, "r") as f:
            data = [json.loads(x) for x in f]

        # Build a dictionary keyed by user
        _users = []
        _sizes = []
        _parameters = []
        for result in data:
            _users.append(result["user"])
            _sizes.append(result["size"])
            params = result["parameters"]
            if isinstance(params, list):
                params = params[0]
            else:
                # params is a dict -> extend by values
                params = list(params.values())[0]
            _parameters.append(params)

        self.register_buffer('users', torch.tensor(_users, requires_grad=False))
        self.register_buffer('sizes', torch.tensor(_sizes, requires_grad=False))
        # self.register_buffer('prior', torch.log(self.sizes))
        self.register_buffer('prior', torch.zeros_like(self.sizes))
        self.register_buffer('user_parameters', torch.stack([torch.tensor(p, requires_grad=False) for p in _parameters]))
        self.register_buffer('log_belief', torch.zeros(self.sizes.size(0), dtype=torch.float32))
        if state_dict is not None:
            self.load_state_dict_force(state_dict)
        else:
            try:
                self.load_state_dict_force(
                    torch.load(
                        f"./pretrain/{self.config.get_evaluation_file_name()}_pretrain.pth",
                        weights_only=True,
                        map_location=self.config.device,
                    )
                )
            except FileNotFoundError:
                pass

    def _prune(self, indices):
        self.register_buffer('users', self.users[indices])
        self.register_buffer('sizes', self.sizes[indices])
        self.register_buffer('prior', self.prior[indices])
        self.register_buffer('user_parameters', self.user_parameters[indices])
        self.register_buffer('log_belief', self.log_belief[indices])

    @torch.inference_mode()
    def prune_size(self, keep):
        if keep >= self.log_belief.size(0):
            return

        _, indices = torch.topk(self.log_belief, k=keep)
        self._prune(indices)

    @torch.inference_mode()
    def prune_threshold(self, threshold: float):
        vals = self.log_belief
        keep_mask = vals.max() <= vals + threshold
        indices = keep_mask.nonzero(as_tuple=True)[0]

        if indices.numel() == vals.numel():
            return

        self._prune(indices)

    @torch.inference_mode()
    def fit(self, train_set, user_id):
        train_set["weights"] *= len(train_set) / train_set["weights"].sum()

        self.train_set = BatchDataset(
            train_set.copy(),
            128,
            max_seq_len=64,
            device=torch.device("cpu"),
        )
        self.train_data_loader = BatchLoader(self.train_set)

        import time
        start = time.time()

        non_user_index = torch.where(self.users != user_id)[0]
        self._prune(non_user_index)

        remaining_weight = len(train_set)
        self.log_belief.zero_()
        guaranteed_size = self.log_belief.size(0)
        for i, batch in enumerate(self.train_data_loader):
            sequences, delta_ts, labels, seq_lens, weights = batch
            H = self.user_parameters.size(0)
            L, B, _ = sequences.shape
            feature_delta_t, feature_rating = sequences.transpose(0, 1).unbind(dim=-1)
            p_hb = self.fsrs.forward(self.user_parameters, feature_delta_t, None, feature_rating, delta_ts, seq_lens)
            review_likelihood_hb = p_hb * labels + (1 - p_hb) * (1 - labels)
            review_log_likelihood_h = (review_likelihood_hb.log() * weights.view(1, -1)).sum(dim=-1)
            self.log_belief.add_(review_log_likelihood_h)

            self.log_belief.add_(self.prior, alpha=weights.sum() / len(train_set))
            so_far_weight = len(train_set) - remaining_weight
            keep_mask = self.log_belief + remaining_weight / np.sqrt(so_far_weight + 1) * 0.5 > self.log_belief.max() - 10
            self._prune(keep_mask)

            # guaranteed pruning
            if i > 0 and i % 20 == 0:
                next_guaranteed_size = max(100, int(guaranteed_size * 0.8))
                self.prune_size(next_guaranteed_size)
                guaranteed_size = next_guaranteed_size


            # print(remaining_weight)
            # print(self.log_belief.size(0), guaranteed_size)
            remaining_weight -= weights.sum()
            if self.log_belief.size(0) == 1:
                break
        
        self.prune_threshold(10)

        self.log_belief -= self.log_belief.max()
    
        values, indices = torch.topk(self.log_belief, k=min(20, self.log_belief.size(0)))
        print("user:", user_id, "belief_size:", self.log_belief.size(0), "similar_users:", self.users[indices])
        print("log-likelihood:", values)
        print("log-sizes:", torch.log(self.sizes[indices]))
        print("sizes:", self.sizes[indices])
        print(f"elapsed {time.time() - start}")
        print()

    def batch_process(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        H = self.user_parameters.size(0)
        L, B, _ = sequences.shape
        feature_delta_t, feature_rating = sequences.transpose(0, 1).unbind(dim=-1)
        p_hb = self.fsrs.forward(self.user_parameters, feature_delta_t, None, feature_rating, delta_ts, seq_lens)
        belief = torch.nn.functional.softmax(self.log_belief)
        pred = (p_hb * belief.view(-1, 1)).sum(dim=0)
        output = {}
        output["retentions"] = pred
        return output
    
    def get_similar_users(self, n=30):
        values, indices = torch.topk(self.log_belief, k=min(n, self.log_belief.size(0)))
        return self.users[indices], values

    def get_belief_size(self):
        return self.log_belief.size(0)
    
    def load_state_dict_force(self, state_dict):
        model_state = self.state_dict()

        # Replace mismatched tensors first
        for key, ckpt_tensor in state_dict.items():
            if key not in model_state:
                continue

            current_tensor = model_state[key]

            if current_tensor.shape != ckpt_tensor.shape:
                module_name, _, param_name = key.rpartition('.')

                # Get module containing the parameter/buffer
                mod = self
                if module_name:
                    for part in module_name.split('.'):
                        mod = getattr(mod, part)

                # Replace either parameter or buffer
                if isinstance(mod._parameters.get(param_name, None), torch.nn.Parameter):
                    setattr(mod, param_name, torch.nn.Parameter(ckpt_tensor.clone()))
                else:
                    mod.register_buffer(param_name, ckpt_tensor.clone())

        # Now all shapes match → safe load
        self.load_state_dict(state_dict, strict=True)
