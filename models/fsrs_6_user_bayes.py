import json
import logging
import pathlib
import torch
from torch import nn, Tensor

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

        DEFAULT_SIZE = 1
        DEFAULT_PARAMS = [
            0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001,
            1.8722, 0.1666, 0.796, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014,
            1.8729, 0.5425, 0.0912, 0.0658, 0.1542,
        ]

        # read data
        result_file = pathlib.Path("./result/FSRS-6-recency.jsonl")
        with open(result_file, "r") as f:
            data = [json.loads(x) for x in f]

        # Build a dictionary keyed by user
        by_user = {}
        for result in data:
            user = result["user"]
            size = result.get("size", DEFAULT_SIZE)

            if "parameters" in result:
                params = result["parameters"]
                if isinstance(params, list):
                    params = params[0]
                else:
                    # params is a dict -> extend by values
                    params = list(params.values())[0]
            else:
                params = DEFAULT_PARAMS

            by_user[user] = (size, params)

        # Determine max user ID (or use 10000 explicitly)
        max_user = max(by_user.keys())  # or: max_user = 10000
        # max_user = 500

        _sizes = []
        _parameters = []

        for u in range(1, max_user + 1):
            if u in by_user:
                s, p = by_user[u]
            else:
                s = DEFAULT_SIZE
                p = DEFAULT_PARAMS
            _sizes.append(s)
            _parameters.append(p)

        self.register_buffer('sizes', torch.tensor(_sizes, requires_grad=False))
        self.register_buffer('user_parameters', torch.stack([torch.tensor(p, requires_grad=False) for p in _parameters]))

        self.log_belief = nn.Parameter(torch.zeros(self.sizes.size(0), dtype=torch.float32))
        if state_dict is not None:
            self.load_state_dict(state_dict)
        else:
            try:
                self.load_state_dict(
                    torch.load(
                        f"./pretrain/{self.config.get_evaluation_file_name()}_pretrain.pth",
                        weights_only=True,
                        map_location=self.config.device,
                    )
                )
            except FileNotFoundError:
                pass

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

        self.log_belief.zero_()
        for i, batch in enumerate(self.train_data_loader):
            sequences, delta_ts, labels, seq_lens, weights = batch
            H = self.user_parameters.size(0)
            L, B, _ = sequences.shape
            feature_delta_t, feature_rating = sequences.transpose(0, 1).unbind(dim=-1)
            p_hbl = self.fsrs.forward(self.user_parameters, feature_delta_t, None, feature_rating, delta_ts.unsqueeze(-1).expand(-1, L))
            p_hb = torch.take_along_dim(
                p_hbl,                          # (H, B, L)
                (seq_lens - 1).view(1, B, 1),              # (1, B, 1) → broadcast over H
                dim=2,
            ).squeeze(-1)                       # → (H, B)
            review_likelihood_hb = p_hb * labels + (1 - p_hb) * (1 - labels)
            review_log_likelihood_h = (review_likelihood_hb.log() * weights.view(1, -1)).sum(dim=-1)
            self.log_belief.add_(review_log_likelihood_h)

        self.log_belief[user_id - 1] = -1e9
        self.log_belief -= self.log_belief.max()
        values, indices = torch.topk(self.log_belief, k=10)

        # print(self.belief)
        print("users:", indices + 1)
        print("log-likelihood:", values)
        print(f"elapsed {time.time() - start}")

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
        p_hbl = self.fsrs.forward(self.user_parameters, feature_delta_t, None, feature_rating, delta_ts.unsqueeze(-1).expand(-1, L))
        p_hb = torch.take_along_dim(
            p_hbl,                          # (H, B, L)
            (seq_lens - 1).view(1, B, 1),              # (1, B, 1) → broadcast over H
            dim=2,
        ).squeeze(-1)                       # → (H, B)
        belief = torch.nn.functional.softmax(self.log_belief)
        # print(self.log_belief)
        pred = (p_hb * belief.view(-1, 1)).sum(dim=0)
        output = {}
        output["retentions"] = pred
        return output
    
    def get_similar_users(self):
        values, indices = torch.topk(self.log_belief, k=20)
        return 1 + indices, values

