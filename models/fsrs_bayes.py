from typing import List, Optional
import torch
from torch import nn, Tensor
from models.fsrs_v6 import FSRS6, FSRS6ParameterClipper
import pandas as pd
import numpy as np
from scipy.optimize import minimize  # type: ignore

class FSRS6BayesParameterClipper(FSRS6ParameterClipper):
    def __init__(self, config):
        self.config = config

    def __call__(self, module):
        if hasattr(module, 'ws'):
            ws = module.ws.data
            for i in range(42):
                # duplicate the original 21-clamp pattern for both halves
                if i % 21 == 0 or i % 21 == 1 or i % 21 == 2 or i % 21 == 3:
                    ws[i] = ws[i].clamp(self.config.s_min, self.config.init_s_max)
                elif i % 21 == 4:
                    ws[i] = ws[i].clamp(1, 10)
                elif i % 21 == 5 or i % 21 == 6:
                    ws[i] = ws[i].clamp(0.001, 4)
                elif i % 21 == 7:
                    ws[i] = ws[i].clamp(0.001, 0.75)
                elif i % 21 == 8:
                    ws[i] = ws[i].clamp(0, 4.5)
                elif i % 21 == 9:
                    ws[i] = ws[i].clamp(0, 0.8)
                elif i % 21 == 10:
                    ws[i] = ws[i].clamp(0.001, 3.5)
                elif i % 21 == 11:
                    ws[i] = ws[i].clamp(0.001, 5)
                elif i % 21 == 12:
                    ws[i] = ws[i].clamp(0.001, 0.25)
                elif i % 21 == 13:
                    ws[i] = ws[i].clamp(0.001, 0.9)
                elif i % 21 == 14:
                    ws[i] = ws[i].clamp(0, 4)
                elif i % 21 == 15:
                    ws[i] = ws[i].clamp(0, 1)
                elif i % 21 == 16:
                    ws[i] = ws[i].clamp(1, 6)
                elif i % 21 == 17 or i % 21 == 18:
                    ws[i] = ws[i].clamp(0, 2)
                elif i % 21 == 19:
                    ws[i] = ws[i].clamp(0, 0.8)
                elif i % 21 == 20:
                    ws[i] = ws[i].clamp(0.1, 0.8)
            module.ws.data = ws


class FSRS6Bayes(FSRS6):
    """
    Doubled FSRS parameters for two independent models with Bayesian weighting.
    No references to FSRS-6/5.
    """

    # Duplicated init arrays (size 42 = 21 * 2)
    init_w = torch.tensor([
        0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001,
        1.8722, 0.1666, 0.796, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014,
        1.8729, 0.5425, 0.0912, 0.0658, 0.1542,
        0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001,
        1.8722, 0.1666, 0.796, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014,
        1.8729, 0.5425, 0.0912, 0.0658, 0.2042,
    ], dtype=torch.float32)

    param_stddev = torch.tensor([
        6.43, 9.66, 17.58, 27.85, 0.57, 0.28, 0.6, 0.12, 0.39, 0.18,
        0.33, 0.3, 0.09, 0.16, 0.57, 0.25, 1.03, 0.31, 0.32, 0.14, 0.27,
        6.43, 9.66, 17.58, 27.85, 0.57, 0.28, 0.6, 0.12, 0.39, 0.18,
        0.33, 0.3, 0.09, 0.16, 0.57, 0.25, 1.03, 0.31, 0.32, 0.14, 0.27,
    ], dtype=torch.float32)

    def __init__(self, config, w: Optional[List[float]] = None, init_belief: Optional[List[float]] = None):
        super().__init__(config)
        self.config = config

        if w is None:
            w = self.init_w.clone()

        self.ws = nn.Parameter(torch.tensor(w).to(self.config.device))  # 42 params

        # if init_belief is None:
        #     init_belief = [0.0, 0.0]
        # self.belief_logits = nn.Parameter(torch.tensor(init_belief, dtype=torch.float32))

        self.initial_w = self.ws.clone().detach()
        self.clipper = FSRS6BayesParameterClipper(config)
        self.w = None

    def split(self):
        w1 = self.ws[:21]
        w2 = self.ws[21:]
        return w1, w2

    def forgetting_curve(self, t, s, decay_param):
        factor = 0.9 ** (1 / decay_param) - 1
        return (1 + factor * t / s) ** decay_param

    def stability_short_term(self, w, state, rating):
        sinc = torch.exp(w[17] * (rating - 3 + w[18])) * torch.pow(state[:, 0], -w[19])
        new_s = state[:, 0] * torch.where(rating >= 2, sinc.clamp(min=1), sinc)
        return new_s

    def stability_after_success(self, w, state, r, rating):
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
    
    def stability_after_failure(self, w, state: Tensor, r: Tensor) -> Tensor:
        old_s = state[:, 0]
        new_s = (
            w[11]
            * torch.pow(state[:, 1], -w[12])
            * (torch.pow(old_s + 1, w[13]) - 1)
            * torch.exp((1 - r) * w[14])
        )
        new_minimum_s = old_s / torch.exp(w[17] * w[18])
        return torch.minimum(new_s, new_minimum_s)

    def init_d(self, w, rating) -> Tensor:
        new_d = w[4] - torch.exp(w[5] * (rating - 1)) + 1
        return new_d
    
    def linear_damping(self, delta_d: Tensor, old_d: Tensor) -> Tensor:
        return delta_d * (10 - old_d) / 9

    def mean_reversion(self, w, init: Tensor, current: Tensor) -> Tensor:
        return w[7] * init + (1 - w[7]) * current

    def next_d(self, w, state: Tensor, rating: Tensor) -> Tensor:
        delta_d = -w[6] * (rating - 3)
        new_d = state[:, 1] + self.linear_damping(delta_d, state[:, 1])
        new_d = self.mean_reversion(w, self.init_d(w, 4), new_d)
        return new_d
    
    def step(self, w, X, state):
        if torch.equal(state, torch.zeros_like(state)):
            keys = torch.tensor([1,2,3,4], device=self.config.device)
            keys = keys.view(1,-1).expand(X[:,1].long().size(0), -1)
            index = (X[:,1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            new_s = torch.ones_like(state[:,0])
            new_s[index[0]] = w[index[1]]
            new_d = self.init_d(w, X[:, 1])
            new_d = new_d.clamp(1, 10)
        else:
            r = self.forgetting_curve(X[:,0], state[:,0], -w[20])
            short_term = X[:,0]<1
            success = X[:,1]>1
            new_s = torch.where(short_term,
                                self.stability_short_term(w,state,X[:,1]),
                                torch.where(success,
                                            self.stability_after_success(w,state,r,X[:,1]),
                                            self.stability_after_failure(w,state,r)))
            new_d = self.next_d(w,state,X[:,1])
            new_d = new_d.clamp(1,10)
        new_s = new_s.clamp(self.config.s_min, 36500)
        return torch.stack([new_s,new_d],dim=1)

    def batch_process(self, sequences, delta_ts, seq_lens, real_batch_size):
        w1,w2 = self.split()
        # print(w1)
        # print(self.state_dict())
        p1,s1,d1 = self._evaluate_model(w1,sequences,delta_ts,seq_lens,real_batch_size)
        p2,s2,d2 = self._evaluate_model(w2,sequences,delta_ts,seq_lens,real_batch_size)

        last_rating_LB = sequences[:, :, 1]
        L, B = last_rating_LB.shape
        success = last_rating_LB > 1
        L1 = torch.where(success,p1,1-p1).clamp(1e-9,1.0)
        L2 = torch.where(success,p2,1-p2).clamp(1e-9,1.0)
        zeros = torch.zeros((1, B))
        log_likelihood1 = torch.concat([zeros, L1.detach().log()], dim=0).cumsum(dim=0)
        log_likelihood2 = torch.concat([zeros, L2.detach().log()], dim=0).cumsum(dim=0)

        log_likelihood_LB2 = torch.stack((log_likelihood1, log_likelihood2), dim=-1)
        posterior_LB2 = torch.softmax(log_likelihood_LB2, dim=-1)

        # print(posterior_LB)
        posterior_B2 = posterior_LB2[seq_lens - 1, torch.arange(real_batch_size,device=self.config.device)]
        print("TEMP")
        posterior_B2 = torch.stack((torch.ones(B), torch.zeros(B)), dim=-1)
        p_B = (posterior_B2 * 
               torch.stack((p1[seq_lens - 1, torch.arange(real_batch_size,device=self.config.device)], p2[seq_lens - 1, torch.arange(real_batch_size,device=self.config.device)]), dim=-1)).sum(dim=-1)
        return {
            'retentions': p_B,
            # 'stabilities': w1w*s1 + w2w*s2,
            # 'difficulties': w1w*d1 + w2w*d2,
            # 'posterior': post,
            'penalty': torch.sum(torch.square(self.ws - self.initial_w)/torch.square(self.param_stddev))*real_batch_size*self.gamma
        }

    def _evaluate_model(self, w, sequences, delta_ts, seq_lens, real_batch_size):
        outputs,_ = self.forward(w, sequences)
        s_LB = outputs[:, :, 0]
        d_LB = outputs[:, :, 1]
        # s,d = outputs[seq_lens-1,torch.arange(real_batch_size,device=self.config.device)].transpose(0,1)
        p_LB = self.forgetting_curve(delta_ts,s_LB,-w[20])
        return p_LB,s_LB,d_LB

    def forward(
        self, w, inputs: Tensor, state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        :param inputs: shape[seq_len, batch_size, 2]
        """
        if state is None:
            state = torch.zeros((inputs.shape[1], 2))
        outputs = []
        for X in inputs:
            state = self.step(w, X, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def state_dict(self):
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["ws"].data,
            )
        )
    
    def initialize_parameters(self, train_set: pd.DataFrame) -> None:
        self.init1(train_set)
        self.init2(train_set)

    def init1(self, train_set) -> None:
        S0_dataset_group = (
            train_set[train_set["i"] == 2]
            .groupby(by=["first_rating", "delta_t"], group_keys=False)
            .agg({"y": ["mean", "count"]})
            .reset_index()
        )
        rating_stability = {}
        rating_count = {}
        average_recall = train_set["y"].mean()
        r_s0_default = {str(i): self.init_w[i - 1].numpy() for i in range(1, 5)}

        for first_rating in ("1", "2", "3", "4"):
            group = S0_dataset_group[S0_dataset_group["first_rating"] == first_rating]
            if group.empty:
                continue
            delta_t = group["delta_t"]
            if self.config.use_secs_intervals:
                recall = group["y"]["mean"]
            else:
                recall = (
                    group["y"]["mean"] * group["y"]["count"] + average_recall * 1
                ) / (group["y"]["count"] + 1)
            count = group["y"]["count"]

            init_s0 = r_s0_default[first_rating]

            def loss(stability):
                y_pred = self.forgetting_curve(delta_t, stability, -self.init_w[20].item())
                logloss = sum(
                    -(recall * np.log(y_pred) + (1 - recall) * np.log(1 - y_pred))
                    * count
                )
                l1 = (
                    np.abs(stability - init_s0) / 16
                    if not self.config.use_secs_intervals
                    else 0
                )
                return logloss + l1

            res = minimize(
                loss,
                x0=init_s0,
                bounds=((self.config.s_min, self.config.init_s_max),),
                options={"maxiter": int(sum(count))},
            )
            params = res.x
            stability = params[0]
            rating_stability[int(first_rating)] = stability
            rating_count[int(first_rating)] = sum(count)

        for small_rating, big_rating in (
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 3),
            (2, 4),
            (1, 4),
        ):
            if small_rating in rating_stability and big_rating in rating_stability:
                # if rating_count[small_rating] > 300 and rating_count[big_rating] > 300:
                #     continue
                if rating_stability[small_rating] > rating_stability[big_rating]:
                    if rating_count[small_rating] > rating_count[big_rating]:
                        rating_stability[big_rating] = rating_stability[small_rating]
                    else:
                        rating_stability[small_rating] = rating_stability[big_rating]

        w1 = 0.41
        w2 = 0.54

        if len(rating_stability) == 0:
            initial_stabilities = list(r_s0_default.values())
        elif len(rating_stability) == 1:
            rating = list(rating_stability.keys())[0]
            factor = rating_stability[rating] / r_s0_default[str(rating)]
            initial_stabilities = list(map(lambda x: x * factor, r_s0_default.values()))
        elif len(rating_stability) == 2:
            if 1 not in rating_stability and 2 not in rating_stability:
                rating_stability[2] = np.power(
                    rating_stability[3], 1 / (1 - w2)
                ) * np.power(rating_stability[4], 1 - 1 / (1 - w2))
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 1 not in rating_stability and 3 not in rating_stability:
                rating_stability[3] = np.power(rating_stability[2], 1 - w2) * np.power(
                    rating_stability[4], w2
                )
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 1 not in rating_stability and 4 not in rating_stability:
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 2 not in rating_stability and 3 not in rating_stability:
                rating_stability[2] = np.power(
                    rating_stability[1], w1 / (w1 + w2 - w1 * w2)
                ) * np.power(rating_stability[4], 1 - w1 / (w1 + w2 - w1 * w2))
                rating_stability[3] = np.power(
                    rating_stability[1], 1 - w2 / (w1 + w2 - w1 * w2)
                ) * np.power(rating_stability[4], w2 / (w1 + w2 - w1 * w2))
            elif 2 not in rating_stability and 4 not in rating_stability:
                rating_stability[2] = np.power(rating_stability[1], w1) * np.power(
                    rating_stability[3], 1 - w1
                )
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            elif 3 not in rating_stability and 4 not in rating_stability:
                rating_stability[3] = np.power(
                    rating_stability[1], 1 - 1 / (1 - w1)
                ) * np.power(rating_stability[2], 1 / (1 - w1))
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            initial_stabilities = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        elif len(rating_stability) == 3:
            if 1 not in rating_stability:
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 2 not in rating_stability:
                rating_stability[2] = np.power(rating_stability[1], w1) * np.power(
                    rating_stability[3], 1 - w1
                )
            elif 3 not in rating_stability:
                rating_stability[3] = np.power(rating_stability[2], 1 - w2) * np.power(
                    rating_stability[4], w2
                )
            elif 4 not in rating_stability:
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            initial_stabilities = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        elif len(rating_stability) == 4:
            initial_stabilities = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        self.ws.data[0:4] = Tensor(
            list(
                map(
                    lambda x: max(min(self.config.init_s_max, x), self.config.s_min),
                    initial_stabilities,
                )
            )
        )

        self.initial_w = self.ws.data.clone().to(self.config.device)
    
    def init2(self, train_set) -> None:
        S0_dataset_group = (
            train_set[train_set["i"] == 2]
            .groupby(by=["first_rating", "delta_t"], group_keys=False)
            .agg({"y": ["mean", "count"]})
            .reset_index()
        )
        rating_stability = {}
        rating_count = {}
        average_recall = train_set["y"].mean()
        r_s0_default = {str(i): self.init_w[i - 1].numpy() for i in range(1, 5)}

        for first_rating in ("1", "2", "3", "4"):
            group = S0_dataset_group[S0_dataset_group["first_rating"] == first_rating]
            if group.empty:
                continue
            delta_t = group["delta_t"]
            if self.config.use_secs_intervals:
                recall = group["y"]["mean"]
            else:
                recall = (
                    group["y"]["mean"] * group["y"]["count"] + average_recall * 1
                ) / (group["y"]["count"] + 1)
            count = group["y"]["count"]

            init_s0 = r_s0_default[first_rating]

            def loss(stability):
                y_pred = self.forgetting_curve(delta_t, stability, -self.init_w[41].item())
                logloss = sum(
                    -(recall * np.log(y_pred) + (1 - recall) * np.log(1 - y_pred))
                    * count
                )
                l1 = (
                    np.abs(stability - init_s0) / 16
                    if not self.config.use_secs_intervals
                    else 0
                )
                return logloss + l1

            res = minimize(
                loss,
                x0=init_s0,
                bounds=((self.config.s_min, self.config.init_s_max),),
                options={"maxiter": int(sum(count))},
            )
            params = res.x
            stability = params[0]
            rating_stability[int(first_rating)] = stability
            rating_count[int(first_rating)] = sum(count)

        for small_rating, big_rating in (
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 3),
            (2, 4),
            (1, 4),
        ):
            if small_rating in rating_stability and big_rating in rating_stability:
                # if rating_count[small_rating] > 300 and rating_count[big_rating] > 300:
                #     continue
                if rating_stability[small_rating] > rating_stability[big_rating]:
                    if rating_count[small_rating] > rating_count[big_rating]:
                        rating_stability[big_rating] = rating_stability[small_rating]
                    else:
                        rating_stability[small_rating] = rating_stability[big_rating]

        w1 = 0.41
        w2 = 0.54

        if len(rating_stability) == 0:
            initial_stabilities = list(r_s0_default.values())
        elif len(rating_stability) == 1:
            rating = list(rating_stability.keys())[0]
            factor = rating_stability[rating] / r_s0_default[str(rating)]
            initial_stabilities = list(map(lambda x: x * factor, r_s0_default.values()))
        elif len(rating_stability) == 2:
            if 1 not in rating_stability and 2 not in rating_stability:
                rating_stability[2] = np.power(
                    rating_stability[3], 1 / (1 - w2)
                ) * np.power(rating_stability[4], 1 - 1 / (1 - w2))
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 1 not in rating_stability and 3 not in rating_stability:
                rating_stability[3] = np.power(rating_stability[2], 1 - w2) * np.power(
                    rating_stability[4], w2
                )
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 1 not in rating_stability and 4 not in rating_stability:
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 2 not in rating_stability and 3 not in rating_stability:
                rating_stability[2] = np.power(
                    rating_stability[1], w1 / (w1 + w2 - w1 * w2)
                ) * np.power(rating_stability[4], 1 - w1 / (w1 + w2 - w1 * w2))
                rating_stability[3] = np.power(
                    rating_stability[1], 1 - w2 / (w1 + w2 - w1 * w2)
                ) * np.power(rating_stability[4], w2 / (w1 + w2 - w1 * w2))
            elif 2 not in rating_stability and 4 not in rating_stability:
                rating_stability[2] = np.power(rating_stability[1], w1) * np.power(
                    rating_stability[3], 1 - w1
                )
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            elif 3 not in rating_stability and 4 not in rating_stability:
                rating_stability[3] = np.power(
                    rating_stability[1], 1 - 1 / (1 - w1)
                ) * np.power(rating_stability[2], 1 / (1 - w1))
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            initial_stabilities = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        elif len(rating_stability) == 3:
            if 1 not in rating_stability:
                rating_stability[1] = np.power(rating_stability[2], 1 / w1) * np.power(
                    rating_stability[3], 1 - 1 / w1
                )
            elif 2 not in rating_stability:
                rating_stability[2] = np.power(rating_stability[1], w1) * np.power(
                    rating_stability[3], 1 - w1
                )
            elif 3 not in rating_stability:
                rating_stability[3] = np.power(rating_stability[2], 1 - w2) * np.power(
                    rating_stability[4], w2
                )
            elif 4 not in rating_stability:
                rating_stability[4] = np.power(
                    rating_stability[2], 1 - 1 / w2
                ) * np.power(rating_stability[3], 1 / w2)
            initial_stabilities = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        elif len(rating_stability) == 4:
            initial_stabilities = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        self.ws.data[21:25] = Tensor(
            list(
                map(
                    lambda x: max(min(self.config.init_s_max, x), self.config.s_min),
                    initial_stabilities,
                )
            )
        )

        self.initial_w = self.ws.data.clone().to(self.config.device)
