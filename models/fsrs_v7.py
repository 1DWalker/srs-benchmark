from typing import List, Union
import torch
from torch import nn, Tensor
from typing import Optional
from models.fsrs_v6 import FSRS6, FSRS6ParameterClipper
import torch.nn.functional as F
from torch.nn import Sigmoid
import pandas as pd
import numpy as np
import tqdm
import time
from scipy.optimize import minimize  # type: ignore

from config import Config


class FSRS7ParameterClipper(FSRS6ParameterClipper):
    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            # Initial S
            w[0] = w[0].clamp(self.config.s_min, self.config.init_s_max)
            w[1] = w[1].clamp(w[0], self.config.init_s_max)
            w[2] = w[2].clamp(w[1], self.config.init_s_max)
            w[3] = w[3].clamp(w[2], self.config.init_s_max)
            # Difficulty
            w[4] = w[4].clamp(1, 10)
            w[5] = w[5].clamp(0.001, 4)
            w[6] = w[6].clamp(0.1, 4)
            # Stability (long-term)
            w[7] = w[7].clamp(0, 3.5)
            w[8] = w[8].clamp(0, 1.5)
            w[9] = w[9].clamp(0.2, 4)
            w[10] = w[10].clamp(0.001, 1.5)
            w[11] = w[11].clamp(0.001, 1.5)
            w[12] = w[12].clamp(0.01, 1)
            w[13] = w[13].clamp(0, 3.5)
            w[14] = w[14].clamp(0, 1)
            w[15] = w[15].clamp(1, 8)
            # Stability (short-term)
            w[16] = w[16].clamp(0, 4.5)
            w[17] = w[17].clamp(0, 1.5)
            w[18] = w[18].clamp(0.2, 5.5)
            w[19] = w[19].clamp(0.001, 1.5)
            w[20] = w[20].clamp(0.001, 1.5)
            w[21] = w[21].clamp(0.001, 1)
            w[22] = w[22].clamp(0.2, 4)
            w[23] = w[23].clamp(0, 1)
            w[24] = w[24].clamp(1, 8)
            # Long-short term transition function
            w[25] = w[25].clamp(2.5, 10)
            w[26] = w[26].clamp(0, 1)
            # Forgetting curve
            w[27] = w[27].clamp(0.005, 0.1)  # delta_t<S part of the curve
            w[28] = w[28].clamp(0.1, 0.3)  # delta_t>S part of the curve
            # blend_start
            w[29] = w[29].clamp(0, 0.79)  # "ceiling" of blend_start
            w[30] = w[30].clamp(0, w[29])  # "floor" of blend_start
            w[31] = w[31].clamp(0.1, 5)  # slope for D
            w[32] = w[32].clamp(0, 10)  # intercept for D
            w[33] = w[33].clamp(0.01, 1)  # power for 1+S
            module.w.data = w


class FSRS7(FSRS6):
    n_epoch: int = 8
    batch_size: int = 1024
    lr: float = 2e-2
    betas: tuple = (0.8, 0.85)  # this is for Adam, default is (0.9, 0.999)

    # Old config
    # n_epoch: int = 5
    # batch_size: int = 512
    # lr: float = 4e-2

    # Multi-user optimization
    init_w = [0.0002, 1.0551, 2.5994, 12.6293,  # Initial S
              4.3654, 0.4526, 3.3309,  # Difficulty
              1.1151, 0.2171, 1.7041, 0.0623, 0.001, 0.985, 0.0, 0.6379, 1.2916,  # Stability (long-term)
              1.4172, 0.3211, 3.7526, 0.001, 0.2414, 0.001, 1.5227, 0.2113, 1.0,  # Stability (short-term)
              2.5, 0.998,  # Long-short term transition function
              0.0053, 0.1102, 0.7899, 0.4503, 5.0, 6.7423, 0.2353]

    # default_params_stddev_tensor = torch.tensor([9999., 9999., 9999., 9999.,  # Initial S
    #                                              9999., 9999., 9999.,  # Difficulty
    #                                              9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.,  # Stability (long-term)
    #                                              9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999., 9999.,  # Stability (short-term)
    #                                              9999., 9999.,  # Long-short term transition function
    #                                              9999., 9999.,  # Forgetting curve
    #                                              9999., 9999., 9999., 9999., 9999., 9999.])

    def __init__(self, config: Config, w: Optional[List[float]] = None):
        super().__init__(config)
        if w is None:
            w = self.init_w
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.init_w_tensor = self.w.data.clone().to(self.config.device)
        self.clipper = FSRS7ParameterClipper(config)

    def batch_process(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs, _ = self.forward(sequences)
        stabilities, difficulties = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=self.config.device),
        ].transpose(0, 1)

        blend_start = self._blend_start(difficulties, self.w[-5], self.w[-4], self.w[-3], self.w[-2])
        retentions = self.forgetting_curve(delta_ts, stabilities, blend_start, -self.w[-7], -self.w[-6], -self.w[-1])
        retentions = retentions.clamp(0.0001, 0.9999)
        output = {
            "retentions": retentions,
            "stabilities": stabilities,
            "difficulties": difficulties,
        }
        # output["penalty"] = (
        #     torch.sum(
        #         torch.square(self.w - self.init_w_tensor)
        #         / torch.square(self.default_params_stddev_tensor)
        #     )
        #     * real_batch_size
        #     * self.gamma
        # )
        return output

    def _blend_start(self, d: Union[float, Tensor],
                     blend_start_ceil=init_w[-5],
                     blend_start_floor=init_w[-4],
                     d_slope=init_w[-3],
                     d_intercept=init_w[-2]) -> Tensor:
        f_d = 2.718281828459045 ** (-d_slope * (d - d_intercept))
        output = (blend_start_ceil - blend_start_floor) / (1 + f_d)
        return output + blend_start_floor

    def forgetting_curve(self, t, s, blend_start, decay1=-init_w[-7], decay2=-init_w[-6], s_pow=-init_w[-1]):
        factor1 = 0.9 ** (1 / decay1) - 1
        factor2 = 0.9 ** (1 / decay2) - 1
        t_over_s = t / s
        R1 = (1 + factor1 * t_over_s) ** decay1
        R2 = (1 + factor2 * t_over_s) ** decay2
        s_factor = (1 + s) ** s_pow
        blending_function = (blend_start * s_factor) * 2.718281828459045 ** (-0.1 * t_over_s)
        return blending_function * R1 + (1 - blending_function) * R2

    def stability_after_success_long_term(self, state: Tensor, r: Tensor, rating: Tensor) -> Tensor:  # type: ignore[override]
        hard_penalty = torch.where(rating == 2, self.w[14], 1)
        easy_bonus = torch.where(rating == 4, self.w[15], 1)
        pls = self.stability_after_failure_long_term(state, r)
        SInc =  1 \
                + torch.exp(self.w[7]) \
                * (11 - state[:, 1]) \
                * torch.pow(state[:, 0], -self.w[8]) \
                * (torch.exp((1 - r) * self.w[9]) - 1) \
                * hard_penalty \
                * easy_bonus

        # Ensure that new S>=PLS
        new_s = state[:, 0] * SInc
        return torch.maximum(pls, new_s)

    def stability_after_success_short_term(self, state: Tensor, r: Tensor, rating: Tensor) -> Tensor:  # type: ignore[override]
        hard_penalty = torch.where(rating == 2, self.w[23], 1)
        easy_bonus = torch.where(rating == 4, self.w[24], 1)
        pls = self.stability_after_failure_short_term(state, r)
        # -1.5 so that w[17] doesn't have to be negative
        SInc =  1 \
                + torch.exp(self.w[16] - 1.5) \
                * (11 - state[:, 1]) \
                * torch.pow(state[:, 0], -self.w[17]) \
                * (torch.exp((1 - r) * self.w[18]) - 1) \
                * hard_penalty\
                * easy_bonus

        # Ensure that new S>=PLS
        new_s = state[:, 0] * SInc
        return torch.maximum(pls, new_s)

    def stability_after_failure_long_term(self, state: Tensor, r: Tensor) -> Tensor:  # type: ignore[override]
        old_s = state[:, 0]
        new_s = (
                self.w[10]
                * torch.pow(state[:, 1], -self.w[11])
                * (torch.pow(state[:, 0] + 1, self.w[12]) - 1)
                * torch.exp((1 - r) * self.w[13])
        )
        return torch.minimum(old_s, new_s)

    def stability_after_failure_short_term(self, state: Tensor, r: Tensor) -> Tensor:  # type: ignore[override]
        old_s = state[:, 0]
        new_s = (
                self.w[19]
                * torch.pow(state[:, 1], -self.w[20])
                * (torch.pow(state[:, 0] + 1, self.w[21]) - 1)
                * torch.exp((1 - r) * self.w[22])
        )
        return torch.minimum(old_s, new_s)

    def transition_function(self, delta_t: Tensor) -> Tensor:
        return 1 - self.w[26] * torch.exp(-self.w[25] * delta_t)

    def init_d(self, rating: Union[int, Tensor]) -> Tensor:
        new_d = self.w[4] - torch.exp(self.w[5] * (rating - 1)) + 1
        return new_d

    def linear_damping(self, delta_d: Tensor, old_d: Tensor) -> Tensor:
        return delta_d * (10 - old_d) / 9

    def mean_reversion(self, init: Tensor, current: Tensor) -> Tensor:
        return 0.01 * init + 0.99 * current

    def next_d(self, state: Tensor, rating: Tensor) -> Tensor:
        delta_d = -self.w[6] * (rating - 3)
        new_d = state[:, 1] + self.linear_damping(delta_d, state[:, 1])
        new_d = self.mean_reversion(self.init_d(4), new_d)
        return new_d

    def bin_interval(self, delta_t):
        """
        Bin intervals according to:
        - < 2 hours: 10-minute bins
        - 2-24 hours: 2-hour bins
        - > 24 hours: 1-day bins
        """
        # Convert to days if needed
        if isinstance(delta_t, pd.Series):
            intervals = delta_t.values
        else:
            intervals = np.array([delta_t]) if not isinstance(delta_t, np.ndarray) else delta_t

        # Define bin boundaries in days
        ten_minutes = 10 / (24 * 60)  # 0.006944...
        two_hours = 2 / 24  # 0.0833...
        one_day = 1.0

        binned = np.zeros_like(intervals)

        # < 2 hours: 10-minute bins
        mask_short = intervals < two_hours
        binned[mask_short] = np.maximum(
            np.floor(intervals[mask_short] / ten_minutes) * ten_minutes,
            ten_minutes  # Ensure minimum of 10 minutes
        )

        # 2-24 hours: 2-hour bins
        mask_medium = (intervals >= two_hours) & (intervals < one_day)
        binned[mask_medium] = np.maximum(
            np.floor(intervals[mask_medium] / two_hours) * two_hours,
            two_hours  # Ensure minimum of 2 hours
        )

        # > 24 hours: 1-day bins
        mask_long = intervals >= one_day
        binned[mask_long] = np.maximum(
            np.floor(intervals[mask_long]),
            one_day  # Ensure minimum of 1 day
        )

        return binned if len(binned) > 1 else binned[0]

    def pretrain(self, train_set: pd.DataFrame) -> None:
        # start = time.perf_counter()
        # Create binned intervals if using --secs
        # With FSRS-7 --secs should always be used
        if self.config.use_secs_intervals:
            train_set_copy = train_set.copy()
            train_set_copy['delta_t_binned'] = self.bin_interval(train_set_copy['delta_t'])
            group_by_cols = ["first_rating", "delta_t_binned"]
        else:
            train_set_copy = train_set
            group_by_cols = ["first_rating", "delta_t"]

        S0_dataset_group = (
            train_set_copy[train_set_copy["i"] == 2]
            .groupby(by=group_by_cols, group_keys=False)
            .agg({"y": ["mean", "count"]})
            .reset_index()
        )

        average_recall = train_set["y"].mean()
        r_s0_default = {str(i): self.init_w[i - 1] for i in range(1, 5)}

        def init_d_pretrain(rating: int, w_4: float, w_5: float):
            new_d = w_4 - np.exp(w_5 * (rating - 1)) + 1
            return new_d

        def evaluate_param_set(param_set):
            """Evaluate a parameter set and return total loss and rating stabilities"""
            decay1, decay2, blend_start_ceil, blend_start_floor, d_slope, d_intercept, s_pow, w_4, w_5 = \
                param_set
            current_rating_stability = {}
            current_rating_count = {}
            total_loss = 0

            # For each rating, optimize initial stability using current forgetting curve params
            for first_rating in ("1", "2", "3", "4"):
                group = S0_dataset_group[S0_dataset_group["first_rating"] == first_rating]
                if group.empty:
                    if self.config.verbose_inadequate_data:
                        tqdm.write(
                            f"Not enough data for first rating {first_rating}. Expected at least 1, got 0."
                        )
                    continue

                if self.config.use_secs_intervals:
                    delta_t = group["delta_t_binned"]
                else:
                    delta_t = group["delta_t"]
                recall = (group["y"]["mean"] * group["y"]["count"] + average_recall * 1) / (group["y"]["count"] + 1)
                count = group["y"]["count"]

                init_s0 = r_s0_default[first_rating]

                d_default = init_d_pretrain(int(first_rating), w_4, w_5)
                d_default = np.clip(d_default, 1, 10)

                def loss(stability):
                    assert first_rating in ["1", "2", "3", "4"]
                    blend_start = self._blend_start(d_default, blend_start_ceil, blend_start_floor, d_slope, d_intercept)
                    # Sanity checks
                    assert blend_start_ceil >= blend_start_floor
                    assert decay1 <= decay2
                    assert 0.01 <= s_pow <= 1.
                    assert (1 + stability) ** -s_pow < 1
                    y_pred = self.forgetting_curve(delta_t, stability, blend_start, -decay1, -decay2, -s_pow)
                    y_pred = np.clip(y_pred, 0.0001, 0.9999)
                    logloss = sum(
                        -(recall * np.log(y_pred) + (1 - recall) * np.log(1 - y_pred))
                        * count
                    )
                    l1 = (np.abs(stability - init_s0)) / 32
                    return logloss + l1

                res = minimize(
                    loss,
                    x0=init_s0,
                    bounds=((self.config.s_min, self.config.init_s_max),),
                    options={"maxiter": int(sum(count))},
                )

                stability = res.x[0]
                current_rating_stability[int(first_rating)] = stability
                current_rating_count[int(first_rating)] = sum(count)
                total_loss += res.fun

            # Apply stability ordering constraints
            for small_rating, big_rating in (
                    (1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4),
            ):
                if small_rating in current_rating_stability and big_rating in current_rating_stability:
                    if current_rating_stability[small_rating] > current_rating_stability[big_rating]:
                        if current_rating_count[small_rating] > current_rating_count[big_rating]:
                            current_rating_stability[big_rating] = current_rating_stability[small_rating]
                        else:
                            current_rating_stability[small_rating] = current_rating_stability[big_rating]

            return total_loss, current_rating_stability

        # Initial parameter sets (grid search)
        initial_forgetting_curve_params = [
            self.init_w[-7:] + self.init_w[4:6],
            # decay1, decay2, blend_start_ceil, blend_start_floor, d_slope, d_intercept, s_pow, w_4, w_5
            [0.005, 0.178, 0.7161, 0.3981, 3.52, 2.968, 0.5148, 4.1579, 0.4291],
            [0.005, 0.1185, 0.7547, 0.3769, 3.6524, 2.9115, 0.2838, 4.2913, 0.53],
            [0.005, 0.1372, 0.79, 0.4012, 3.5494, 3.0804, 0.1446, 4.611, 0.6235],
            [0.005, 0.1, 0.79, 0.4045, 3.7489, 2.8624, 0.2144, 4.7442, 0.4859],
            [0.005, 0.1233, 0.79, 0.3538, 3.7158, 2.6577, 0.2826, 4.6638, 0.4589],
            [0.005, 0.1846, 0.79, 0.5172, 3.496, 2.7762, 0.132, 5.0726, 0.5166],
            [0.005, 0.1184, 0.79, 0.5439, 3.7936, 2.8059, 0.2543, 4.5834, 0.5916],
            [0.005, 0.1763, 0.7595, 0.3978, 3.6942, 2.9052, 0.1569, 4.2623, 0.5103],
            [0.005, 0.1839, 0.7561, 0.4849, 3.5836, 2.9189, 0.3721, 4.1967, 0.4184],
            [0.005, 0.1518, 0.79, 0.3663, 3.6678, 2.9241, 0.4017, 4.1075, 0.5569],
            [0.005, 0.1722, 0.79, 0.2255, 3.9109, 3.2016, 0.2979, 4.6527, 0.5054],
            [0.005, 0.1353, 0.79, 0.5503, 3.6498, 2.6347, 0.2889, 4.7851, 0.456],
            [0.005, 0.1072, 0.79, 0.4332, 3.4019, 2.7145, 0.2215, 4.4336, 0.4954],
            [0.005, 0.1217, 0.7342, 0.443, 3.4413, 2.9602, 0.4536, 4.1251, 0.4555]
        ]

        # Track all candidates with their losses
        candidates = []  # List of (loss, param_set, rating_stability)

        # Evaluate initial parameter sets
        for param_set in initial_forgetting_curve_params:
            total_loss, rating_stability = evaluate_param_set(param_set)
            candidates.append((total_loss, param_set.copy(), rating_stability.copy()))

        # Sort candidates by loss (best first)
        candidates.sort(key=lambda x: x[0])

        # Use the best combination found
        best_total_loss, best_forgetting_curve_params, best_rating_stability = candidates[0]

        rating_stability = best_rating_stability

        if self.config.verbose_inadequate_data:
            tqdm.write(f"Best forgetting curve params: {best_forgetting_curve_params}")
            tqdm.write(f"Best total loss: {best_total_loss}")

        w1 = 0.41
        w2 = 0.54

        if len(rating_stability) == 0:
            raise Exception("Not enough data for pretraining!")
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

        # Update initial stabilities (w[0:4])
        self.w.data[0:4] = Tensor(
            list(
                map(
                    lambda x: max(min(self.config.init_s_max, x), self.config.s_min),
                    initial_stabilities,
                )
            )
        )

        # Update forgetting curve parameters with the best found parameters
        if best_forgetting_curve_params is not None:
            # forgetting_curve
            self.w.data[-7:] = Tensor(best_forgetting_curve_params[:-2])
            # init_d
            self.w.data[4:6] = Tensor(best_forgetting_curve_params[-2:])

        self.init_w_tensor = self.w.data.clone().to(self.config.device)

        # end = time.perf_counter()
        # print(f'Pretrain took {end - start:.2f} seconds, {(end - start) * 1000:.0f} milliseconds')

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty, state[:,2] is success of the previous review
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            keys = torch.tensor([1, 2, 3, 4], device=self.config.device)
            keys = keys.view(1, -1).expand(X[:, 1].long().size(0), -1)
            index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            # first learn, init memory states
            new_s = torch.ones_like(state[:, 0], device=self.config.device)
            new_s[index[0]] = self.w[index[1]]
            new_d = self.init_d(X[:, 1])
            new_d = new_d.clamp(1, 10)
        else:
            success = X[:, 1] > 1
            # The initial steep drop-off is larger if the last rating was Again
            blend_start = self._blend_start(state[:, 1], self.w[-5], self.w[-4], self.w[-3], self.w[-2])
            r = self.forgetting_curve(X[:, 0], state[:, 0], blend_start, -self.w[-7], -self.w[-6], -self.w[-1])

            if not torch.isfinite(r).all():
                print('R contains NaN/Inf')
                print(f'r={r}\n')

            new_s_long_term = torch.where(
                    success,
                    self.stability_after_success_long_term(state, r, X[:, 1]),
                    self.stability_after_failure_long_term(state, r),
            )
            new_s_short_term = torch.where(
                    success,
                    self.stability_after_success_short_term(state, r, X[:, 1]),
                    self.stability_after_failure_short_term(state, r),
            )
            # A number between 0 and 1 that represents how much of a non-same-day review this is
            # 1 = long-term
            # 0 = short-term (same-day)
            coefficient = self.transition_function(X[:,0])
            new_s = coefficient * new_s_long_term + (1 - coefficient) * new_s_short_term

            if not torch.isfinite(new_s).all():
                print('S contains NaN/Inf')
                print(f's={state[:,0]}')
                print(f'new_s={new_s}\n')

            new_d = self.next_d(state, X[:, 1])
            new_d = new_d.clamp(1, 10)

            if not torch.isfinite(new_d).all():
                print('D contains NaN/Inf')
                print(f'd={state[:,1]}')
                print(f'new_d={new_d}\n')

        new_s = new_s.clamp(self.config.s_min, 36500)
        return torch.stack([new_s, new_d], dim=1)

    def get_optimizer(self, lr: float, wd: float, betas) -> torch.optim.Optimizer:
        return torch.optim.NAdam(self.parameters(), lr=lr, betas=betas)