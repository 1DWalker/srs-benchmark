import copy
from itertools import accumulate
import logging
import math
from multiprocessing import Process
import signal
import sys
import os
import optuna
import pandas as pd
import numpy as np
from typing import List, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import torch
import json
from torch import nn
from torch import Tensor
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from sklearn.metrics import roc_auc_score, root_mean_squared_error, log_loss  # type: ignore
from tqdm.auto import tqdm  # type: ignore
from scipy.optimize import minimize  # type: ignore
from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore
import warnings
from reptile_trainer import get_inner_opt, finetune
from script import cum_concat, remove_non_continuous_rows, remove_outliers, sort_jsonl
import pyarrow.parquet as pq  # type: ignore
from config import create_parser
from utils import catch_exceptions, rmse_matrix

parser = create_parser()
args = parser.parse_args()

DEV_MODE = args.dev
DRY_RUN = args.dry
MODEL_NAME = args.algo
SHORT_TERM = args.short
SECS_IVL = args.secs
NO_TEST_SAME_DAY = args.no_test_same_day
NO_TRAIN_SAME_DAY = args.no_train_same_day
EQUALIZE_TEST_WITH_NON_SECS = args.equalize_test_with_non_secs
TWO_BUTTONS = args.two_buttons
FILE = args.file
PLOT = args.plot
WEIGHTS = args.weights
PARTITIONS = args.partitions
RAW = args.raw
PROCESSES = args.processes
DATA_PATH = Path(args.data)
RECENCY = args.recency
TRAIN_EQUALS_TEST = args.train_equals_test
torch.set_num_threads(2)
# torch.set_num_interop_threads(2)

model_list = (
    "FSRSv1",
    "FSRSv2",
    "FSRSv3",
    "FSRSv4",
    "FSRS-4.5",
    "FSRS-5",
    "FSRS-6",
    "Ebisu-v2",
    "SM2",
    "HLR",
    "GRU",
    "GRU-P",
    "LSTM",
    "RNN",
    "AVG",
    "RMSE-BINS-EXPLOIT",
    "90%",
    "DASH",
    "DASH[MCM]",
    "DASH[ACT-R]",
    "ACT-R",
    "NN-17",
    "Transformer",
    "SM2-trainable",
    "Anki",
)

if MODEL_NAME not in model_list:
    raise ValueError(f"Model name must be one of {model_list}")

if DEV_MODE:
    sys.path.insert(0, os.path.abspath("../fsrs-optimizer/src/fsrs_optimizer/"))

from fsrs_optimizer import BatchDataset, BatchLoader, plot_brier  # type: ignore

if MODEL_NAME.startswith("Ebisu"):
    import ebisu  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(42)
tqdm.pandas()

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    and MODEL_NAME in ["GRU", "GRU-P", "LSTM", "RNN", "NN-17", "Transformer"]
    else "cpu"
)
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

n_splits: int = 5
batch_size: int = 512
max_seq_len: int = 64
verbose: bool = False
verbose_inadequate_data: bool = False

FILE_NAME = (
    MODEL_NAME
    + ("-dry-run" if DRY_RUN else "")
    + ("-short" if SHORT_TERM else "")
    + ("-secs" if SECS_IVL else "")
    + ("-recency" if RECENCY else "")
    + ("-no_test_same_day" if NO_TEST_SAME_DAY else "")
    + ("-no_train_same_day" if NO_TRAIN_SAME_DAY else "")
    + ("-equalize_test_with_non_secs" if EQUALIZE_TEST_WITH_NON_SECS else "")
    + ("-train_equals_test" if TRAIN_EQUALS_TEST else "")
    + ("-" + PARTITIONS if PARTITIONS != "none" else "")
    + ("-dev" if DEV_MODE else "")
)
OPT_NAME = FILE_NAME + "_opt"

S_MIN = 0.001
INIT_S_MAX = 100
S_MAX = 36500


class FSRS(nn.Module):
    def __init__(self):
        super(FSRS, self).__init__()

    def forgetting_curve(self, t, s):
        raise NotImplementedError("Forgetting curve not implemented")

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs, _ = self.forward(sequences)
        stabilities, difficulties, *_ = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=DEVICE),
        ].transpose(0, 1)
        retentions = self.forgetting_curve(delta_ts, stabilities)
        return {
            "retentions": retentions,
            "stabilities": stabilities,
            "difficulties": difficulties,
        }

    def pretrain(self, train_set):
        S0_dataset_group = (
            train_set[train_set["i"] == 2]
            .groupby(by=["first_rating", "delta_t"], group_keys=False)
            .agg({"y": ["mean", "count"]})
            .reset_index()
        )
        rating_stability = {}
        rating_count = {}
        average_recall = train_set["y"].mean()
        r_s0_default = {str(i): self.init_w[i - 1] for i in range(1, 5)}

        for first_rating in ("1", "2", "3", "4"):
            group = S0_dataset_group[S0_dataset_group["first_rating"] == first_rating]
            if group.empty:
                if verbose:
                    tqdm.write(
                        f"Not enough data for first rating {first_rating}. Expected at least 1, got 0."
                    )
                continue
            delta_t = group["delta_t"]
            if SECS_IVL:
                recall = group["y"]["mean"]
            else:
                recall = (
                    group["y"]["mean"] * group["y"]["count"] + average_recall * 1
                ) / (group["y"]["count"] + 1)
            count = group["y"]["count"]

            init_s0 = r_s0_default[first_rating]

            def loss(stability):
                y_pred = self.forgetting_curve(delta_t, stability)
                logloss = sum(
                    -(recall * np.log(y_pred) + (1 - recall) * np.log(1 - y_pred))
                    * count
                )
                l1 = np.abs(stability - init_s0) / 16 if not SECS_IVL else 0
                return logloss + l1

            res = minimize(
                loss,
                x0=init_s0,
                bounds=((S_MIN, INIT_S_MAX),),
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
            raise Exception("Not enough data for pretraining!")
        elif len(rating_stability) == 1:
            rating = list(rating_stability.keys())[0]
            factor = rating_stability[rating] / r_s0_default[str(rating)]
            init_s0 = list(map(lambda x: x * factor, r_s0_default.values()))
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
            init_s0 = [
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
            init_s0 = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        elif len(rating_stability) == 4:
            init_s0 = [
                item[1] for item in sorted(rating_stability.items(), key=lambda x: x[0])
            ]
        stabilities = list(map(lambda x: max(min(INIT_S_MAX, x), S_MIN), init_s0))
        for i in range(4):
            self.w_params[i].data = torch.tensor(
                stabilities[i], dtype=torch.float, device=DEVICE
            )
        self.init_w_tensor = torch.stack(list(self.w_params)).clone().to(DEVICE)


class FSRS6ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w_params"):
            with torch.no_grad():
                w = module.w_params
                w[0].data = w[0].clamp(S_MIN, 100)
                w[1].data = w[1].clamp(S_MIN, 100)
                w[2].data = w[2].clamp(S_MIN, 100)
                w[3].data = w[3].clamp(S_MIN, 100)
                w[4].data = w[4].clamp(1, 10)
                w[5].data = w[5].clamp(0.001, 4)
                w[6].data = w[6].clamp(0.001, 4)
                w[7].data = w[7].clamp(0.001, 0.75)
                w[8].data = w[8].clamp(0, 4.5)
                w[9].data = w[9].clamp(0, 0.8)
                w[10].data = w[10].clamp(0.001, 3.5)
                w[11].data = w[11].clamp(0.001, 5)
                w[12].data = w[12].clamp(0.001, 0.25)
                w[13].data = w[13].clamp(0.001, 0.9)
                w[14].data = w[14].clamp(0, 4)
                w[15].data = w[15].clamp(0, 1)
                w[16].data = w[16].clamp(1, 6)
                w[17].data = w[17].clamp(0, 2)
                w[18].data = w[18].clamp(0, 2)
                w[19].data = w[19].clamp(0, 0.8)
                w[20].data = w[20].clamp(0.1, 0.8)


class FSRS6(FSRS):
    init_w = [
        0.2172,
        1.1771,
        3.2602,
        16.1507,
        7.0114,
        0.57,
        2.0966,
        0.0069,
        1.5261,
        0.112,
        1.0178,
        1.849,
        0.1133,
        0.3127,
        2.2934,
        0.2191,
        3.0004,
        0.7536,
        0.3332,
        0.1437,
        0.2,
    ]
    clipper = FSRS6ParameterClipper()
    lr: float = 4e-2
    gamma: float = 1
    wd: float = 1e-5
    n_epoch: int = 5
    default_params_stddev_tensor = torch.tensor(
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

    def __init__(self, w: List[float] = init_w):
        super(FSRS6, self).__init__()
        self.w_params = torch.nn.ParameterList(
            [
                torch.tensor(w[i], dtype=torch.float, requires_grad=True)
                for i in range(len(w))
            ]
        )
        self.init_w_tensor = torch.stack(list(self.w_params)).clone().to(DEVICE)

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        self.w = torch.stack(list(self.w_params))
        outputs, _ = self.forward(sequences)
        stabilities, difficulties = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=DEVICE),
        ].transpose(0, 1)
        retentions = self.forgetting_curve(delta_ts, stabilities, -self.w[20])
        output = {
            "retentions": retentions,
            "stabilities": stabilities,
            "difficulties": difficulties,
        }
        output["penalty"] = (
            torch.sum(
                torch.square(self.w - self.init_w_tensor)
                / torch.square(self.default_params_stddev_tensor)
            )
            * real_batch_size
            * self.gamma
        )
        return output

    def forgetting_curve(self, t, s, decay=-0.2):
        factor = 0.9 ** (1 / decay) - 1
        return (1 + factor * t / s) ** decay

    def stability_after_success(
        self, state: Tensor, r: Tensor, rating: Tensor
    ) -> Tensor:
        hard_penalty = torch.where(rating == 2, self.w[15], 1)
        easy_bonus = torch.where(rating == 4, self.w[16], 1)
        new_s = state[:, 0] * (
            1
            + torch.exp(self.w[8])
            * (11 - state[:, 1])
            * torch.pow(state[:, 0], -self.w[9])
            * (torch.exp((1 - r) * self.w[10]) - 1)
            * hard_penalty
            * easy_bonus
        )
        return new_s

    def stability_after_failure(self, state: Tensor, r: Tensor) -> Tensor:
        old_s = state[:, 0]
        new_s = (
            self.w[11]
            * torch.pow(state[:, 1], -self.w[12])
            * (torch.pow(old_s + 1, self.w[13]) - 1)
            * torch.exp((1 - r) * self.w[14])
        )
        new_minimum_s = old_s / torch.exp(self.w[17] * self.w[18])
        return torch.minimum(new_s, new_minimum_s)

    def stability_short_term(self, state: Tensor, rating: Tensor) -> Tensor:
        sinc = torch.exp(self.w[17] * (rating - 3 + self.w[18])) * torch.pow(
            state[:, 0], -self.w[19]
        )
        new_s = state[:, 0] * torch.where(rating >= 3, sinc.clamp(min=1), sinc)
        return new_s

    def init_d(self, rating: Union[int, Tensor]) -> Tensor:
        new_d = self.w[4] - torch.exp(self.w[5] * (rating - 1)) + 1
        return new_d

    def linear_damping(self, delta_d: Tensor, old_d: Tensor) -> Tensor:
        return delta_d * (10 - old_d) / 9

    def next_d(self, state: Tensor, rating: Tensor) -> Tensor:
        delta_d = -self.w[6] * (rating - 3)
        new_d = state[:, 1] + self.linear_damping(delta_d, state[:, 1])
        new_d = self.mean_reversion(self.init_d(4), new_d)
        return new_d

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            keys = torch.tensor([1, 2, 3, 4], device=DEVICE)
            keys = keys.view(1, -1).expand(X[:, 1].long().size(0), -1)
            index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            # first learn, init memory states
            new_s = torch.ones_like(state[:, 0], device=DEVICE)
            new_s[index[0]] = self.w[index[1]]
            new_d = self.init_d(X[:, 1])
            new_d = new_d.clamp(1, 10)
        else:
            r = self.forgetting_curve(X[:, 0], state[:, 0], -self.w[20])
            short_term = X[:, 0] < 1
            success = X[:, 1] > 1
            new_s = torch.where(
                short_term,
                self.stability_short_term(state, X[:, 1]),
                torch.where(
                    success,
                    self.stability_after_success(state, r, X[:, 1]),
                    self.stability_after_failure(state, r),
                ),
            )
            new_d = self.next_d(state, X[:, 1])
            new_d = new_d.clamp(1, 10)
        new_s = new_s.clamp(S_MIN, 36500)
        return torch.stack([new_s, new_d], dim=1)

    def forward(
        self, inputs: Tensor, state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        :param inputs: shape[seq_len, batch_size, 2]
        """
        if state is None:
            state = torch.zeros((inputs.shape[1], 2))
        outputs = []
        for X in inputs:
            state = self.step(X, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def mean_reversion(self, init: Tensor, current: Tensor) -> Tensor:
        return self.w[7] * init + (1 - self.w[7]) * current


def iter(model, batch):
    sequences, delta_ts, labels, seq_lens, weights = batch
    real_batch_size = seq_lens.shape[0]
    result = {"labels": labels, "weights": weights}
    outputs = model.iter(sequences, delta_ts, seq_lens, real_batch_size)
    result.update(outputs)
    return result


class Trainer:
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        MODEL: nn.Module,
        train_set: pd.DataFrame,
        test_set: Optional[pd.DataFrame],
        hyperparams,
        n_epoch: int = 1,
        lr: float = 1e-2,
        wd: float = 1e-4,
        batch_size: int = 256,
        max_seq_len: int = 64,
    ) -> None:
        self.model = MODEL.to(device=DEVICE)
        if isinstance(MODEL, (FSRS6)):
            self.model.pretrain(train_set)  # type: ignore
        groups = []
        for i, (_, param) in enumerate(self.model.named_parameters()):
            groups.append(
                {
                    "params": param,
                    "weight_decay": 0.0,
                    "lr": hyperparams["lrs"][i],
                    "betas": (hyperparams["beta1s"][i], hyperparams["beta2s"][i]),
                    "eps": hyperparams["epsilons"][i],
                }
            )
        self.optimizer = torch.optim.Adam(groups)

        self.clipper = MODEL.clipper if hasattr(MODEL, "clipper") else None
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.build_dataset(train_set, test_set)
        self.n_epoch = n_epoch
        self.batch_nums = self.train_data_loader.batch_nums
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.batch_nums * n_epoch
        )
        self.avg_train_losses: list[float] = []
        self.avg_eval_losses: list[float] = []
        self.loss_fn = nn.BCELoss(reduction="none")

    def build_dataset(self, train_set: pd.DataFrame, test_set: Optional[pd.DataFrame]):
        self.train_set = BatchDataset(
            train_set.copy(),
            self.batch_size,
            max_seq_len=self.max_seq_len,
            device=DEVICE,
        )
        self.train_data_loader = BatchLoader(self.train_set)

        self.test_set = (
            []
            if test_set is None
            else BatchDataset(
                test_set.copy(),
                batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
                device=DEVICE,
            )
        )
        self.test_data_loader = (
            [] if test_set is None else BatchLoader(self.test_set, shuffle=False)
        )

    def train(self):
        best_loss = np.inf
        epoch_len = len(self.train_set.y_train)
        for k in range(self.n_epoch):
            weighted_loss, w = self.eval()
            if weighted_loss < best_loss:
                best_loss = weighted_loss
                best_w = w
            for i, batch in enumerate(self.train_data_loader):
                self.model.train()
                self.optimizer.zero_grad()
                result = iter(self.model, batch)
                loss = (
                    self.loss_fn(result["retentions"], result["labels"])
                    * result["weights"]
                ).sum()
                if "penalty" in result:
                    loss += result["penalty"] / epoch_len
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.clipper:
                    self.model.apply(self.clipper)
        weighted_loss, w = self.eval()
        if weighted_loss < best_loss:
            best_loss = weighted_loss
            best_w = w

        return best_w

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            losses = []
            self.train_data_loader.shuffle = False
            for data_loader in (self.train_data_loader, self.test_data_loader):
                if len(data_loader) == 0:
                    losses.append(0)
                    continue
                loss = 0
                total = 0
                epoch_len = len(data_loader.dataset.y_train)
                for batch in data_loader:
                    result = iter(self.model, batch)
                    loss += (
                        (
                            self.loss_fn(result["retentions"], result["labels"])
                            * result["weights"]
                        )
                        .sum()
                        .detach()
                        .item()
                    )
                    if "penalty" in result:
                        loss += (result["penalty"] / epoch_len).detach().item()
                    total += batch[3].shape[0]
                losses.append(loss / total)
            self.train_data_loader.shuffle = True
            self.avg_train_losses.append(losses[0])
            self.avg_eval_losses.append(losses[1])

            w = copy.deepcopy(self.model.state_dict())
            weighted_loss = (
                losses[0] * len(self.train_set) + losses[1] * len(self.test_set)
            ) / (len(self.train_set) + len(self.test_set))

            return weighted_loss, w

    def plot(self):
        fig = plt.figure()
        ax = fig.gca()
        self.avg_train_losses = [x.item() for x in self.avg_train_losses]
        self.avg_eval_losses = [x.item() for x in self.avg_eval_losses]
        ax.plot(self.avg_train_losses, label="train")
        ax.plot(self.avg_eval_losses, label="test")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        return fig


class Collection:
    def __init__(self, MODEL) -> None:
        self.model = MODEL.to(device=DEVICE)
        self.model.eval()

    def batch_predict(self, dataset):
        batch_dataset = BatchDataset(
            dataset, batch_size=8192, sort_by_length=False, device=DEVICE
        )
        batch_loader = BatchLoader(batch_dataset, shuffle=False)
        retentions = []
        stabilities = []
        difficulties = []
        with torch.no_grad():
            for batch in batch_loader:
                result = iter(self.model, batch)
                retentions.extend(result["retentions"].cpu().tolist())
                if "stabilities" in result:
                    stabilities.extend(result["stabilities"].cpu().tolist())
                if "difficulties" in result:
                    difficulties.extend(result["difficulties"].cpu().tolist())

        return retentions, stabilities, difficulties


def create_features_helper(df, model_name, secs_ivl=SECS_IVL):
    df["review_th"] = range(1, df.shape[0] + 1)
    df.sort_values(by=["card_id", "review_th"], inplace=True)
    df.drop(df[~df["rating"].isin([1, 2, 3, 4])].index, inplace=True)

    if TWO_BUTTONS:
        df["rating"] = df["rating"].replace({2: 3, 4: 3})
    df["i"] = df.groupby("card_id").cumcount() + 1
    df.drop(df[df["i"] > max_seq_len * 2].index, inplace=True)
    if (
        "delta_t" not in df.columns
        and "elapsed_days" in df.columns
        and "elapsed_seconds" in df.columns
    ):
        df["delta_t"] = df["elapsed_days"]
        if secs_ivl:
            df["delta_t_secs"] = df["elapsed_seconds"] / 86400
            df["delta_t_secs"] = df["delta_t_secs"].map(lambda x: max(0, x))
    global SHORT_TERM
    if model_name.startswith("FSRS-5") or model_name.startswith("FSRS-6"):
        SHORT_TERM = True
    if not SHORT_TERM:
        # exclude reviews that are on the same day from features and labels
        df.drop(df[df["elapsed_days"] == 0].index, inplace=True)
        df["i"] = df.groupby("card_id").cumcount() + 1
    df["delta_t"] = df["delta_t"].map(lambda x: max(0, x))
    t_history_non_secs_list = df.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    if secs_ivl:
        t_history_secs_list = df.groupby("card_id", group_keys=False)[
            "delta_t_secs"
        ].apply(lambda x: cum_concat([[i] for i in x]))
    r_history_list = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: cum_concat([[i] for i in x])
    )
    last_rating = []
    for t_sublist, r_sublist in zip(t_history_non_secs_list, r_history_list):
        for t_history, r_history in zip(t_sublist, r_sublist):
            flag = True
            for t, r in zip(reversed(t_history[:-1]), reversed(r_history[:-1])):
                if t > 0:
                    last_rating.append(r)
                    flag = False
                    break
            if flag:
                last_rating.append(r_history[0])
    df["last_rating"] = last_rating
    df["r_history"] = [
        ",".join(map(str, item[:-1])) for sublist in r_history_list for item in sublist
    ]
    df["t_history"] = [
        ",".join(map(str, item[:-1]))
        for sublist in t_history_non_secs_list
        for item in sublist
    ]
    if secs_ivl:
        if EQUALIZE_TEST_WITH_NON_SECS:
            df["t_history"] = [
                ",".join(map(str, item[:-1]))
                for sublist in t_history_non_secs_list
                for item in sublist
            ]
            df["t_history_secs"] = [
                ",".join(map(str, item[:-1]))
                for sublist in t_history_secs_list
                for item in sublist
            ]
        else:
            df["t_history"] = [
                ",".join(map(str, item[:-1]))
                for sublist in t_history_secs_list
                for item in sublist
            ]
        df["delta_t"] = df["delta_t_secs"]
        t_history_used = t_history_secs_list
    else:
        t_history_used = t_history_non_secs_list

    if model_name.startswith("FSRS") or model_name in (
        "RNN",
        "GRU",
        "Transformer",
        "SM2-trainable",
        "Anki",
        "90%",
    ):
        df["tensor"] = [
            torch.tensor((t_item[:-1], r_item[:-1]), dtype=torch.float32).transpose(
                0, 1
            )
            for t_sublist, r_sublist in zip(t_history_used, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    elif model_name in "LSTM":
        # Create features (currently unused):
        # # - number of unique cards in the revlog
        # # - the number of new cards that were introduced today so far
        # # - the number of reviews that were done today so far
        # # - the number of new cards that were introduced since the last review of this card
        # # - the number of reviews that were done since the last review of this card
        df["is_new_card"] = (~df["card_id"].duplicated()).astype(int)
        df["cum_new_cards"] = df["is_new_card"].cumsum()
        df["diff_new_cards"] = df.groupby("card_id")["cum_new_cards"].diff().fillna(0)
        df["diff_reviews"] = np.maximum(
            0, -1 + df.groupby("card_id")["review_th"].diff().fillna(0)
        )
        df["cum_new_cards_today"] = df.groupby("day_offset")["is_new_card"].cumsum()
        df["cum_reviews_today"] = df.groupby("day_offset").cumcount()
        df["delta_t_days"] = df["elapsed_days"].map(lambda x: max(0, x))

        if secs_ivl:
            # Use days for the forgetting curve
            # This also indirectly causes --no_train_on_same_day and --no_test_on_same_day.
            df["delta_t"] = df["delta_t_days"]

        features = ["delta_t_secs" if secs_ivl else "delta_t", "duration", "rating"]

        def get_history(group):
            rows = group.apply(
                lambda row: torch.tensor(
                    [row[feature] for feature in features],
                    dtype=torch.float32,
                    requires_grad=False,
                ),
                axis=1,
            ).tolist()

            cum_rows = list(
                accumulate(
                    rows,
                    lambda x, y: torch.cat((x, y.unsqueeze(0))),
                    initial=torch.empty(
                        (0, len(features)), dtype=torch.float32, requires_grad=False
                    ),
                )
            )[:-1]
            return pd.Series(cum_rows, index=group.index)

        grouped = df.groupby("card_id", group_keys=False)
        df["tensor"] = grouped[df.columns.difference(["card_id"])].apply(get_history)
    elif model_name in "GRU-P":
        df["tensor"] = [
            torch.tensor((t_item[1:], r_item[:-1]), dtype=torch.float32).transpose(0, 1)
            for t_sublist, r_sublist in zip(t_history_used, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    elif model_name == "HLR":
        df["tensor"] = [
            torch.tensor(
                [
                    np.sqrt(
                        r_item[:-1].count(2)
                        + r_item[:-1].count(3)
                        + r_item[:-1].count(4)
                    ),
                    np.sqrt(r_item[:-1].count(1)),
                ],
                dtype=torch.float32,
            )
            for r_sublist in r_history_list
            for r_item in r_sublist
        ]
    elif model_name == "ACT-R":
        df["tensor"] = [
            (torch.cumsum(torch.tensor([t_item]), dim=1)).transpose(0, 1)
            for t_sublist in t_history_used
            for t_item in t_sublist
        ]
    elif model_name in ("DASH", "DASH[MCM]"):

        def dash_tw_features(r_history, t_history, enable_decay=False):
            features = np.zeros(8)
            r_history = np.array(r_history) > 1
            tau_w = np.array([0.2434, 1.9739, 16.0090, 129.8426])
            time_windows = np.array([1, 7, 30, np.inf])

            # Compute the cumulative sum of t_history in reverse order
            cumulative_times = np.cumsum(t_history[::-1])[::-1]

            for j, time_window in enumerate(time_windows):
                # Calculate decay factors for each time window
                if enable_decay:
                    decay_factors = np.exp(-cumulative_times / tau_w[j])
                else:
                    decay_factors = np.ones_like(cumulative_times)

                # Identify the indices where cumulative times are within the current time window
                valid_indices = cumulative_times <= time_window

                # Update features using decay factors where valid
                features[j * 2] += np.sum(decay_factors[valid_indices])
                features[j * 2 + 1] += np.sum(
                    r_history[valid_indices] * decay_factors[valid_indices]
                )

            return features

        df["tensor"] = [
            torch.tensor(
                dash_tw_features(r_item[:-1], t_item[1:], "MCM" in model_name),
                dtype=torch.float32,
            )
            for t_sublist, r_sublist in zip(t_history_used, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    elif model_name == "DASH[ACT-R]":

        def dash_actr_features(r_history, t_history):
            r_history = torch.tensor(np.array(r_history) > 1, dtype=torch.float32)
            sp_history = torch.tensor(t_history, dtype=torch.float32)
            cumsum = torch.cumsum(sp_history, dim=0)
            features = [r_history, sp_history - cumsum + cumsum[-1:None]]
            return torch.stack(features, dim=1)

        df["tensor"] = [
            torch.tensor(
                dash_actr_features(r_item[:-1], t_item[1:]),
                dtype=torch.float32,
            )
            for t_sublist, r_sublist in zip(t_history_used, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    elif model_name == "NN-17":

        def r_history_to_l_history(r_history):
            l_history = [0 for _ in range(len(r_history) + 1)]
            for i, r in enumerate(r_history):
                l_history[i + 1] = l_history[i] + (r == 1)
            return l_history[:-1]

        df["tensor"] = [
            torch.tensor(
                (t_item[:-1], r_item[:-1], r_history_to_l_history(r_item[:-1]))
            ).transpose(0, 1)
            for t_sublist, r_sublist in zip(t_history_used, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
    elif model_name == "SM2":
        df["sequence"] = df["r_history"]
    elif model_name.startswith("Ebisu"):
        df["sequence"] = [
            tuple(zip(t_item[:-1], r_item[:-1]))
            for t_sublist, r_sublist in zip(t_history_used, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]

    df["first_rating"] = df["r_history"].map(lambda x: x[0] if len(x) > 0 else "")
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    if SHORT_TERM:
        df = df[(df["delta_t"] != 0) | (df["i"] == 1)].copy()
    df["i"] = (
        df.groupby("card_id")
        .apply(lambda x: (x["elapsed_days"] > 0).cumsum())
        .reset_index(level=0, drop=True)
        + 1
    )
    if not secs_ivl:
        filtered_dataset = (
            df[df["i"] == 2]
            .groupby(by=["first_rating"], as_index=False, group_keys=False)[df.columns]
            .apply(remove_outliers)
        )
        if filtered_dataset.empty:
            return pd.DataFrame()
        df[df["i"] == 2] = filtered_dataset
        df.dropna(inplace=True)
        df = df.groupby("card_id", as_index=False, group_keys=False)[df.columns].apply(
            remove_non_continuous_rows
        )
    return df[df["delta_t"] > 0].sort_values(by=["review_th"])


def create_features(df, model_name="FSRSv3", secs_ivl=SECS_IVL):
    if secs_ivl and EQUALIZE_TEST_WITH_NON_SECS:
        df_non_secs = create_features_helper(df.copy(), model_name, False)
        df_secs = create_features_helper(df.copy(), model_name, True)
        df_intersect = df_secs[df_secs["review_th"].isin(df_non_secs["review_th"])]
        # rmse_bins requires that delta_t, i, r_history, t_history remains the same as with non secs
        assert len(df_intersect) == len(df_non_secs)
        assert np.equal(df_intersect["i"], df_non_secs["i"]).all()
        assert np.equal(df_intersect["t_history"], df_non_secs["t_history"]).all()
        assert np.equal(df_intersect["r_history"], df_non_secs["r_history"]).all()

        tscv = TimeSeriesSplit(n_splits=n_splits)
        for split_i, (_, non_secs_test_index) in enumerate(tscv.split(df_non_secs)):
            non_secs_test_set = df_non_secs.iloc[non_secs_test_index]
            # For the resulting train set, only allow reviews that are less than the smallest review_th in non_secs_test_set
            allowed_train = df_secs[
                df_secs["review_th"] < non_secs_test_set["review_th"].min()
            ]
            df_secs[f"{split_i}_train"] = df_secs["review_th"].isin(
                allowed_train["review_th"]
            )

            # For the resulting test set, only allow reviews that exist in non_secs_test_set
            df_secs[f"{split_i}_test"] = df_secs["review_th"].isin(
                non_secs_test_set["review_th"]
            )

        return df_secs
    else:
        return create_features_helper(df, model_name, secs_ivl)


def process(user_id, dataset, params):
    Model = FSRS6

    # dataset = create_features(df_revlogs, MODEL_NAME, SECS_IVL)
    if dataset.shape[0] < 6:
        raise Exception(f"{user_id} does not have enough data.")
    if PARTITIONS != "none":
        df_cards = pd.read_parquet(
            DATA_PATH / "cards", filters=[("user_id", "=", user_id)]
        )
        df_cards.drop(columns=["user_id"], inplace=True)
        df_decks = pd.read_parquet(
            DATA_PATH / "decks", filters=[("user_id", "=", user_id)]
        )
        df_decks.drop(columns=["user_id"], inplace=True)
        dataset = dataset.merge(df_cards, on="card_id", how="left").merge(
            df_decks, on="deck_id", how="left"
        )
        dataset.fillna(-1, inplace=True)
        if PARTITIONS == "preset":
            dataset["partition"] = dataset["preset_id"].astype(int)
        elif PARTITIONS == "deck":
            dataset["partition"] = dataset["deck_id"].astype(int)
    else:
        dataset["partition"] = 0
    w_list = []
    testsets = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for split_i, (train_index, test_index) in enumerate(tscv.split(dataset)):
        if not TRAIN_EQUALS_TEST:
            train_set = dataset.iloc[train_index]
            test_set = dataset.iloc[test_index]
            if EQUALIZE_TEST_WITH_NON_SECS:
                # Ignores the train_index and test_index
                train_set = dataset[dataset[f"{split_i}_train"]]
                test_set = dataset[dataset[f"{split_i}_test"]]
                train_index, test_index = (
                    None,
                    None,
                )  # train_index and test_index no longer have the same meaning as before
        else:
            train_set = dataset.copy()
            test_set = dataset.copy()
        if NO_TEST_SAME_DAY:
            test_set = test_set[test_set["elapsed_days"] > 0].copy()
        if NO_TRAIN_SAME_DAY:
            train_set = train_set[train_set["elapsed_days"] > 0].copy()

        testsets.append(test_set)
        partition_weights = {}
        for partition in train_set["partition"].unique():
            try:
                train_partition = train_set[train_set["partition"] == partition].copy()
                if not TRAIN_EQUALS_TEST:
                    assert (
                        train_partition["review_th"].max() < test_set["review_th"].min()
                    )
                if RECENCY:
                    x = np.linspace(0, 1, len(train_partition))
                    train_partition["weights"] = 0.25 + 0.75 * np.power(x, 3)

                model = Model()
                if DRY_RUN:
                    partition_weights[partition] = model.state_dict()
                    continue

                if MODEL_NAME == "LSTM":
                    model = model.to(DEVICE)
                    inner_opt = get_inner_opt(
                        model.parameters(), path=f"./pretrain/{OPT_NAME}_pretrain.pth"
                    )
                    trained_model = finetune(
                        train_partition, model, inner_opt.state_dict()
                    )
                    partition_weights[partition] = copy.deepcopy(
                        trained_model.state_dict()
                    )
                else:
                    trainer = Trainer(
                        model,
                        train_partition,
                        None,
                        n_epoch=model.n_epoch,
                        lr=model.lr,
                        wd=model.wd,
                        batch_size=batch_size,
                        hyperparams=params,
                    )
                    partition_weights[partition] = trainer.train()
                    print("weights", partition_weights[partition])
                    for name, param in partition_weights[partition].items():
                        assert (param >= 0).all(), "neg weights after ."

            except Exception as e:
                if str(e).endswith("inadequate."):
                    if verbose_inadequate_data:
                        print("Skipping - Inadequate data")
                else:
                    print(f"User: {user_id}")
                    print(e)
                    raise e
                partition_weights[partition] = Model().state_dict()
        w_list.append(partition_weights)

        if TRAIN_EQUALS_TEST:
            break

    p = []
    y = []
    save_tmp = []

    for i, (w, testset) in enumerate(zip(w_list, testsets)):
        for partition in testset["partition"].unique():
            partition_testset = testset[testset["partition"] == partition].copy()
            weights = w.get(partition, None)
            model = Model()
            model.load_state_dict(weights)
            my_collection = Collection(model)
            retentions, stabilities, difficulties = my_collection.batch_predict(
                partition_testset
            )
            partition_testset["p"] = retentions
            if stabilities:
                partition_testset["s"] = stabilities
            if difficulties:
                partition_testset["d"] = difficulties
            p.extend(retentions)
            y.extend(partition_testset["y"].tolist())
            save_tmp.append(partition_testset)

    save_tmp = pd.concat(save_tmp)
    del save_tmp["tensor"]
    if FILE:
        save_tmp.to_csv(f"evaluation/{FILE_NAME}/{user_id}.tsv", sep="\t", index=False)

    stats, raw = evaluate(y, p, save_tmp, FILE_NAME, user_id, w_list)
    print(stats)
    return stats, raw


def evaluate(y, p, df, file_name, user_id, w_list=None):
    if PLOT:
        fig = plt.figure()
        plot_brier(p, y, ax=fig.add_subplot(111))
        fig.savefig(f"evaluation/{file_name}/{user_id}.png")
    p_calibrated = lowess(
        y, p, it=0, delta=0.01 * (max(p) - min(p)), return_sorted=False
    )
    ici = np.mean(np.abs(p_calibrated - p))
    rmse_raw = root_mean_squared_error(y_true=y, y_pred=p)
    logloss = log_loss(y_true=y, y_pred=p, labels=[0, 1])
    rmse_bins = rmse_matrix(df)
    try:
        auc = round(roc_auc_score(y_true=y, y_score=p), 6)
    except:
        auc = None
    stats = {
        "metrics": {
            "RMSE": round(rmse_raw, 6),
            "LogLoss": round(logloss, 6),
            "RMSE(bins)": round(rmse_bins, 6),
            "ICI": round(ici, 6),
            "AUC": auc,
        },
        "user": int(user_id),
        "size": len(y),
    }
    if (
        w_list
        and type(w_list[0]) == dict
        and all(isinstance(w, list) for w in w_list[0].values())
    ):
        stats["parameters"] = {
            int(partition): list(map(lambda x: round(x, 6), w))
            for partition, w in w_list[-1].items()
        }
    elif WEIGHTS:
        Path(f"weights/{file_name}").mkdir(parents=True, exist_ok=True)
        torch.save(w_list[-1], f"weights/{file_name}/{user_id}.pth")
    if RAW:
        raw = {
            "user": int(user_id),
            "p": list(map(lambda x: round(x, 4), p)),
            "y": list(map(int, y)),
        }
    else:
        raw = None
    return stats, raw


NUM_FSRS_PARAMS = 21


def objective(trial, df_list):
    # per-parameter lr
    lrs = []
    for i in range(NUM_FSRS_PARAMS):
        lr_i = trial.suggest_float(f"lr_{i}", 5e-3, 4e-1, log=True)
        # lr_i = 4e-2
        lrs.append(lr_i)

    # betas
    beta1 = trial.suggest_float(f"beta1", 0.9**10, 0.9 ** (1.0 / 10), log=True)
    beta2 = trial.suggest_float(f"beta2", 0.7, 0.999, log=True)
    beta1s = NUM_FSRS_PARAMS * [beta1]
    beta2s = NUM_FSRS_PARAMS * [beta2]
    epsilon = trial.suggest_float(f"eps", 1e-5, 1e-0, log=True)
    epsilons = NUM_FSRS_PARAMS * [epsilon]

    # beta1s, beta2s = [], []
    # for i in range(NUM_FSRS_PARAMS):
    # beta1_i = trial.suggest_float(f"beta1_{i}", 0.85, 0.95, log=True)
    # beta2_i = trial.suggest_float(f"beta2_{i}", 0.9, 0.9999, log=True)
    # beta1s.append(beta1_i)
    # beta2s.append(beta2_i)
    params = {
        "lrs": np.array(lrs),
        "beta1s": np.array(beta1s),
        "beta2s": np.array(beta2s),
        "epsilons": np.array(epsilons),
    }
    logloss_tot = 0
    size_tot = 0
    encountered_nan = False
    for df_i, df in enumerate(df_list):
        user = df.iloc[0]["user_id"]
        if not encountered_nan:
            try:
                stats, raw = process(user, df, params)
                logloss_tot += stats["metrics"]["LogLoss"] * stats["size"]
                size_tot += stats["size"]
            except ValueError as e:
                print("ValueError encountered. LRs might be too high.")
                encountered_nan = True

        if encountered_nan:
            trial.report(3.0, step=df_i)
        else:
            trial.report(logloss_tot / size_tot, step=df_i)

        if trial.should_prune():
            raise optuna.TrialPruned()

    if encountered_nan:
        return 3.0
    else:
        return logloss_tot / size_tot


def get_initial_trial():
    params = {}
    params["beta1"] = 0.9
    params["beta2"] = 0.999
    params["eps"] = 1e-8
    for i in range(NUM_FSRS_PARAMS):
        params[f"lr_{i}"] = 4e-2
        # params[f"beta1_{i}"] = 0.9
        # params[f"beta2_{i}"] = 0.999
    print(params)
    return params


def process_user(user_id):
    dataset = pd.read_parquet(
        DATA_PATH / "revlogs", filters=[("user_id", "=", user_id)]
    )
    dataset = create_features(dataset, model_name=MODEL_NAME, secs_ivl=SECS_IVL)
    return user_id, dataset


STUDY_NAME = "parallel_study"
STORAGE_NAME = "sqlite:///tune_fsrs.db"


def worker(df_list):
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_NAME)

    def optuna_objective(trial):
        return objective(trial, df_list)

    print("Starting.")
    study.optimize(optuna_objective, n_trials=200)
    print("Done.")


TEST_PARAMS = {
    "lr_0": 0.016177384447800675,
    "lr_1": 0.1707061273142042,
    "lr_2": 0.06707101642492873,
    "lr_3": 0.15430275870041393,
    "lr_4": 0.02993275032551017,
    "lr_5": 0.012204786010969916,
    "lr_6": 0.1675268292405444,
    "lr_7": 0.39517432783032763,
    "lr_8": 0.01237788929131578,
    "lr_9": 0.006107253999689876,
    "lr_10": 0.1010752152937203,
    "lr_11": 0.0833382535109321,
    "lr_12": 0.019452990211649722,
    "lr_13": 0.006899844322187149,
    "lr_14": 0.3243853096633517,
    "lr_15": 0.02129906354817224,
    "lr_16": 0.09736974913886626,
    "lr_17": 0.012795033098247408,
    "lr_18": 0.010609181291426146,
    "lr_19": 0.17264348274791047,
    "lr_20": 0.3531538053435248,
    "beta1": 0.5132884932202296,
    "beta2": 0.7521293622919385,
    "eps": 0.0002993057988564082,
}

if __name__ == "__main__":
    assert MODEL_NAME == "FSRS-6"
    users = [i for i in range(1, 2)]

    df_dict = {}
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_user,
                user_id,
            )
            for user_id in users
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            user_id, dataset = future.result()
            df_dict[user_id] = dataset

    df_list = [df_dict[user_id] for user_id in users]

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        storage=STORAGE_NAME,
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(),
    )
    # study.enqueue_trial(get_initial_trial())
    study.enqueue_trial(TEST_PARAMS)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # worker(df_list)

    processes = [Process(target=worker, args=(df_list,)) for _ in range(PROCESSES)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
