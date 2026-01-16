import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.fsrs_v7 import FSRS7
from models.lstm import LSTM

SECOND = 1 / 86400

class CustomConfig:
    def __init__(self):
        self.lstm_use_duration = False
        self.use_secs_intervals = True
        self.device = torch.device("cpu")
        _s_min_base = 0.0001 if self.use_secs_intervals else 0.01
        self.s_min = _s_min_base

        self.init_s_max: float = 100.0  # Max initial stability
        self.s_max: float = 36500.0  # Max stability (e.g., 100 years)



def batched_binary_search(l, r, f, iters=60):
    for _ in range(iters):
        mid = (l + r) / 2
        cond = f(mid)
        l, r = torch.where(cond, l, mid), torch.where(cond, mid, r)
    return l

def inverse_fsrs(fsrs_params, retentions, history, min_interval=SECOND, max_interval=50 * 365 * 24 * 60 * 60):
    """Return intervals to hit near the retentions, or nan if impossible within reasonable interval bounds"""
    fsrs = FSRS7(CustomConfig(), fsrs_params)
    L, B, _ = history.shape
    start_l = torch.full_like(retentions, min_interval)
    start_r = torch.full_like(retentions, max_interval)
    stabilities = fsrs.batch_process(history, start_l, torch.full((1,), L), B)["stabilities"]
    def eval_x(x):
        return fsrs.forgetting_curve_current_params(x, stabilities)
    def predicate(x):
        return eval_x(x) < retentions
    result = batched_binary_search(start_l, start_r, predicate)
    pred = eval_x(result)
    answer = torch.where(torch.abs(pred - retentions) < 1e-3, result, torch.full_like(result, np.nan))

    return answer

def eval_lstm(lstm_params, history, intervals):
    lstm = LSTM(CustomConfig(), lstm_params)
    L, B, _ = history.shape
    w_nh, s_nh, d_nh = lstm.batch_process(history, torch.tensor([1]), torch.full((1,), L), B)["curve_params"]
    result = lstm.forgetting_curve(intervals.unsqueeze(-1), w_nh, s_nh, d_nh)
    return result


def get_fsrs_params_dict():
    path = "result/FSRS-7-secs-recency.jsonl"
    user_to_params = {}

    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            user = obj["user"]
            params = obj["parameters"]["0"]
            user_to_params[user] = params

    return user_to_params

def get_lstm_params_dict(users):
    user_to_params = {}
    for user in users:
        user_to_params[user] = torch.load(f"lstm_weights/lstm_user_{user}.pth", weights_only=True)
    return user_to_params

def create_plot(x, y_preds, plot_diff=False):
    """
    x: (N,)
    y_preds: (B, N) with NaN prefix/suffix per row
    plot_diff: if True, plot (y - x) and remove identity line
    """

    B, N = y_preds.shape

    all_vals = []  # collect for axis scaling in diff mode

    # Plot each batch line (only non-NaN region)
    for b in range(B):
        y = y_preds[b]
        mask = ~np.isnan(y)

        y_plot = y[mask] - x[mask] if plot_diff else y[mask]
        all_vals.append(y_plot)

        plt.plot(
            x[mask],
            y_plot,
            color="black",
            alpha=0.15,
            linewidth=1
        )

    # Convert to torch for nan-safe stats
    y_t = torch.tensor(y_preds)

    mean = torch.nanmean(y_t, dim=0).numpy()
    median = torch.nanmedian(y_t, dim=0).values.numpy()

    valid = ~np.isnan(mean)

    mean_plot = mean[valid] - x[valid] if plot_diff else mean[valid]
    median_plot = median[valid] - x[valid] if plot_diff else median[valid]

    all_vals += [mean_plot, median_plot]

    # Overlay mean & median
    plt.plot(x[valid], mean_plot, linewidth=2, label="mean")
    plt.plot(x[valid], median_plot, linewidth=2, label="median")

    # Identity line (only if not diff mode)
    lims = [0, 1]
    if not plot_diff:
        plt.plot(lims, lims, linestyle=":", linewidth=2, label="y = x")

        plt.xlim(lims)
        plt.ylim(lims)
        plt.gca().set_aspect("equal", adjustable="box")

    else:
        # Center y-axis around 0
        max_abs = np.nanmax(np.abs(np.concatenate(all_vals)))
        plt.ylim(-max_abs, max_abs)
        plt.axhline(0, linestyle=":", linewidth=1)

    plt.xlabel("x (probability)")
    plt.ylabel("y_pred - x" if plot_diff else "y_pred (probability)")
    plt.legend()
    plt.show()


@torch.no_grad()
def main():
    history = [[0, 1], [600 * SECOND, 3], [1, 1], [600 * SECOND, 3]]
    r_space = torch.linspace(0.99, 0.01, 199)
    sequences = torch.tensor(history).unsqueeze(1)
    user_to_fsrs_params = get_fsrs_params_dict()

    users = list(range(1, 101))
    # for user in range(1, 101):
    lstm_params = get_lstm_params_dict(users)
    lstm_preds = []
    for user in users:
        intervals = inverse_fsrs(user_to_fsrs_params[user], r_space, sequences)
        # print("int days", intervals)
        # print("int secs", intervals / SECOND)
        lstm_pred = eval_lstm(lstm_params[user], sequences, intervals)
        lstm_preds.append(lstm_pred)
        # print("pred", lstm_pred)

    create_plot(x=r_space.numpy(), y_preds=torch.stack(lstm_preds).numpy())



if __name__ == '__main__':
    main()