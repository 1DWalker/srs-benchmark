import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.fsrs_v7 import FSRS7
from models.lstm import LSTM

SECOND = 1 / 86400
MINUTE = 60 * SECOND
HOUR = 60 * MINUTE

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

def inverse_lstm(lstm_params, retentions, history, min_interval=SECOND, max_interval=50 * 365 * 24 * 60 * 60):
    """Return intervals to hit near the retentions, or nan if impossible within reasonable interval bounds"""
    lstm = LSTM(CustomConfig(), lstm_params)
    L, B, _ = history.shape
    start_l = torch.full_like(retentions, min_interval)
    start_r = torch.full_like(retentions, max_interval)
    w_nh, s_nh, d_nh = lstm.batch_process(history, torch.tensor([1]), torch.full((1,), L), B)["curve_params"]
    def eval_x(x):
        return lstm.forgetting_curve(x.unsqueeze(-1), w_nh, s_nh, d_nh)
    def predicate(x):
        return eval_x(x) < retentions
    result = batched_binary_search(start_l, start_r, predicate)
    pred = eval_x(result)
    answer = torch.where(torch.abs(pred - retentions) < 1e-3, result, torch.full_like(result, np.nan))
    return answer

def eval_fsrs(fsrs_params, history, intervals):
    fsrs = FSRS7(CustomConfig(), fsrs_params)
    L, B, _ = history.shape
    return fsrs.batch_process(history, intervals, torch.full((1,), L), B)["retentions"]

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
        assert user_to_params[user] is not None, f"{user} LSTM file not found."
    return user_to_params

def create_plot(x, y_preds, title, plot_diff=False,min_val=0.6):
    """
    x: (N,)
    y_preds: (B, N) with NaN prefix/suffix per row
    plot_diff: if True, plot (y - x) and remove identity line
    min_val: if set, only plot points where x >= min_val AND y >= min_val
    """

    B, N = y_preds.shape
    all_vals = []  # collect for axis scaling in diff mode

    # Plot each batch line (only non-NaN region)
    for b in range(B):
        y = y_preds[b]
        mask = ~np.isnan(y)

        if min_val is not None:
            mask &= (x >= min_val) & (y >= min_val)

        y_plot = y[mask] - x[mask] if plot_diff else y[mask]
        all_vals.append(y_plot)

        plt.plot(
            x[mask],
            y_plot,
            color="black",
            alpha=0.10,
            linewidth=1
        )

    # Convert to torch for nan-safe stats
    y_t = torch.tensor(y_preds)

    mean = torch.nanmean(y_t, dim=0).numpy()
    median = torch.nanmedian(y_t, dim=0).values.numpy()

    valid = ~np.isnan(mean)

    if min_val is not None:
        valid &= (x >= min_val) & (mean >= min_val)

    mean_plot = mean[valid] - x[valid] if plot_diff else mean[valid]
    median_plot = median[valid] - x[valid] if plot_diff else median[valid]

    all_vals += [mean_plot, median_plot]

    # Overlay mean & median
    plt.plot(x[valid], mean_plot, linewidth=2, label="mean")
    plt.plot(x[valid], median_plot, linewidth=2, label="median")

    # Identity line (only if not diff mode)
    lo = min_val if min_val is not None else 0
    lims = [lo, 1]
    if not plot_diff:
        plt.plot(lims, lims, linewidth=2, label="identity", color="black")
        plt.xlim(lims)
        plt.ylim(lims)
        plt.gca().set_aspect("equal", adjustable="box")

    else:
        # Center y-axis around 0
        max_abs = np.nanmax(np.abs(np.concatenate(all_vals)))
        plt.ylim(-max_abs, max_abs)
        plt.axhline(0, linewidth=2, color="black")

    plt.xlabel("LSTM R")
    plt.ylabel("FSRS-7-dev R - LSTM R" if plot_diff else "FSRS-7-dev R")
    plt.legend()
    plt.title(title)
    plt.show()

def format_history(history):
    def fmt_time(t_days):
        secs = t_days / SECOND  # convert to seconds

        if secs < 60:
            return f"{round(secs)}s"
        elif secs < 3600:
            return f"{round(secs / 60)}m"
        elif secs < 86400:
            return f"{round(secs / 3600)}h"
        else:
            return f"{round(secs / 86400)}d"

    parts = []
    for t, r in history:
        parts.append(f"({fmt_time(t)}, r={r})")

    return " → ".join(parts)


@torch.no_grad()
def main():
    history = [[0, 1]]
    # history = [[0, 1], [12 * HOUR, 3], [5, 1], [10 * MINUTE, 1]]
    # history = [[0, 1], [600 * SECOND, 3], [1, 1], [600 * SECOND, 3]]
    # history = [[0, 1], [600 * SECOND, 1], [600 * SECOND, 3]]
    # history = [[0, 1], [600 * SECOND, 1], [600 * SECOND, 1], [600 * SECOND, 1], [600 * SECOND, 3]]
    r_space = torch.linspace(0.999, 0.001, 399)
    sequences = torch.tensor(history).unsqueeze(1)
    user_to_fsrs_params = get_fsrs_params_dict()

    users = list(range(1, 201))
    user_to_lstm_params = get_lstm_params_dict(users)
    fsrs_preds = []
    for user in users:
        intervals = inverse_lstm(user_to_lstm_params[user], r_space, sequences)
        lstm_pred = eval_fsrs(user_to_fsrs_params[user], sequences, intervals)
        fsrs_preds.append(lstm_pred)

    title = format_history(history) + f"; {len(users)} users"
    create_plot(x=r_space.numpy(), y_preds=torch.stack(fsrs_preds).numpy(), title=title)



if __name__ == '__main__':
    main()