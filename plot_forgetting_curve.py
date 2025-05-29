import random
import lmdb
from matplotlib import pyplot as plt
import numpy as np
import torch

from full_db_config import FULL_DB_PATH
from utils import load_tensor, print_env_size, transfer_lmdb

def fsrs6_forgetting_curve(t, s, decay):
    factor = 0.9 ** (1 / -decay) - 1
    return (1 + factor * t / s) ** -decay

def rwkv_raw_forgetting_curve(label_elapsed_seconds, w):
    num_curves = 128
    s_point_spread = 18.5
    s_max = 22
    s_space_raw = torch.exp(
        torch.linspace(0, s_point_spread, num_curves, device=w.device)
    )
    s_space = 0.1 + (s_space_raw - 1) * (np.e ** (s_max - s_point_spread))
    label_elapsed_seconds = torch.max(torch.tensor(0.001), label_elapsed_seconds)
    return 1e-5 + (1 - 2 * 1e-5) * torch.sum(
        w * 0.9 ** (label_elapsed_seconds / s_space), dim=-1
    )

def rwkv_forgetting_curve(label_elapsed_seconds, w, out_ahead_logits):
    label_elapsed_seconds = label_elapsed_seconds.unsqueeze(0).unsqueeze(0)
    w = w.unsqueeze(0)
    out_ahead_logits = out_ahead_logits.unsqueeze(0)
    def interp(out_ahead_logits, label_elapsed_seconds):
        label_elapsed_seconds = torch.clamp(label_elapsed_seconds.contiguous(), min=1)
        max_e = 21
        point_spread = 18.5
        num_points = 128
        point_space_raw = torch.exp(
            torch.linspace(
                0, point_spread, num_points, device=out_ahead_logits.device
            )
        )
        point_space = 0.5 + (point_space_raw - 1) * (
            np.e ** (max_e - point_spread)
        )
        right_idx = torch.searchsorted(point_space, label_elapsed_seconds)
        left_idx = torch.clamp(right_idx - 1, min=0)
        xl, xr = point_space[left_idx], point_space[right_idx]
        yl = torch.gather(out_ahead_logits, dim=-1, index=left_idx)
        yr = torch.gather(out_ahead_logits, dim=-1, index=right_idx)
        res = 1e-5 + (1 - 2 * 1e-5) * (
            yl + (yr - yl) * (label_elapsed_seconds - xl) / (xr - xl)
        )
        return res

    curve_probs_raw = rwkv_raw_forgetting_curve(label_elapsed_seconds, w)
    curve_logits_raw = torch.log(
        curve_probs_raw / (1 - curve_probs_raw)
    )  # inverse sigmoid
    ahead_logit_residual = interp(out_ahead_logits, label_elapsed_seconds)
    curve_logits = curve_logits_raw + ahead_logit_residual
    return torch.sigmoid(curve_logits).item()



@torch.inference_mode()
def main():
    # transfer_lmdb("full_db", "full_db_2")
    # exit()
    print("Setting random seed.")
    random.seed(123)

    user_id = 11
    env = lmdb.open(FULL_DB_PATH, readonly=True, lock=False)
    print_env_size(env)

    with env.begin() as txn:
        print("start")
        rwkv_w = load_tensor(txn, f"{user_id}_RWKV_w", device=torch.device("cpu"))
        rwkv_ahead_logits = load_tensor(txn, f"{user_id}_RWKV_ahead_logits", device=torch.device("cpu"))
        fsrs_s = load_tensor(txn, f"{user_id}_FSRS-6-recency_s", device=torch.device("cpu")).numpy()
        fsrs_decay = load_tensor(txn, f"{user_id}_FSRS-6-recency_decay", device=torch.device("cpu")).numpy()
        card_ids = load_tensor(txn, f"{user_id}_card_id", device=torch.device("cpu")).numpy()
        ys = load_tensor(txn, f"{user_id}_y", device=torch.device("cpu")).numpy()
        elapsed_seconds_list = load_tensor(txn, f"{user_id}_elapsed_seconds", device=torch.device("cpu")).numpy()
        same_day_review = load_tensor(txn, f"{user_id}_same_day_review", device=torch.device("cpu")).numpy()
        card_id_locs = {}
        for i, card_id in enumerate(card_ids):
            if fsrs_s[i] != -1:
                if card_id not in card_id_locs:
                    elapsed_seconds_list[i] = 0  # Fix some values
                    card_id_locs[card_id] = []
                card_id_locs[card_id].append(i)

        # print(card_id_locs)
        # for k, v in card_id_locs.items():
        #     tot_elapsed_time = 0
        #     if len(v) > 8:
        #         for i in v:
        #             tot_elapsed_time += elapsed_seconds_list[i]
        #         if tot_elapsed_time < 20 * 86400:
        #             print(k, v)
        #             exit()

        # card_id = 1725
        # card_id = 2558
        possible_card_ids = list(card_id_locs.keys())
        possible_card_ids = [x for x in possible_card_ids if len(card_id_locs[x]) > 3]
        random.shuffle(possible_card_ids)

        # possible_card_ids = [7372] * 100

        for card_id in possible_card_ids[:20]:
            print("Running", card_id)

            locs = card_id_locs[card_id]
            tot_time_seconds = 0
            cumulative_seconds = []
            for i in locs:
                tot_time_seconds += elapsed_seconds_list[i]
                cumulative_seconds.append(tot_time_seconds)
                # print(i, fsrs_s[i], fsrs_decay[i], elapsed_seconds_list[i] / 86400, ys[i])

            cumulative_seconds = np.array(cumulative_seconds)

            t_space = list(np.linspace(0.01, tot_time_seconds * 1.3, num=1000))
            # Add some extra values so RWKV's instant forgetting will appear & lapses aren't skipped
            for t in cumulative_seconds:
                if t - 0.01 > 1e-5:
                    t_space.append(t - 0.01)
                t_space.append(t + 0.01)
                t_space.append(t + 0.1)
            t_space.sort()
            t_space = np.array(t_space)

            curve_spaces = []
            fsrs_subcurves = []
            rwkv_raw_subcurves = []
            rwkv_subcurves = []
            # fsrs_p = []
            # rwkv_p = []
            tot = 0
            for cum_i in range(len(cumulative_seconds)):
                space_subcurve = []
                fsrs_subcurve = []
                rwkv_raw_subcurve = []
                rwkv_subcurve = []
                for t_i, t in enumerate(t_space):
                    if t >= cumulative_seconds[cum_i] and (cum_i == len(cumulative_seconds) - 1 or cumulative_seconds[cum_i + 1] > t):
                        tot += 1
                        i = locs[cum_i]
                        space_subcurve.append(t)
                        fsrs_subcurve.append(fsrs6_forgetting_curve((t - cumulative_seconds[cum_i]) / 86400, fsrs_s[i], fsrs_decay[i]))
                        rwkv_raw_subcurve.append(rwkv_raw_forgetting_curve(torch.tensor(t - cumulative_seconds[cum_i]), rwkv_w[i]))
                        rwkv_subcurve.append(rwkv_forgetting_curve(torch.tensor(t - cumulative_seconds[cum_i]), rwkv_w[i], rwkv_ahead_logits[i]))
                
                curve_spaces.append(np.array(space_subcurve))
                fsrs_subcurves.append(np.array(fsrs_subcurve))
                rwkv_raw_subcurves.append(np.array(rwkv_raw_subcurve))
                rwkv_subcurves.append(np.array(rwkv_subcurve))
            assert tot == len(t_space)

            plt.clf()
            plt.figure(figsize=(12, 4))
            # Create the plot
            for curve_i, (curve_space, fsrs_subcurve, rwkv_raw_subcurve, rwkv_subcurve) in enumerate(zip(curve_spaces, fsrs_subcurves, rwkv_raw_subcurves, rwkv_subcurves)):
                FSRS_COLOR = '#1f77b4' 
                RWKV_RAW_COLOR = '#ff7f0e'
                RWKV_COLOR = '#2ca02c'
                if curve_i == 0:
                    plt.plot(curve_space / 86400, rwkv_raw_subcurve, label='RWKV-raw', color=RWKV_RAW_COLOR)
                    plt.plot(curve_space / 86400, rwkv_subcurve, label='RWKV', color=RWKV_COLOR)
                    plt.plot(curve_space / 86400, fsrs_subcurve, label='FSRS-6-recency', color=FSRS_COLOR)
                else:
                    # No label to avoid duplicates in the legend
                    plt.plot(curve_space / 86400, rwkv_raw_subcurve, color=RWKV_RAW_COLOR)
                    plt.plot(curve_space / 86400, rwkv_subcurve, color=RWKV_COLOR)
                    plt.plot(curve_space / 86400, fsrs_subcurve, color=FSRS_COLOR)

            for i, review_time_seconds in zip(locs[1:], cumulative_seconds[1:]):
                if not same_day_review[i]:
                    color = 'r' if ys[i] == 0 else 'g'
                    plt.axvline(x=review_time_seconds / 86400, color=color, linestyle='--')


            # Add labels and title
            plt.ylim(top=1.0)
            plt.xlabel('Days')
            plt.ylabel('R')
            plt.title(f'predicted R vs time. User: {user_id}, Card_id: {card_id}')

            # Add legend
            plt.legend()

            # Show the plot
            # plt.show()
            plt.savefig(f'plots/forgetting-curves/{user_id}_{card_id}.png', dpi=600, bbox_inches='tight')
            print("Saved.")
            # exit()

if __name__ == '__main__':
    main()