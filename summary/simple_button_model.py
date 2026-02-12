import json
import os
import torch

def normalize(a):
    t = sum(a)
    return [x / t for x in a]

class SimpleButtonModel:
    def __init__(self, repo_path):
        path = os.path.join(repo_path, "button_usage.jsonl")
        self.by_user = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                user_id = obj.pop("user")  # remove user from inner dict
                self.by_user[user_id] = obj     # remaining fields go into per-user dict

    def get_first_dist(self, user, device):
        return torch.tensor(normalize(self.by_user[user]["first_rating_prob"]), device=device)