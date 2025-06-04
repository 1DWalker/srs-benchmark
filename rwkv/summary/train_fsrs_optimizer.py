import torch
from rwkv.summary.summary_model import FSRSSummaryModel
from rwkv.utils import get_number_of_trainable_parameters
import random

from utils import parse_toml

WEIGHT_DECAY = 1e-2
ADAMW_EPS = 1e-18
ADAMW_BETAS = (0.90, 0.999)
CLIP = 0.5

def get_optimizer(config, model):
    encode_params = []
    decay_params = []
    channel_mixer_params = []
    decay_head_params = []
    other_params = []
    head_targets = [
        "fsrs_linear",
    ]
    for name, param in model.named_parameters():
        # Param constraint is to exclude layer/group norm weights
        if (
            "weight" in name
            and "lora" not in name
            and "scale" not in name
            and len(param.squeeze().shape) >= 2
        ):
            is_head_param = False
            for head_target in head_targets:
                if head_target in name:
                    is_head_param = True
            if is_head_param:
                decay_head_params.append(param)
            elif "features2card" in name:
                encode_params.append(param)
            elif "channel_mixer" in name:
                channel_mixer_params.append(param)
            else:
                decay_params.append(param)
        else:
            other_params.append(param)

    return torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": WEIGHT_DECAY,
                "lr": config.PEAK_LR,
            },
            {
                "params": channel_mixer_params,
                "weight_decay": WEIGHT_DECAY,
                "lr": config.PEAK_LR,
            },
            {
                "params": decay_head_params,
                "weight_decay": WEIGHT_DECAY,
                "lr": config.PEAK_LR,
            },
            {"params": encode_params, "weight_decay": 1e-2, "lr": config.PEAK_LR},
            {"params": other_params, "weight_decay": 0.0, "lr": config.PEAK_LR},
        ],
        eps=ADAMW_EPS,
        betas=ADAMW_BETAS,
    )

def main(config):
    random.seed(config.SEED)

    master_model = FSRSSummaryModel()
    model = FSRSSummaryModel().selective_cast(config.DTYPE).to(config.DEVICE)
    optimizer = get_optimizer(config, master_model)
    print("Number of trainable parameters:", get_number_of_trainable_parameters(model))

    train_users = list(range(config.TRAIN_USER_START, config.TRAIN_USER_END + 1))
    validate_users = list(range(config.VALIDATE_USER_START, config.VALIDATE_USER_END + 1))

    step = -1
    for epoch in range(int(1e9)):
        random.shuffle(train_users)
        for user in train_users:
            step += 1


            if step == 100:
                break
        if step == 100:
            break

    exit()

if __name__ == '__main__':
    config = parse_toml()
    main(config)