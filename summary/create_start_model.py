from pathlib import Path

from summary import fsrs_encoder_model, fsrs_encoder_model_curve
import torch
from utils import parse_toml

def main(config):
    model_path = f"{config.LOAD_MODEL_FOLDER}/{config.LOAD_MODEL_NAME}.pth"
    # optim_path = f"{config.LOAD_MODEL_FOLDER}/{config.LOAD_MODEL_NAME}_optim.pth"
    model = fsrs_encoder_model.Model()
    filtered_state = {}
    model_state = model.state_dict()
    loaded_state = torch.load(model_path, weights_only=True)
    for name, param in loaded_state.items():
        if name in model_state:
            if model.is_frozen_param(name):
                print("copy:", name)
                filtered_state[name] = param
            else:
                print("exclude load:", name)
                pass
            pass
        else:
            # print("missing:", name)
            pass
        pass
    model.load_state_dict(filtered_state, strict=False)
    save_model_path = (
        f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_{0}.pth"
    )
    # save_optim_path = f"{config.SAVE_MODEL_FOLDER}/{config.SAVE_MODEL_PREFIX}_{step}_optim.pth"
    Path(config.SAVE_MODEL_FOLDER).mkdir(parents=True, exist_ok=True)               
    torch.save(model.state_dict(), save_model_path)
    print("Saved.")

if __name__ == '__main__':
    config = parse_toml()
    main(config)