import torch
import json
from pathlib import Path
from importlib import import_module


def load_pretrained_traj_model(model_dir: Path, device: torch.device):
    experiment_dir = Path(model_dir)
    config_path = experiment_dir / "config.json"
    model_path = experiment_dir / "last_model.pt"

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Import model module dynamically
    model_module = import_module(config["model_module"])
    ModelClass = getattr(model_module, config["model_class"])
    model = ModelClass(**config["model_params"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, config
