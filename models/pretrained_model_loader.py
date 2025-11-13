import torch
import json
from pathlib import Path
from importlib import import_module
from utils.scaler import scale_per_agent


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


def extract_traj_embeddings(
    pretrained_model,
    traj_data,
    scaler_X,
    lookback,
    features_per_agent,
    device,
):
    # Scale trajectory data
    # Convert to numpy before scaling
    if isinstance(traj_data, torch.Tensor):
        traj_data = traj_data.detach().cpu().numpy()
    traj_scaled = scale_per_agent(traj_data, scaler_X, features_per_agent) # (num_drones, lookback, features_per_agent)
    input_seq = traj_scaled[-lookback:]  # (num_drones, lookback, features_per_agent)
    
    # Convert back to tensor
    X_tensor = torch.from_numpy(input_seq.reshape(1, lookback, -1)).float().to(device)  # (1, lookback, num_drones * features_per_agent)

    return embeddings
