import torch
from pathlib import Path
from data.data_loader import DroneInteractionDataset


# Pretrained trajectory prediction model directory
pretrained_model_dir = Path("experiments/20251112_170556")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = DroneInteractionDataset(
    trajectory_csv="data/drone_states.csv",
    relation_csv="data/drone_relations.csv",
    device=device.type,
    lookback=50,
)

# Load model + config
model, config = load_pretrained_traj_model(pretrained_model_dir, device)


