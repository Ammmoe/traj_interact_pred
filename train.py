import torch
import joblib
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import Subset
from data.data_loader import DroneInteractionDataset
from models.pretrained_model_loader import load_pretrained_traj_model


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

# Load scalers
scaler_X = joblib.load(pretrained_model_dir / "scaler_X.pkl")

# Train test split
flight_ids = dataset.flights
num_train = int(0.8 * len(flight_ids))
train_flights = flight_ids[:num_train]
test_flights = flight_ids[num_train:]

# Train and test indices
train_indices = [
    i for i, (fid, _) in enumerate(dataset.valid_indices) if fid in train_flights
]
test_indices = [
    i for i, (fid, _) in enumerate(dataset.valid_indices) if fid in test_flights
]

# Train and test datasets
train_ds = Subset(dataset, train_indices)
test_ds = Subset(dataset, test_indices)

# Create dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


