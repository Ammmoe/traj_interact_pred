import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = DroneInteractionDataset(
    trajectory_csv="data/drone_states.csv",
    relation_csv="data/drone_relations.csv",
    lookback=50,
    device=device,
)