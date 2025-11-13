"""
Module: drone_interaction_dataset

This module defines the DroneInteractionDataset class, a PyTorch Dataset designed
to load and preprocess multi-drone trajectory and relationship data for interaction
prediction tasks.

The dataset handles loading trajectory and relation CSV files, encoding drone roles,
filtering timesteps with complete drone data, and providing temporal context windows
and corresponding relationship labels for model training.

Typical usage example:
    dataset = DroneInteractionDataset("trajectories.csv", "relations.csv", lookback=50, device="cuda")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
"""

import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np


class DroneInteractionDataset(Dataset):
    """
    PyTorch Dataset for drone interaction prediction.

    Loads trajectory and relation CSVs, encodes roles, and filters timesteps
    with complete data. Provides lookback windows of features and current
    timestep interaction labels.

    Args:
        trajectory_csv (str): Path to drone trajectory CSV file.
        relation_csv (str): Path to drone relationship CSV file.
        lookback (int): Number of past timesteps to include.
        device (str): Torch device for output tensors.

    Returns:
        dict: Contains context window, current features, relationships,
            labels, flight ID, and timestamp.
    """

    def __init__(self, trajectory_csv, relation_csv, device, lookback=50):
        self.lookback = lookback
        self.device = device

        # Load CSVs
        self.traj_df = pd.read_csv(trajectory_csv)
        self.relation_df = pd.read_csv(relation_csv)

        # Encode roles (friendly=1, unauth=0)
        self.traj_df["role_id"] = LabelEncoder().fit_transform(self.traj_df["role"])

        # Sort by drone_id and timestamp for easier slicing
        self.traj_df = self.traj_df.sort_values(
            ["flight_id", "time_stamp", "drone_id"]
        ).reset_index(drop=True)

        self.rel_df = self.rel_df.sort_values(["flight_id", "time_stamp"]).reset_index(
            drop=True
        )

        # Extract unique flights and group relation data in a dict
        self.flights = list(self.traj_df["flight_id"].unique())
        self.flight_data = {fid: df for fid, df in self.traj_df.groupby("flight_id")}
        self.rel_data = {fid: df for fid, df in self.rel_df.groupby("flight_id")}

        # Precompute available timesteps per flight
        self.flight_timesteps = {
            fid: sorted(df["time_stamp"].unique())
            for fid, df in self.flight_data.items()
        }
        self.valid_indices = self._build_indices()

    # Build valid (flight_id, timestep_index) pairs with full data
    def _build_indices(self):
        indices = []
        for fid, timesteps in self.flight_timesteps.items():
            for i in range(self.lookback, len(timesteps)):
                current_time = timesteps[i]
                df_current = self.flight_data[fid][["time_stamp"] == current_time]
                if len(df_current) == 6:  # Only keep full timesteps
                    indices.append((fid, i))
                else:
                    print(
                        f"[Skipping] Flight {fid}, timestep {current_time} has {len(df_current)} drones."
                    )
        return indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        fid, i = self.valid_indices[idx]
        timesteps = self.flight_timesteps[fid]
        past_times = timesteps[i - self.lookback : i]
        current_time = timesteps[i]

        # Build feature tensors for lookback window
        past_features = []
        for t in past_times:
            df_t = self.flight_data[fid][self.flight_data[fid]["time_stamp"] == t]
            feats = df_t[
                ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "role_id"]
            ].values
            past_features.append(feats)
        past_features = np.stack(past_features)  # [lookback, num_drones, feat_dim]

        # Current features
        current_df = self.flight_data[fid][
            self.flight_data[fid]["time_stamp"] == current_time
        ]
        current_features = current_df[
            ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "role_id"]
        ].values  # [num_drones, feat_dim]

        # Relationships at current timestep
        rel_t = self.rel_data[fid][self.rel_data[fid]["time_stamp"] == current_time]
        rel_pairs = rel_t[["drone_id", "target_id"]].values  # [num_pairs, 2]
        rel_labels = (
            (rel_t["relationship"] == "following").astype(np.float32).values
        )  # [num_pairs]

        return {
            "context_window": torch.tensor(
                past_features, dtype=torch.float32, device=self.device
            ),
            "current_features": torch.tensor(
                current_features, dtype=torch.float32, device=self.device
            ),
            "relationships": torch.tensor(
                rel_pairs, dtype=torch.long, device=self.device
            ),
            "labels": torch.tensor(rel_labels, dtype=torch.float32, device=self.device),
            "flight_id": fid,
            "time_stamp": current_time,
        }
