"""
data_loader.py

This module defines the DroneInteractionDataset, a PyTorch Dataset for
predicting interaction relations between drones in flight sequences.

The dataset loads drone trajectory and relationship data from CSV files,
processes sliding windows of timesteps with full agent presence, and
returns padded trajectory tensors, agent roles, interaction pairs, and labels.

Intended for use in multi-agent trajectory modeling and interaction classification
tasks where drones are categorized as friendly or unauthorized.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np


class DroneInteractionDataset(Dataset):
    """
    PyTorch Dataset for drone interaction relation prediction.

    This dataset processes drone trajectory data and interaction labels over
    sliding time windows for each flight. It filters timesteps to only include
    those with complete agent data, pads missing data if needed, and creates
    pairs of friendly and unauthorized drones for interaction classification.

    Data is loaded from two CSV files: one for trajectories (position, velocity,
    roles over time), and one for interaction relations (e.g., 'following').

    Each sample consists of:
    - A tensor of drone trajectories over a lookback window, shaped [lookback, num_agents, 6]
        (with features: pos_x, pos_y, pos_z, vel_x, vel_y, vel_z).
    - A tensor of agent role labels (friendly=0, unauthorized=1).
    - A boolean mask indicating valid agents in the sample.
    - A tensor of agent index pairs (friendly → unauthorized) for interaction prediction.
    - A tensor of corresponding binary labels for each pair indicating interaction presence.

    Args:
        trajectory_csv (str): Path to the CSV file containing drone trajectories.
        relation_csv (str): Path to the CSV file containing drone interactions.
        lookback (int): Number of past timesteps in each sample window.
        device (str or torch.device): Device to load tensors onto.
        expected_agents (int): Number of agents expected per timestep (used for filtering).
        transform (callable, optional): Optional transform to apply on trajectory tensors.
        stride (int): Step size for sliding windows.

    Example:
        dataset = DroneInteractionDataset(
            trajectory_csv="traj.csv",
            relation_csv="relations.csv",
            lookback=50,
            device="cuda",
            expected_agents=6,
            stride=1,
        )
    """

    def __init__(
        self,
        trajectory_csv,
        relation_csv,
        lookback=50,
        device="cpu",
        max_agents=6,
        transform=None,
        stride=1,
    ):
        self.traj_df = pd.read_csv(trajectory_csv)
        self.relation_df = pd.read_csv(relation_csv)
        self.lookback = lookback
        self.device = device
        self.transform = transform
        self.max_agents = max_agents
        self.transform = transform
        self.stride = stride

        # Role mapping
        self.role_map = {"friendly": 0, "unauthorized": 1}

        # Prepare samples (flight_id + start_idx of sliding window)
        self.samples = []  # list of (flight_id, start_idx)
        self.flight_groups = {}
        self.flight_valid_timesteps = {}

        flights = self.traj_df["flight_id"].unique()
        for fid in flights:
            flight_data = self.traj_df[self.traj_df["flight_id"] == fid].sort_values(
                "time_stamp"
            )
            self.flight_groups[fid] = flight_data

            # Find valid timesteps with all agents present
            # Skip timesteps with missing agents
            # Skip if not enough valid timesteps for lookback
            valid_timesteps = [
                t
                for t, df_t in flight_data.groupby("time_stamp")
                if len(df_t) == self.max_agents
            ]
            if len(valid_timesteps) < lookback:
                continue
            self.flight_valid_timesteps[fid] = valid_timesteps

            # Sliding window samples
            for start_idx in range(0, len(valid_timesteps) - self.lookback + 1, stride):
                self.samples.append((fid, start_idx))

        print(f"Total samples (sliding windows): {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flight_id, start_idx = self.samples[idx]
        flight_data = self.flight_groups[flight_id]
        sliding_window_ts = self.flight_valid_timesteps[flight_id][
            start_idx : start_idx + self.lookback
        ]

        agents = flight_data["drone_id"].unique().tolist()
        num_agents = len(agents)
        agent_id2idx = {agent: i for i, agent in enumerate(agents)}

        trajectories = []
        roles = []

        for agent in agents:
            agent_rows = flight_data[flight_data["drone_id"] == agent].set_index(
                "time_stamp"
            )
            agent_traj = []
            last_row = np.zeros(
                6, dtype=np.float32
            )  # pos_x, pos_y, pos_z, vel_x, vel_y, vel_z

            for t in sliding_window_ts:
                if t in agent_rows.index:
                    row = agent_rows.loc[
                        t, ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"]
                    ].values.astype(np.float32)
                    last_row = row
                agent_traj.append(last_row.copy())  # copy to avoid references

            agent_traj = np.stack(agent_traj, axis=0)  # [lookback, 6]
            trajectories.append(agent_traj)
            roles.append(agent_rows["role"].iloc[0])

        trajectories = np.stack(trajectories, axis=0)  # [num_agents, lookback, 6]
        trajectories = torch.tensor(
            trajectories, dtype=torch.float32, device=self.device
        )
        if self.transform:
            trajectories = self.transform(trajectories)
        trajectories = trajectories.transpose(0, 1)  # [lookback, num_agents, 6]

        roles_tensor = torch.tensor(
            [self.role_map[r] for r in roles], dtype=torch.long, device=self.device
        )
        agent_mask = torch.ones(num_agents, dtype=torch.bool, device=self.device)

        # Build pairs and labels
        pairs, labels = [], []
        friendly_agents = [
            a
            for a in agents
            if self.role_map[flight_data[flight_data["drone_id"] == a]["role"].iloc[0]]
            == 0
        ]
        unauth_agents = [
            a
            for a in agents
            if self.role_map[flight_data[flight_data["drone_id"] == a]["role"].iloc[0]]
            == 1
        ]

        flight_relations = self.relation_df[self.relation_df["flight_id"] == flight_id]

        for src in friendly_agents:
            for tgt in unauth_agents:
                pairs.append([agent_id2idx[src], agent_id2idx[tgt]])
                mask = (flight_relations["drone_id"] == src) & (
                    flight_relations["target_id"] == tgt
                )
                labels.append(
                    int(
                        (flight_relations[mask]["relationship"] == "following").any()
                        if mask.any()
                        else 0
                    )
                )

        return (
            trajectories,  # [lookback, num_agents, 6]
            roles_tensor,  # [num_agents]
            agent_mask,  # [num_agents]
            torch.tensor(pairs, dtype=torch.long, device=self.device),  # [num_pairs, 2]
            torch.tensor(labels, dtype=torch.long, device=self.device),  # [num_pairs]
        )
        
        
def collate_fn(batch):
    batch_size = len(batch)
    lookback = batch[0][0].shape[0]
    max_agents = max(item[0].shape[1] for item in batch)
    
