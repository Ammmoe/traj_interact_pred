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

            # Do not skip timesteps with missing agents
            # We will pad the missing agents later
            valid_timesteps = [t for t, _ in flight_data.groupby("time_stamp")]
            if len(valid_timesteps) < lookback:
                continue
            self.flight_valid_timesteps[fid] = valid_timesteps

            # Sliding window samples
            for start_idx in range(0, len(valid_timesteps) - self.lookback + 1, stride):
                self.samples.append((fid, start_idx))

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
        agent_mask = []  # We will add a mask for samples with missing agents

        for agent in agents:
            agent_rows = flight_data[flight_data["drone_id"] == agent].set_index(
                "time_stamp"
            )

            # Get role for this agent
            role_str = agent_rows["role"].iloc[0]
            role_val = self.role_map[role_str]

            # Build trajectory for this agent
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
            roles.append(role_val)
            agent_mask.append(1)  # Valid agent

        # Add padding if num_agents < max_agents
        if num_agents < self.max_agents:
            pad_size = self.max_agents - num_agents

            # Pad trajectories with zeros: shape [pad_size, lookback, 6]
            pad_traj = np.zeros((pad_size, self.lookback, 6), dtype=np.float32)
            trajectories = np.stack(
                trajectories, axis=0
            )  # shape [num_agents, lookback, 6]

            # Shape [max_agents, lookback, 6]
            trajectories = np.concatenate([trajectories, pad_traj], axis=0)

            # Pad roles: invalid [2]
            roles.extend([2] * pad_size)

            # Pad mask with 0 (invalid agent)
            agent_mask.extend([0] * pad_size)

        # Convert to torch
        trajectories = torch.tensor(
            trajectories, dtype=torch.float32, device=self.device
        )
        if self.transform:
            trajectories = self.transform(trajectories)
        trajectories = trajectories.transpose(0, 1)  # [lookback, num_agents, 6]

        roles_tensor = torch.tensor(roles, dtype=torch.long, device=self.device)

        # Mask tensor (1 = valid agent, 0 = padded)
        agent_mask = torch.tensor(agent_mask, dtype=torch.bool, device=self.device)

        # Only consider valid agents when building pairs
        valid_agent_ids = agents  # Original list

        # Build pairs and labels
        pairs, labels = [], []
        friendly_agents = [
            a
            for a in valid_agent_ids
            if self.role_map[flight_data[flight_data["drone_id"] == a]["role"].iloc[0]]
            == 0
        ]
        unauth_agents = [
            a
            for a in valid_agent_ids
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
    """
    Custom collate function for DroneInteractionDataset.

    This function takes a list of samples produced by the dataset, where each
    sample consists of:
        - trajectories: Tensor [lookback, num_agents, num_features]
        - roles:        Tensor [num_agents]
        - agent_mask:   Tensor [num_agents]
        - pairs:        Tensor [num_pairs, 2]
        - labels:       Tensor [num_pairs]

    Because different samples may contain different numbers of agents,
    this function pads the agent dimension of:
        - trajectories
        - roles
        - agent_mask

    Padding ensures that all samples in the batch have the same number of agents
    (max_agents across the batch). The padded agents are filled with zeros and
    masked out via the agent_mask.

    The pair lists and label lists are **not padded** because each sample
    may contain a different number of interaction pairs. They are therefore
    returned as Python lists of tensors with varying shapes.

    Returns:
        batch_trajectories: Tensor
            Shape [batch_size, lookback, max_agents, num_features]

        batch_roles: Tensor
            Shape [batch_size, max_agents]

        batch_agent_mask: Tensor
            Shape [batch_size, max_agents]

        pairs_list: list of length batch_size
            Each element is a Tensor [num_pairs, 2]

        labels_list: list of length batch_size
            Each element is a Tensor [num_pairs]
    """
    lookback = batch[0][0].shape[0]
    max_agents = max(item[0].shape[1] for item in batch)
    num_features = batch[0][0].shape[2]

    traj_list, roles_list, agent_mask_list, pairs_list, labels_list = [], [], [], [], []

    for trajectories, roles, agent_mask, pairs, labels in batch:
        num_agents = trajectories.shape[1]
        pad_size = max_agents - num_agents

        # Pad trajectories along agent dimension
        if pad_size > 0:
            pad_traj = torch.zeros(lookback, pad_size, num_features)
            trajectories = torch.cat([trajectories, pad_traj], dim=1)
            roles = torch.cat([roles, torch.zeros(pad_size, dtype=roles.dtype)])
            agent_mask = torch.cat(
                [agent_mask, torch.zeros(pad_size, dtype=agent_mask.dtype)]
            )

        traj_list.append(trajectories)
        roles_list.append(roles)
        agent_mask_list.append(agent_mask)
        pairs_list.append(pairs)
        labels_list.append(labels)

    batch_trajectories = torch.stack(
        traj_list, dim=0
    )  # [batch_size, lookback, max_agents, num_features]
    batch_roles = torch.stack(roles_list, dim=0)  # [batch_size, max_agents]
    batch_agent_mask = torch.stack(agent_mask_list, dim=0)  # [batch_size, max_agents]

    return batch_trajectories, batch_roles, batch_agent_mask, pairs_list, labels_list


def load_datasets(
    val_split, test_split, trajectory_csv, relation_csv, lookback, device, max_agents=6
):
    """
    Load and split the DroneInteractionDataset into training, validation, and test subsets.

    Args:
        val_split (float): Proportion of the dataset to allocate to the validation set (e.g., 0.1 for 10%).
        test_split (float): Proportion of the dataset to allocate to the test set (e.g., 0.1 for 10%).
        trajectory_csv (str): Path to the CSV file containing trajectory data.
        relation_csv (str): Path to the CSV file containing relation data.
        lookback (int): Number of timesteps to consider in each sample window.
        device (str): Device to load tensors on, e.g., "cpu" or "cuda".
        max_agents (int, optional): Maximum number of agents expected per sample. Defaults to 6.

    Returns:
        tuple: A tuple containing three subsets of the dataset:
            - train_set (torch.utils.data.Subset): Training subset.
            - val_set (torch.utils.data.Subset): Validation subset.
            - test_set (torch.utils.data.Subset): Test subset.
    """
    dataset = DroneInteractionDataset(
        trajectory_csv=trajectory_csv,
        relation_csv=relation_csv,
        lookback=lookback,
        device=device,
        max_agents=max_agents,
    )

    dataset_length = len(dataset)
    val_length = int(dataset_length * val_split)
    test_length = int(dataset_length * test_split)
    train_length = dataset_length - val_length - test_length

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_length, val_length, test_length]
    )
    return train_set, val_set, test_set
