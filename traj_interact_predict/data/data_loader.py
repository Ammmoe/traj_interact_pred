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

import random
import torch
import pandas as pd
from torch.utils.data import Dataset, Subset
import numpy as np


class DroneInteractionDataset(Dataset):
    """
    Dataset for classifying drone interactions using sliding-window trajectory data.

    This dataset loads trajectory and interaction CSVs, builds fixed-length
    lookback windows per flight, handles missing agents via padding (optional,
    per-flight, and role-aware), and generates friendly→unauthorized agent pairs
    with corresponding interaction labels.

    Each sample returns:
        - trajectories: [lookback, num_agents, 6]
        - roles: [num_agents] (0=friendly, 1=unauthorized, 2=padded)
        - agent_mask: [num_agents] (1=valid, 0=padded)
        - pairs: [num_pairs, 2] friendly→unauthorized index pairs
        - labels: [num_pairs] binary interaction labels

    Args:
        trajectory_csv (str): Path to trajectory CSV.
        relation_csv (str): Path to interaction/relationship CSV.
        lookback (int): Sliding window length.
        device (str): Target device for tensors.
        max_agents (int): Max agents per sample after padding.
        stride (int): Step size for sliding windows.
        num_friendly_to_pad (int): Number of friendly agents to randomly zero-pad.
        num_unauth_to_pad (int): Number of unauthorized agents to randomly zero-pad.
        transform (callable): Optional transform for trajectories.
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
        num_friendly_to_pad=0,
        num_unauth_to_pad=0,
    ):
        self.traj_df = pd.read_csv(trajectory_csv)
        self.relation_df = pd.read_csv(relation_csv)
        self.lookback = lookback
        self.device = device
        self.transform = transform
        self.max_agents = max_agents
        self.transform = transform
        self.stride = stride
        self.num_friendly_to_pad = num_friendly_to_pad
        self.num_unauth_to_pad = num_unauth_to_pad
        self.flight_padding_map = {}  # Per-flight padding plan

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

            # Select drones to pad for this flight
            friendly_ids = (
                flight_data[flight_data["role"] == "friendly"]["drone_id"]
                .unique()
                .tolist()
            )
            unauth_ids = (
                flight_data[flight_data["role"] == "unauthorized"]["drone_id"]
                .unique()
                .tolist()
            )

            # Randomize the ids for more robust padding
            random.shuffle(friendly_ids)
            random.shuffle(unauth_ids)

            pad_friendly = friendly_ids[: self.num_friendly_to_pad]
            pad_unauth = unauth_ids[: self.num_unauth_to_pad]

            self.flight_padding_map[fid] = {
                "friendly": pad_friendly,
                "unauth": pad_unauth,
            }

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

        # Convert to numpy
        trajectories = np.stack(trajectories, axis=0)  # shape [num_agents, lookback, 6]

        # Apply per-flight padding plan in agent_id space
        pad_plan = self.flight_padding_map[flight_id]
        pad_ids = pad_plan["friendly"] + pad_plan["unauth"]

        for drone_id in pad_ids:
            if drone_id in agent_id2idx:
                pad_idx = agent_id2idx[drone_id]

                # Zero out the trajectory
                trajectories[pad_idx, :, :] = 0.0

                # Set role to padding role
                roles[pad_idx] = 2

                # Mark invalid
                agent_mask[pad_idx] = 0

        # Add padding if num_agents < max_agents
        if num_agents < self.max_agents:
            pad_size = self.max_agents - num_agents

            # Pad trajectories with zeros: shape [pad_size, lookback, 6]
            pad_traj = np.zeros((pad_size, self.lookback, 6), dtype=np.float32)

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
        # valid_agent_ids = agents  # Original list
        # Check for agent mask instead of looking at original unique list for inducing padding
        valid_agent_ids = [i for i, m in enumerate(agent_mask) if m == 1]

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
    trajectory_csv,
    relation_csv,
    lookback,
    device,
    val_split=0.15,
    test_split=0.15,
    max_agents=6,
    num_friendly_to_pad=0,
    num_unauth_to_pad=0,
):
    """
    Load the DroneInteractionDataset and split it into train/val/test subsets.

    Splits are computed by **deterministic contiguous slicing** (no shuffling).
    Useful when the dataset is already time-ordered.

    Args:
        val_split (float): Fraction of samples for validation.
        test_split (float): Fraction of samples for testing.
        trajectory_csv (str): Path to trajectory data CSV.
        relation_csv (str): Path to interaction/label CSV.
        lookback (int): Number of timesteps per sample.
        device (str): "cpu" or "cuda".
        max_agents (int): Max number of agents per sample.
        num_friendly_to_pad (int): How many friendly agents to randomly pad per flight.
        num_unauth_to_pad (int): How many unauthorized agents to randomly pad per flight.

    Returns:
        (train_set, val_set, test_set): Dataset subsets as torch.utils.data.Subset.
    """
    dataset = DroneInteractionDataset(
        trajectory_csv=trajectory_csv,
        relation_csv=relation_csv,
        lookback=lookback,
        device=device,
        max_agents=max_agents,
        num_friendly_to_pad=num_friendly_to_pad,
        num_unauth_to_pad=num_unauth_to_pad,
    )

    dataset_length = len(dataset)
    val_length = int(dataset_length * val_split)
    test_length = int(dataset_length * test_split)
    train_length = dataset_length - val_length - test_length

    # Deterministic split by index slicing
    train_indices = list(range(0, train_length))
    val_indices = list(range(train_length, train_length + val_length))
    test_indices = list(range(train_length + val_length, dataset_length))

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    return train_set, val_set, test_set
