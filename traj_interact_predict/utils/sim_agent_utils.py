"""
Module for filtering out a specific agent's data from trajectories, roles, pairs, and labels.

Provides:
- filter_my_id: Removes one agent's data and updates related metadata accordingly.
"""

import torch


def filter_my_id(trajectories, roles_tensor, agent_mask, pairs, labels, my_id):
    """
    Filter out the agent with ID `my_id` from trajectory data and related metadata.

    Args:
        trajectories (torch.Tensor): Tensor of shape (lookback, num_agents, features),
            containing trajectories of all agents.
        roles_tensor (torch.Tensor): Tensor of shape (num_agents,), agent role IDs.
        agent_mask (torch.Tensor): Tensor of shape (num_agents,), boolean mask for agents.
        pairs (torch.Tensor): Tensor of shape (num_pairs, 2), pairs of agent indices.
        labels (torch.Tensor): Tensor of shape (num_pairs,), interaction labels for pairs.
        my_id (int): ID of the agent to filter out.

    Returns:
        tuple:
            filtered_trajectories (torch.Tensor): Trajectories excluding `my_id`, shape
                (lookback, num_agents - 1, features).
            filtered_roles (torch.Tensor): Roles excluding `my_id`, shape (num_agents - 1,).
            filtered_mask (torch.Tensor): Mask excluding `my_id`, shape (num_agents - 1,).
            filtered_pairs (torch.Tensor): Pairs with `my_id` removed, updated indices.
            filtered_labels (torch.Tensor): Labels corresponding to filtered pairs.
            filtered_ids (torch.Tensor): Original agent IDs excluding `my_id`.
    """
    num_agents = trajectories.shape[1]

    # Create original ID array
    original_ids = torch.arange(num_agents)

    # Filter trajectories along agent dim
    filtered_trajectories = torch.cat(
        [trajectories[:, :my_id, :], trajectories[:, my_id + 1 :, :]], dim=1
    )

    # Filter roles, mask, and ids
    filtered_roles = torch.cat([roles_tensor[:my_id], roles_tensor[my_id + 1 :]])
    filtered_mask = torch.cat([agent_mask[:my_id], agent_mask[my_id + 1 :]])
    filtered_ids = torch.cat([original_ids[:my_id], original_ids[my_id + 1 :]])

    # Filter pairs and labels
    mask_pairs_keep = (pairs != my_id).all(dim=1)

    filtered_pairs = pairs[mask_pairs_keep].clone()
    filtered_labels = labels[mask_pairs_keep]

    # Update pair indices (> my_id → minus 1)
    filtered_pairs[filtered_pairs > my_id] -= 1

    return (
        filtered_trajectories,
        filtered_roles,
        filtered_mask,
        filtered_pairs,
        filtered_labels,
        filtered_ids,
    )
