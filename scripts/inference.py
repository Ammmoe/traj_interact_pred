"""
Inference function for the dual-encoder Bi-GRU drone interaction model.

Loads the trained model and config from an experiment directory, runs a forward pass on
a single sample of drone trajectories and agent info, and returns raw logits for agent pairs.

Inputs:
- trajectories: [1, num_agents, lookback, features] tensor
- roles: [num_agents] tensor (0=friendly, 1=unauthorized, 2=padded)
- pairs: [num_pairs, 2] tensor of agent pairs
- agent_mask: [num_agents] bool tensor indicating valid agents
- experiment_dir: path to model weights and config

Output:
- logits: [num_pairs] tensor of raw interaction scores

Automatically uses GPU if available.
"""

import os
import json
import torch
from traj_interact_predict.models.bi_gru_encoder import TrajEmbeddingExtractor
from traj_interact_predict.models.dual_encoder_classifier import DualEncoderModel


def run_inference(
    trajectories: torch.Tensor,  # shape: [1, num_agents, lookback, features]
    roles: torch.Tensor,  # shape: [num_agents], values 0=friendly, 1=unauthorized, 2=padded
    pairs: torch.Tensor,  # shape: [num_pairs, 2], agent pair ids (friendly -> unauth)
    agent_mask: torch.Tensor,  # shape: [num_agents], True = valid agent, False = padded
    experiment_dir: str,  # experiment directory
) -> torch.Tensor:
    """
    Run inference on a single sample.

    Args:
        trajectories: Tensor of shape [1, num_agents, lookback, features].
        roles: Tensor of shape [num_agents].
        pairs: Tensor of shape [num_pairs, 2].
        agent_mask: Bool tensor of shape [num_agents].
        model_dir: Path to folder containing 'best_model.pt' and 'config.json'.

    Returns:
        logits: Tensor of shape [num_pairs], raw logits before sigmoid.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Get encoder params used in the trained model
    encoder_params = config.get("encoder_params")

    # Initialize trajectory encoders
    encoder_friendly = TrajEmbeddingExtractor(**encoder_params).to(device)
    encoder_unauth = TrajEmbeddingExtractor(**encoder_params).to(device)

    # Initialize dual encoder model
    model = DualEncoderModel(
        encoder_friendly=encoder_friendly,
        encoder_unauth=encoder_unauth,
        embedding_dim=encoder_params["embedding_dim"],
    ).to(device)

    # Load weights
    model_path = os.path.join(experiment_dir, "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Move inputs to device
    trajectories = trajectories.to(device)
    roles = roles.to(device)
    pairs = pairs.to(device)
    agent_mask = agent_mask.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(trajectories, roles, pairs, agent_mask)

    return logits.squeeze()  # shape: [num_pairs]
