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
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from traj_interact_predict.models.bi_gru_encoder import TrajEmbeddingExtractor
from traj_interact_predict.models.dual_encoder_classifier import DualEncoderModel
from traj_interact_predict.utils.logger import get_logger
from traj_interact_predict.data.data_loader import load_datasets
from traj_interact_predict.utils.train_utils import calculate_evaluation_scores


# pylint: disable=all
def run_inference(
    trajectories: torch.Tensor,  # shape: [1, num_agents, lookback, features]
    roles: torch.Tensor,  # shape: [num_agents], values 0=friendly, 1=unauthorized, 2=padded
    pairs: torch.Tensor,  # shape: [num_pairs, 2], agent pair ids (friendly -> unauth)
    agent_mask: torch.Tensor,  # shape: [num_agents], True = valid agent, False = padded
    experiment_dir: str = "experiments/20251201_180721",  # experiment directory
) -> torch.Tensor:  # shape: [num_pairs]
    """
    Run inference on a single sample.

    Args:
        trajectories: Tensor of shape [batch, num_agents, lookback, features].
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
        logits_list = model(trajectories, roles, pairs, agent_mask)

    # logits_list is a list of tensors, each [num_pairs_i, 1], concatenate along dim=0
    logits = torch.cat(logits_list, dim=0).squeeze(-1)  # [total_num_pairs]

    return logits


def main():
    """
    Runs inference on the test dataset, logs timing and accuracy metrics.

    - Loads configuration and datasets.
    - Performs batch-wise inference using the trained model.
    - Records total and average inference time.
    - Computes and logs evaluation accuracy.

    Entry point for the inference script.
    """
    # Setup logger
    experiment_dir = "experiments/20251201_180721"
    logger, _ = get_logger(exp_dir=experiment_dir, log_name="inference.log")

    # Log start of new inference session
    logger.info("\n" + "=" * 80 + "\nStarting New Inference Session\n" + "=" * 80)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Experiment folder: %s", experiment_dir)

    # Load config
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Inference config
    batch_size = 32
    num_friendly_to_pad = 2
    num_unauth_to_pad = 2

    # Log batch size and padded agents
    logger.info("Inference batch size: %d", batch_size)
    logger.info(
        "Friendly agents padded during dataset preparation: %s", num_friendly_to_pad
    )
    logger.info(
        "Unauthorised agents padded during dataset preparation: %s", num_unauth_to_pad
    )

    # Load datasets
    _, _, test_set = load_datasets(
        trajectory_csv="data/drone_states.csv",
        relation_csv="data/drone_relations.csv",
        val_split=0.15,
        test_split=0.15,
        lookback=config.get("LOOK_BACK"),
        device=device,
        max_agents=config.get("MAX_AGENTS"),
        num_friendly_to_pad=num_friendly_to_pad,
        num_unauth_to_pad=num_unauth_to_pad,
    )

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    logger.info("Starting inference on test dataset...")

    all_logits, all_preds, all_labels = [], [], []

    # Record inference time
    start_time = time.time()
    for batch in tqdm(test_loader, desc="Test Inference", leave=True, ncols=100):
        # Unpack batch (adapt to your actual output of your loader)
        trajectories, roles, agent_mask, pairs, labels = batch

        # Run inference for this sample
        logits = run_inference(
            trajectories,
            roles,
            pairs,
            agent_mask,
            experiment_dir=experiment_dir,
        )

        # Apply sigmoid and threshold for predictions
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        # labels shape is [1, num_pairs], squeeze it to [num_pairs]
        labels = labels.squeeze(0)  # remove batch dim

        all_logits.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    # Log inference time
    end_time = time.time()
    elapsed_time = end_time - start_time
    num_batches = len(test_loader)
    avg_inference_time = elapsed_time / num_batches

    logger.info("Total inference time: %.2f seconds", elapsed_time)
    logger.info("Average inference time per batch: %.4f seconds", avg_inference_time)
    logger.info(
        "Average inference time per sequence: %.4f seconds",
        avg_inference_time / batch_size,
    )

    # Concatenate all results across samples
    all_logits = torch.cat(all_logits, dim=0)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).flatten().numpy()

    # Evaluate results using your existing utils
    accuracy = calculate_evaluation_scores(
        labels=all_labels,
        preds=all_preds,
        probs=all_logits,
        phase="Inference",
        logger=logger,
        exp_dir=experiment_dir,
    )

    logger.info("Final Inference Accuracy: %.4f", accuracy)


if __name__ == "__main__":
    main()
