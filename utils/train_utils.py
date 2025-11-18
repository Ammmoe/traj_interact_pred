"""
Training and evaluation utilities for multi-agent trajectory relation classification model.

Includes functions for:
- Training one epoch with backpropagation
- Evaluating model performance on validation/test set without gradient updates
- Printing detailed classification metrics and confusion matrix visualization

Dependencies:
- PyTorch for tensor operations and model training
- scikit-learn for evaluation metrics and confusion matrix plotting
- matplotlib for visualization
- tqdm for progress bars
"""

import torch
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Perform one training epoch over the dataset.

    Args:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): DataLoader providing training batches.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        criterion (torch.nn.Module): Loss function, e.g., BCEWithLogitsLoss.
        device (torch.device or str): Device to perform computation ('cpu' or 'cuda').

    Returns:
        float: Average training loss over all batches in the epoch.
    """
    model.train()
    total_loss = 0.0

    for (
        batch_trajectories,
        batch_roles,
        batch_agent_mask,
        pairs_list,
        labels_list,
    ) in tqdm(loader, desc="Training", leave=True, ncols=200):
        # Move tensors to device
        batch_trajectories = batch_trajectories.to(
            device
        )  # [batch_size, num_agents, lookback, feat_dim]
        batch_roles = batch_roles.to(device)  # [batch_size, num_agents]
        batch_agent_mask = batch_agent_mask.to(device)  # [batch_size, num_agents]
        pairs_list = [
            pair.to(device) for pair in pairs_list
        ]  # [batch_size, num_pairs, 2]
        labels_list = [
            label.to(device) for label in labels_list
        ]  # [batch_size, num_pairs]

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass: return batch_size length list of [num_pairs_i, 1]
        logits_list = model(batch_trajectories, pairs_list)

        # Pack into tensors for loss computation
        logits = torch.cat(logits_list, dim=0).squeeze(-1)  # [total_num_pairs]
        labels = torch.cat(labels_list, dim=0).float()  # [total_num_pairs]

        # Compute loss
        loss = criterion(logits, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)  # Average loss over all batches


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    """
    Evaluate the model on validation/test data without computing gradients.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader providing evaluation batches.
        device (torch.device or str): Device for computation.

    Returns:
        Tuple:
            - torch.Tensor: Logits for all samples concatenated.
            - numpy.ndarray: Predicted labels (0 or 1) for all samples.
            - numpy.ndarray: True labels for all samples.
    """
    model.eval()
    total_loss = 0.0
    all_logits, all_preds, all_labels = [], [], []

    for (
        batch_trajectories,
        batch_roles,
        batch_agent_mask,
        pairs_list,
        labels_list,
    ) in tqdm(loader, desc="Evaluating", leave=True, ncols=200):
        # Move tensors to device
        batch_trajectories = batch_trajectories.to(
            device
        )  # [batch_size, num_agents, lookback, feat_dim]
        batch_roles = batch_roles.to(device)  # [batch_size, num_agents]
        batch_agent_mask = batch_agent_mask.to(device)  # [batch_size, num_agents]
        pairs_list = [
            pair.to(device) for pair in pairs_list
        ]  # [batch_size, num_pairs, 2]
        labels_list = [
            label.to(device) for label in labels_list
        ]  # [batch_size, num_pairs]

        # Forward pass: return batch_size length list of [num_pairs_i, 1]
        logits_list = model(batch_trajectories, pairs_list)

        # Pack into tensors for metric computation
        logits = torch.cat(logits_list, dim=0).squeeze(-1)  # [total_num_pairs]
        labels = torch.cat(labels_list, dim=0)  # [total_num_pairs]

        # Compute loss
        loss = criterion(logits, labels)
        total_loss += loss.item()

        # Predictions
        preds = (torch.sigmoid(logits) >= 0.5).long()

        all_logits.append(logits.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_logits, all_preds.numpy(), all_labels.numpy(), total_loss / len(loader)


def calculate_evaluation_scores(labels, preds, probs, phase, logger, exp_dir):
    """
    Calculate and print evaluation metrics, plot confusion matrix,
    and save calibration plot.

    Args:
        labels (array-like or torch.Tensor): True binary labels.
        preds (array-like or torch.Tensor): Predicted binary labels.
        logits (array-like or torch.Tensor): Raw model logits (before sigmoid).
        phase (str): Identifier for the current evaluation stage, e.g. 'Epoch 5' or 'Test'.

    Returns:
        float: Accuracy score.

    Side Effects:
        Prints accuracy, precision, recall, and F1 scores.
        Saves confusion matrix and calibration plots as PNG files.
    """
    # If inputs are tensors, convert to numpy
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()
    if hasattr(preds, "cpu"):
        preds = preds.cpu().numpy()
    if hasattr(probs, "cpu"):
        probs = probs.cpu().numpy()

    # Compute evaluation metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)

    # Log evaluation metrics
    logger.info(f"{phase} Evaluation Metrics:")
    logger.info(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )
    logger.info(f"Confusion Matrix:\n{conf_matrix}")

    # Plot and save confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - Epoch {phase}")
    conf_mat_path = os.path.join(exp_dir, f"confusion_matrix_epoch_{phase}.png")
    plt.savefig(conf_mat_path)
    plt.close()

    # Calibration curve (reliability diagram)
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)

    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve - {phase}")
    plt.legend()
    plt.grid(True)
    calib_curve_path = os.path.join(exp_dir, f"calibration_curve_{phase}.png")
    plt.savefig(calib_curve_path)
    plt.close()

    return accuracy
