import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for (
        batch_trajectories,
        batch_roles,
        batch_agent_mask,
        pairs_list,
        labels_list,
    ) in tqdm(loader, desc="Training", leave=False):
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
def evaluate_model(model, loader, device):
    model.eval()
    all_logits, all_preds, all_labels = [], [], []

    for (
        batch_trajectories,
        batch_roles,
        batch_agent_mask,
        pairs_list,
        labels_list,
    ) in tqdm(loader, desc="Evaluating", leave=False):
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

        # Predictions
        preds = (torch.sigmoid(logits) >= 0.5).long()
        
        all_logits.append(logits.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_logits, all_preds, all_labels
