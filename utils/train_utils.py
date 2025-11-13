import torch
from tqdm import tqdm
from models.pretrained_model_loader import extract_traj_embeddings


def balance_pairs(relationships, labels, max_neg_per_pos=1):
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_indices = torch.nonzero(pos_mask).squeeze(1)  # Indices of positive samples
    neg_indices = torch.nonzero(neg_mask).squeeze(1)  # Indices of negative samples

    num_pos = len(pos_indices)
    num_neg = len(neg_indices)

    if num_pos == 0 or num_neg == 0:
        return relationships, labels  # Skip balancing if only one class present

    # Sample negatives
    num_keep_neg = min(num_neg, num_pos * max_neg_per_pos)
    sampled_neg_indices = neg_indices[torch.randperm(num_neg)[:num_keep_neg]]

    # Combine positive and sampled negative indices and shuffle
    keep_indices = torch.cat([pos_indices, sampled_neg_indices])
    keep_indices = keep_indices[torch.randperm(len(keep_indices))]

    return relationships[keep_indices], labels[keep_indices]


def train_epoch(
    model, loader, optimizer, criterion, scaler_X, pretrained_model, config, device
):
    model.train()
    pretrained_model.eval()  # pretrained model weights are frozen
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=True, ncols=100):
        # Reset gradients
        optimizer.zero_grad()
        batch_loss = 0.0

        # Move tensors to device and process each sample in batch
        context_window = batch["context_window"].to(
            device
        )  # [B, num_drones, lookback, feat_dim]
        current_features = batch["current_features"].to(
            device
        )  # [B, num_drones, feat_dim]
        relationships = batch["relationships"]  # [B, num_pairs, 2]
        labels = batch["labels"]  # [B, num_pairs]

        # Get features_per_agent from config
        features_per_agent = config["FEATURES_PER_AGENT"]

        batch_size = context_window.shape[0]  # Batch size

        # Reshape for extracting trajectory embeddings
        # Flatten num_drones and feat_dim for pretrained model input
        traj_data = context_window[:, :, :, :features_per_agent].reshape(
            batch_size, config["LOOK_BACK"], -1
        )  # Exclude role_id

        # Extract trajectory embeddings for entire batch
        traj_embeddings = extract_traj_embeddings(
            pretrained_model,
            traj_data=traj_data,
            scaler_X=scaler_X,
            lookback=config["LOOK_BACK"],
            features_per_agent=features_per_agent,
            device=device,
        )  # returns [batch_size, num_drones, embedding_dim]

        for b in range(batch_size):  # Iterate over batch
            rel = relationships[b].to(device)  # [num_pairs, 2]
            label = labels[b].to(device)  # [num_pairs]

            # Balance pairs per sample
            balanced_rel, balanced_label = balance_pairs(rel, label, max_neg_per_pos=1)

            preds = model(current_features[b], traj_embeddings[b], balanced_rel)
            loss = criterion(preds, balanced_label)
            batch_loss += loss

        # Backpropagation
        batch_loss /= batch_size  # Average loss over batch
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

    return total_loss / len(loader) # Average loss over all batches
