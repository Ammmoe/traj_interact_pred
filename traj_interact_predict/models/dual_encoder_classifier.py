"""
dual_encoder_classifier.py

Defines a dual-encoder neural network model for binary classification of relationships
between pairs of agents, using separate encoders for friendly and unauthorized agents.

The model extracts embeddings for each agent's trajectory, constructs a relation vector
by combining embeddings, and classifies the relationship (e.g., "following" or "none").
"""

import torch
import torch.nn as nn


class DualEncoderModel(nn.Module):
    """
    Dual-encoder model for binary relation classification between agent pairs.

    Uses separate encoders for friendly and unauthorized agents to generate embeddings.
    Relation vectors are formed by combining embeddings and their interactions,
    then classified to predict relationships (e.g., "following" or "none").

    Args:
        encoder_friendly (nn.Module): Encoder for friendly agents.
        encoder_unauth (nn.Module): Encoder for unauthorized agents.
        embedding_dim (int): Dimension of the agent embeddings.
    """

    def __init__(self, encoder_friendly, encoder_unauth, embedding_dim):
        super().__init__()

        self.encoder_friendly = encoder_friendly  # friendly encoder
        self.encoder_unauth = encoder_unauth  # unauthorized encoder

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 5, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),  # binary classification: following or none
        )

        self.layer_norm = nn.LayerNorm(embedding_dim * 5)

        self.cross_agent_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=4, batch_first=True
        )

        self.attn_norm = nn.LayerNorm(embedding_dim)

        self.role_embedding = nn.Embedding(
            num_embeddings=2,  # 0=friendly, 1=unauthorized
            embedding_dim=embedding_dim,
        )

    def forward(self, batch_trajectories, batch_roles, pairs_list, batch_agent_mask):
        """
        Predicts relation logits for given agent pairs in a batch.

        Args:
            batch_trajectories (Tensor): [batch_size, lookback, max_agents, feat_dim] trajectories.
            batch_roles (Tensor): [batch_size, max_agents] role labels (0=friendly, 1=unauthorized).
            pairs_list (list of Tensors): Length batch_size; each tensor [num_pairs, 2] of
                                        (friendly_id, unauth_id).
            batch_agent_mask (Tensor): [batch_size, max_agents] boolean mask (True=valid agent,
                                        False=padded).

        Returns:
            list of Tensors: Length batch_size; each tensor [num_pairs, 1] of logits.
        """
        device = batch_trajectories.device
        batch_size, lookback, max_agents, feat_dim = batch_trajectories.shape
        embed_dim = self.encoder_friendly.embedding_dim

        # Flatten agents into batch dimension for encoding
        traj_flat = batch_trajectories.permute(
            0, 2, 1, 3
        )  # [batch, max_agents, lookback, feat_dim]
        traj_flat = traj_flat.reshape(batch_size * max_agents, lookback, feat_dim)
        roles_flat = batch_roles.reshape(batch_size * max_agents)
        agent_mask_flat = batch_agent_mask.reshape(batch_size * max_agents)

        # Encode all agents using dual encoders
        embeddings = torch.zeros(
            batch_size * max_agents, embed_dim, device=device
        )

        friendly_mask = (roles_flat == 0) & agent_mask_flat
        unauth_mask = (roles_flat == 1) & agent_mask_flat

        if friendly_mask.any():
            embeddings[friendly_mask] = self.encoder_friendly(traj_flat[friendly_mask])

        if unauth_mask.any():
            embeddings[unauth_mask] = self.encoder_unauth(traj_flat[unauth_mask])

        agent_embeddings = embeddings.view(
            batch_size, max_agents, embed_dim
        )  # [batch_size, max_agents, embed_dim]

        # Add role embeddings
        role_embeddings = self.role_embedding(
            batch_roles
        )  # [batch_size, max_agents, embed_dim]
        agent_embeddings = (
            agent_embeddings + role_embeddings
        )  # [batch_size, max_agents, embed_dim]

        # Cross-agent self attention layer (batched)
        attn_out, _ = self.cross_agent_attn(
            query=agent_embeddings,
            key=agent_embeddings,
            value=agent_embeddings,
            key_padding_mask=~batch_agent_mask,  # Mask padded agents
        )  # [batch_size, max_agents, embed_dim]

        agent_embeddings = self.attn_norm(agent_embeddings + attn_out)

        # Flatten all pairs across the batch for classification
        batch_ids = []
        friendly_ids = []
        unauth_ids = []

        for batch_i, pairs in enumerate(pairs_list):
            if len(pairs) == 0:
                continue
            batch_ids.append(
                torch.full((len(pairs),), batch_i, device=device, dtype=torch.long)
            )
            friendly_ids.append(pairs[:, 0].to(device))
            unauth_ids.append(pairs[:, 1].to(device))

        if len(batch_ids) == 0:
            return [torch.empty((0, 1), device=device) for _ in range(batch_size)]

        batch_ids = torch.cat(batch_ids)  # [total_pairs]
        friendly_ids = torch.cat(friendly_ids)  # [total_pairs]
        unauth_ids = torch.cat(unauth_ids)  # [total_pairs]

        # Gather embeddings for all pairs
        emb_friendly = agent_embeddings[
            batch_ids, friendly_ids, :
        ]  # [total_pairs, embed_dim]
        emb_unauth = agent_embeddings[
            batch_ids, unauth_ids, :
        ]  # [total_pairs, embed_dim]

        # Build relation vectors
        relation_vector = torch.cat(
            [
                emb_friendly,
                emb_unauth,
                torch.abs(emb_friendly - emb_unauth),
                emb_friendly - emb_unauth,
                emb_friendly * emb_unauth,
            ],
            dim=-1,
        )  # [total_pairs, 5 * embed_dim]

        relation_vector = self.layer_norm(relation_vector)
        logits = self.classifier(relation_vector)  # [total_pairs, 1]

        # Split logits back into batch-wise lists
        logits_all = []
        start_idx = 0
        for pairs in pairs_list:
            num_pairs = len(pairs)
            logits_all.append(logits[start_idx : start_idx + num_pairs])
            start_idx += num_pairs

        return logits_all
