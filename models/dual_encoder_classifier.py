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
    Dual-encoder model for relation classification between pairs of agents in a multi-agent system.

    The model uses two separate encoders to embed trajectories of friendly and unauthorized agents.
    It then constructs a relation vector by concatenating the embeddings and their element-wise interactions,
    which is fed into a classifier to predict the binary relation.

    Args:
        encoder_friendly (nn.Module): Encoder module for friendly agents.
        encoder_unauth (nn.Module): Encoder module for unauthorized agents.
        embedding_dim (int): Dimension of the output embedding for each agent.
    """

    def __init__(self, encoder_friendly, encoder_unauth, embedding_dim):
        super().__init__()

        self.encoder_friendly = encoder_friendly  # friendly encoder
        self.encoder_unauth = encoder_unauth  # unauthorized encoder

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),  # binary classification: following or none
        )

    def forward(self, batch_trajectories, batch_roles, pairs_list):
        """
        Forward pass to classify relationships between agent pairs in the batch.

        Args:
            batch_trajectories (Tensor): Tensor of shape [batch_size, lookback, max_agents, feat_dim]
                                        containing trajectories for all agents.
            pairs_list (list of Tensors): List length batch_size, each element is a tensor of shape [num_pairs, 2]
                                        specifying pairs of (friendly_agent_idx, unauthorized_agent_idx) for that batch item.

        Returns:
            list of Tensors: List length batch_size, each tensor of shape [num_pairs, 1] containing
                            logits for the relation classification of each pair.
        """
        logits_all = []
        batch_size, lookback, _, feat_dim = batch_trajectories.shape

        for batch_i in range(batch_size):
            roles = batch_roles[batch_i]  # [max_agents]

            # Select agent indices by role: 0=friendly, 1=unauthorized
            friendly_ids = (roles == 0).nonzero(as_tuple=True)[0]
            unauth_ids = (roles == 1).nonzero(as_tuple=True)[0]

            # Select trajectories for friendly and unauthorized agents
            traj_friendly = batch_trajectories[
                batch_i : batch_i + 1, :, friendly_ids, :
            ]  # [1, lookback, num_friendly, feat_dim]
            traj_unauth = batch_trajectories[
                batch_i : batch_i + 1, :, unauth_ids, :
            ]  # [1, lookback, num_unauth, feat_dim]

            # Reshape for encoder input: [batch=1, lookback, num_agents * feat_dim]
            num_friendly = traj_friendly.shape[2]
            num_unauth = traj_unauth.shape[2]
            friendly_input = traj_friendly.reshape(1, lookback, num_friendly * feat_dim)
            unauth_input = traj_unauth.reshape(1, lookback, num_unauth * feat_dim)

            # Encode once per role per batch sample
            emb_friendly_all = self.encoder_friendly(
                friendly_input
            )  # [1, num_friendly, embed_dim]
            emb_unauth_all = self.encoder_unauth(
                unauth_input
            )  # [1, num_unauth, embed_dim]

            # Map original agent indices to embedding indices for lookup
            friendly_id_map = {
                orig.item(): idx for idx, orig in enumerate(friendly_ids)
            }
            unauth_id_map = {orig.item(): idx for idx, orig in enumerate(unauth_ids)}

            # Compute logits for pairs
            pairs = pairs_list[batch_i]
            if len(pairs) == 0:
                logits_all.append(torch.empty((0, 1), device=batch_trajectories.device))
                continue

            batch_logits = []
            for friendly_id, unauth_id in pairs:
                f_idx = friendly_id_map.get(friendly_id.item(), None)
                u_idx = unauth_id_map.get(unauth_id.item(), None)

                # Actually None pairs are ommitted during data loading process
                # This is just a safety net # (sigmoid(0) = 0.5)
                if f_idx is None or u_idx is None:
                    batch_logits.append(
                        torch.zeros(1, 1, device=batch_trajectories.device)
                    )
                    continue

                emb_friendly = emb_friendly_all[0, f_idx, :].unsqueeze(
                    0
                )  # [1, embed_dim]
                emb_unauth = emb_unauth_all[0, u_idx, :].unsqueeze(0)  # [1, embed_dim]

                relation_vector = torch.cat(
                    [
                        emb_friendly,
                        emb_unauth,
                        torch.abs(emb_friendly - emb_unauth),
                        emb_friendly * emb_unauth,
                    ],
                    dim=-1,
                )

                logit = self.classifier(relation_vector)  # [1, 1]
                batch_logits.append(logit)

            batch_logits = torch.cat(batch_logits, dim=0)  # [num_pairs, 1]
            logits_all.append(batch_logits)

        return logits_all
