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

    def encode_agent(self, encoder, batch_trajectories, agent_id):
        """
        Extracts the embedding for a specific agent from the batch using the given encoder.

        Args:
            encoder (nn.Module): Encoder module to process agent trajectories.
            batch_trajectories (Tensor): Tensor of shape [batch_size, lookback, max_agents, feat_dim]
                                        containing trajectories for all agents in the batch.
            agent_id (int): Index of the agent to extract embedding for.

        Returns:
            Tensor: Embedding tensor of shape [batch_size, embedding_dim] for the specified agent.
        """
        batch_size, lookback, max_agents, feat_dim = batch_trajectories.shape
        batch_traj_reshaped = batch_trajectories.reshape(
            batch_size, lookback, max_agents * feat_dim
        )

        emb_all = encoder(
            batch_traj_reshaped, return_embedding=True
        )  # [batch_size, max_agents, embedding_dim]
        return emb_all[:, agent_id, :]

    def forward(self, batch_trajectories, pairs_list):
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

        for batch_i, pair in enumerate(pairs_list):
            if len(pair) == 0:
                logits_all.append(torch.empty((0, 1), device=batch_trajectories.device))
                continue

            # Extract batch slice
            batch_traj_i = batch_trajectories[batch_i : batch_i + 1]

            batch_logits = []

            for friendly_id, unauth_id in pair:
                # Encode friendly and unauthorized separately
                emb_friendly = self.encode_agent(
                    self.encoder_friendly, batch_traj_i, friendly_id
                )
                emb_unauth = self.encode_agent(
                    self.encoder_unauth, batch_traj_i, unauth_id
                )

                # Combine embeddings (standard relation vector)
                relation_vector = torch.cat(
                    [
                        emb_friendly,
                        emb_unauth,
                        torch.abs(emb_friendly - emb_unauth),
                        emb_friendly * emb_unauth,
                    ],
                    dim=-1,
                )

                # Classify relations
                logit = self.classifier(relation_vector)  # [1, 1]
                batch_logits.append(logit)

            # Concatenate logits for the batch
            batch_logits = torch.cat(batch_logits, dim=0)  # [num_pairs, 1]
            logits_all.append(batch_logits) # [batch_size length list of [num_pairs, 1]]

        return logits_all
