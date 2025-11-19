"""
bi_gru_encoder.py

Defines a bidirectional GRU-based encoder module to extract fixed-size embeddings from multi-agent trajectories.

This module processes past trajectory sequences for multiple agents, encoding each agent's trajectory
into a learned embedding vector. These embeddings can be used as input features for downstream tasks
such as interaction classification, relation prediction, or trajectory clustering.

Features:
    - Bidirectional GRU encoder
    - Supports variable number of agents per batch by processing agents independently
    - Projects concatenated forward and backward hidden states into a fixed embedding dimension
"""

import torch
from torch import nn


class TrajEmbeddingExtractor(nn.Module):
    """
    Extracts fixed-size embeddings for multi-agent trajectories using a bidirectional GRU encoder.

    This module processes past trajectory sequences per agent and outputs learned embeddings,
    useful as input features for downstream tasks like classification or relation prediction.

    Features:
        - Bidirectional encoder GRU
        - Projects concatenated forward and backward hidden states to a fixed embedding dimension
        - Supports variable number of agents by processing each agent separately

    Args:
        input_size (int): Number of input features per agent per timestep (e.g., 6 for pos + vel).
        enc_hidden_size (int): Number of hidden units in the encoder GRU.
        embed_dim (int): Output embedding dimension for each agent.
        num_layers (int): Number of stacked GRU layers in the encoder.
    """

    def __init__(
        self, input_size=6, enc_hidden_size=64, embedding_dim=64, num_layers=1
    ):
        super().__init__()
        self.input_size = input_size
        self.enc_hidden_size = enc_hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Shared modules for all agents
        self.encoder = nn.GRU(
            input_size,
            enc_hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.enc_to_embed = nn.Linear(enc_hidden_size * 2, embedding_dim)

    def forward(self, src):
        """
        Extract embeddings for each agent's trajectory in the batch.

        Args:
            src: Tensor of shape [batch_size, seq_len, num_agents * input_size]
                Input past trajectories concatenated along feature dimension for all agents.

        Returns:
            embeddings: Tensor of shape [batch_size, num_agents, embed_dim]
                        Learned embeddings representing each agent's past trajectory.
        """
        batch_size, lookback, total_features = src.size()
        num_agents = total_features // self.input_size

        # Reshape to [batch_size * num_agents, lookback, input_size]
        src = src.view(batch_size, lookback, num_agents, self.input_size)
        src = src.permute(0, 2, 1, 3)  # [batch_size, num_agents, lookback, input_size]
        src = src.reshape(batch_size * num_agents, lookback, self.input_size)

        # hidden_shape: [num_layers * 2, batch_size * num_agents, enc_hidden_size]
        _, hidden = self.encoder(src)

        # Concatenate last layer's forward and backward hidden states
        hidden_cat = torch.cat(
            [hidden[-2], hidden[-1]], dim=1
        )  # [batch_size * num_agents, enc_hidden_size * 2]

        # Linear projection to embedding dimension
        emb = self.enc_to_embed(
            hidden_cat
        )  # [batch_size * num_agents, embedding_dim]

        # Reshape back to [batch_size, num_agents, embedding_dim]
        emb = emb.view(batch_size, num_agents, self.embedding_dim)

        return emb
