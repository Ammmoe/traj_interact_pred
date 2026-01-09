"""
bi_gru_encoder.py

Bidirectional GRU encoder for extracting fixed-size embeddings from multi-agent trajectories.

Encodes each agent's past trajectory independently into learned embeddings suitable for
downstream tasks like relation prediction or classification.

Features:
    - Bidirectional GRU encoder
    - Handles variable agent counts per batch
    - Projects concatenated hidden states to fixed embedding size
"""

import torch
from torch import nn


class TrajEmbeddingExtractor(nn.Module):
    """
    Encodes multi-agent trajectories into fixed-size embeddings using a bidirectional GRU.

    Processes past trajectories per agent in parallel and outputs embeddings for downstream use.

    Args:
        input_size (int): Features per timestep per agent (e.g., 6 for position + velocity).
        enc_hidden_size (int): Hidden units in GRU encoder.
        embedding_dim (int): Output embedding dimension per agent.
        num_layers (int): Number of stacked GRU layers.
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
        Encode trajectories into embeddings per agent.

        Args:
            src (Tensor): [batch_size * num_agents, seq_len, input_size]

        Returns:
            Tensor: [batch_size * num_agents, embedding_dim]
        """
        _, hidden = self.encoder(src)

        # hidden shape: [num_layers * 2, batch_size * num_agents, enc_hidden_size]
        # Take last layer's forward and backward hidden states
        hidden_cat = torch.cat(
            [hidden[-2], hidden[-1]], dim=1
        )  # [batch_size * num_agents, enc_hidden_size * 2]

        emb = self.enc_to_embed(hidden_cat)  # [batch_size * num_agents, embedding_dim]
        return emb
