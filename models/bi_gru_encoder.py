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
        self, input_size=6, enc_hidden_size=64, dec_hidden_size=64, num_layers=1
    ):
        super().__init__()
        self.input_size = input_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers

        # Shared modules for all agents
        self.encoder = nn.GRU(
            input_size,
            enc_hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.enc_to_dec = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, src, return_embedding=True):
        """
        Extract embeddings for each agent's trajectory in the batch.

        Args:
            src: Tensor of shape [batch_size, seq_len, num_agents * input_size]
                Input past trajectories concatenated along feature dimension for all agents.

        Returns:
            embeddings: Tensor of shape [batch_size, num_agents, embed_dim]
                        Learned embeddings representing each agent's past trajectory.
        """
        _, _, total_features = src.size()
        num_agents = total_features // self.input_size
        src_agents = torch.split(src, self.input_size, dim=2)

        if return_embedding:
            embeddings = []
            for agent_idx in range(num_agents):
                _, hidden = self.encoder(src_agents[agent_idx])
                # Combine bidirectional hidden states
                hidden_cat = torch.cat(
                    [hidden[-2], hidden[-1]], dim=1
                )  # [B, 2*enc_hidden_size]
                emb = self.enc_to_dec(
                    hidden_cat
                )  # Project to dec_hidden_size or embed_dim
                embeddings.append(emb.unsqueeze(1))
            embeddings = torch.cat(embeddings, dim=1)  # [B, num_agents, embed_dim]
            return embeddings
