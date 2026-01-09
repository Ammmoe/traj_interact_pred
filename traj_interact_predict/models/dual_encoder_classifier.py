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
        logits_all = []
        batch_size, lookback, _, feat_dim = batch_trajectories.shape

        for batch_i in range(batch_size):
            roles = batch_roles[batch_i]  # [max_agents]
            agent_mask = batch_agent_mask[batch_i]

            # Select agent indices by role: 0=friendly, 1=unauthorized
            friendly_ids = ((roles == 0) & agent_mask).nonzero(as_tuple=True)[0]
            unauth_ids = ((roles == 1) & agent_mask).nonzero(as_tuple=True)[0]

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

            ## Cross-agent attention layer to model interactions ##
            # Concatenate agents into one sequence (along agents dimension)
            all_emb = torch.cat(
                [emb_friendly_all, emb_unauth_all], dim=1
            )  # [1, num_friendly + num_unauth, embed_dim]

            # Add role embeddings
            friendly_roles = torch.zeros(
                (1, num_friendly), dtype=torch.long, device=all_emb.device
            )  # 0 for friendly
            unauth_roles = torch.ones(
                (1, num_unauth), dtype=torch.long, device=all_emb.device
            )  # 1 for unauthorized
            role_ids = torch.cat(
                [friendly_roles, unauth_roles], dim=1
            )  # [1, total_agents]

            role_embeddings = self.role_embedding(
                role_ids
            )  # [1, total_agents, embed_dim]
            all_emb = all_emb + role_embeddings  # [1, total_agents, embed_dim]

            # Self-attention across all agents
            attn_out, _ = self.cross_agent_attn(
                query=all_emb, key=all_emb, value=all_emb
            )  # [1, num_friendly + num_unauth, embed_dim]

            # Residual + normalization
            all_emb = self.attn_norm(all_emb + attn_out)

            # Split back into roles
            emb_friendly_all = all_emb[
                :, :num_friendly, :
            ]  # [1, num_friendly, embed_dim]
            emb_unauth_all = all_emb[:, num_friendly:, :]  # [1, num_unauth, embed_dim]

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
                # Skip pairs where either agent is masked out
                if not (agent_mask[friendly_id] and agent_mask[unauth_id]):
                    # If invalid agent in pair, skip
                    continue

            # Convert original agent ids to embedding indices for all pairs
            friendly_indices = torch.tensor(
                [friendly_id_map[f.item()] for f, _ in pairs],
                device=all_emb.device,
                dtype=torch.long,
            )
            unauth_indices = torch.tensor(
                [unauth_id_map[u.item()] for _, u in pairs],
                device=all_emb.device,
                dtype=torch.long,
            )

            # Gather embeddings for all pairs
            emb_friendly = emb_friendly_all[
                0, friendly_indices, :
            ]  # [num_pairs, embed_dim]
            emb_unauth = emb_unauth_all[0, unauth_indices, :]  # [num_pairs, embed_dim]

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
            )  # [num_pairs, 5 * embed_dim]

            # Normalize and classify
            relation_vector = self.layer_norm(relation_vector)
            batch_logits = self.classifier(relation_vector)  # [num_pairs, 1]

            logits_all.append(batch_logits)

        return logits_all
