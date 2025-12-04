"""
Module for simulating an agent that predicts interaction pairs between friendly and unauthorized
agents based on trajectory data and agent roles.
"""

import torch
from traj_interact_predict.data.data_loader import load_datasets
from traj_interact_predict.utils.sim_agent_utils import filter_my_id
from scripts.inference import run_inference


class SimAgent:
    """
    SimAgent simulates an agent that tracks other agents' trajectories and predicts interaction
    pairs.

    Attributes:
        my_id (int): The agent's own ID.
        other_agents_ids (list): List of other agents' IDs after filtering out self.
        other_agents_traj (np.ndarray): Trajectories of other agents.
    """

    def __init__(self, my_id):
        """Initialize SimAgent with its own ID."""
        # Save my_id and other_agents_ids
        self.my_id = my_id
        self.other_agents_ids = []

    def sim_traj_tracker(self):
        """
        Collects and filters trajectory and role data excluding self.my_id,
        returning a mapping of agent IDs to roles and trajectories of other agents.

        Returns:
            agent_role (dict): Mapping from agent ID to role string ("friendly" or "unauthorized").
            other_agents_traj (np.ndarray): Trajectories of other agents with shape
            (TRAJ_LEN, N_OTHER_AGENTS, N_FEATURES).
        """
        # Get one random sample from the dataset
        one_random_sample = load_datasets(
            trajectory_csv="data/drone_states.csv",
            relation_csv="data/drone_relations.csv",
            lookback=60,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            max_agents=6,
            num_friendly_to_pad=0,
            num_unauth_to_pad=0,
            return_one_sample=True,
        )

        # Extract the information from one_random_sample
        (
            trajectories,  # shape: (lookback=60, num_agents, 6)
            roles_tensor,  # shape: (num_agents,)
            agent_mask,  # shape: (num_agents,)
            pairs,
            labels,
        ) = one_random_sample

        # Filter the information without my_id agent
        (
            other_agents_traj,
            filtered_roles,
            _,
            filtered_pairs,
            filtered_labels,
            filtered_ids,
        ) = filter_my_id(
            trajectories=trajectories,
            roles_tensor=roles_tensor,
            agent_mask=agent_mask,
            pairs=pairs,
            labels=labels,
            my_id=self.my_id,
        )

        # Save the actual agent IDs (critical)
        self.other_agents_ids = filtered_ids.cpu().numpy().tolist()

        # Map role integers to role strings
        role_map = {0: "friendly", 1: "unauthorized"}

        # Create agent_role
        agent_role = {
            int(filtered_ids[i]): role_map[int(filtered_roles[i])]
            for i in range(len(filtered_roles))
        }

        # Print self agent info from unfiltered roles
        try:
            my_role_int = roles_tensor[self.my_id].item()
        except IndexError as e:
            raise IndexError(
                f"self.my_id={self.my_id} is out of bounds for "
                f"roles_tensor with length {len(roles_tensor)}"
            ) from e
        my_role_str = role_map.get(my_role_int, "unknown")

        print(f"My Agent ID: {self.my_id}, Role: {my_role_str}")
        print("\nAgent roles after filtering (agent_id: role):")
        for aid, role in agent_role.items():
            print(f"  Agent ID {aid}: {role}")

        print("=== Original data before filtering ===")

        # Original indices of agents
        num_agents = trajectories.shape[1]
        original_ids = list(range(num_agents))
        print(f"Original agent indices: {original_ids}")

        # Roles for all agents
        role_map = {0: "friendly", 1: "unauthorized"}
        roles_list = roles_tensor.tolist()
        print("Original agent roles (index: role):")
        for i, r in enumerate(roles_list):
            role_str = role_map.get(r, "unknown")
            print(f"  Index {i}: {role_str}")

        # Pairs before filtering
        print("\nOriginal pairs (agent indices):")
        print(pairs.cpu().tolist())

        print("\nOriginal labels for pairs:")
        print(labels.cpu().tolist())

        print("=" * 35)

        # Create list of filtered roles as ints
        filtered_roles_list = [int(r) for r in filtered_roles]

        # Find indices of friendly and unauthorized agents in filtered list
        friendly_idx = [i for i, r in enumerate(filtered_roles_list) if r == 0]
        unauth_idx = [i for i, r in enumerate(filtered_roles_list) if r == 1]

        # Construct pairs list same as prediction order
        sorted_pairs = [(f, u) for f in friendly_idx for u in unauth_idx]

        # Print actual labels in this sorted order
        # Create a dictionary mapping pair tuples to labels for quick lookup
        pair_to_label = {
            (filtered_pairs[i, 0].item(), filtered_pairs[i, 1].item()): filtered_labels[
                i
            ].item()
            for i in range(len(filtered_labels))
        }

        print(
            "\nActual labels sorted by pairs (friendly_id, unauthorized_id, follow_flag):"
        )
        for f, u in sorted_pairs:
            f_id = filtered_ids[f].item()
            u_id = filtered_ids[u].item()
            label = pair_to_label.get((f, u), None)  # safely get label
            print(
                f"  Friendly Agent {f_id} <-> Unauthorized Agent {u_id}, Label: {label}"
            )

        return agent_role, other_agents_traj.cpu().numpy()

    def interact_predict(self, agent_role, other_agents_traj):
        """
        Predicts interaction pairs between friendly and unauthorized agents
        based on their trajectories and roles.

        Args:
            agent_role (dict): Mapping of agent IDs to role strings.
            other_agents_traj (np.ndarray): Trajectories of other agents,
                shape (LOOKBACK, N_OTHER_AGENTS, N_FEATURES).

        Returns:
            interact_pair (list): List of tuples
                (friendly_agent_id, unauthorized_agent_id, follow_flag).
        """
        # Convert trajectory shape and data type
        trajectories = torch.tensor(other_agents_traj, dtype=torch.float32).unsqueeze(
            0
        )  # (1, lookback, num_agents, features)

        # Define sorted agent ID order
        sorted_ids = sorted(self.other_agents_ids)

        # Compute index order mapping
        idx_order = [self.other_agents_ids.index(aid) for aid in sorted_ids]

        # Reorder trajectories along agent dimension (dim=2)
        trajectories = trajectories[:, :, idx_order, :]

        # Reorder other_agents_ids
        self.other_agents_ids = sorted_ids

        # Build roles from reordered IDs
        role_map_reverse = {"friendly": 0, "unauthorized": 1}
        roles = torch.tensor(
            [
                role_map_reverse[agent_role[agent_id]]
                for agent_id in self.other_agents_ids
            ],
            dtype=torch.long,
        ).unsqueeze(0)  # (1, num_agents)

        # Create agent mask
        agent_mask = torch.ones(len(self.other_agents_ids), dtype=torch.bool).unsqueeze(
            0
        )  # (1, num_agents)

        # Agent pairs
        roles_list = roles.squeeze(0).tolist()
        friendly_idx = [idx for idx, role in enumerate(roles_list) if role == 0]
        unauth_idx = [idx for idx, role in enumerate(roles_list) if role == 1]
        pairs_list = [(f, u) for f in friendly_idx for u in unauth_idx]

        if pairs_list:
            pairs = torch.tensor(pairs_list, dtype=torch.long).unsqueeze(
                0
            )  # (1, num_pairs, 2)
        else:
            pairs = torch.empty((1, 0, 2), dtype=torch.long)

        # Run inference
        logits = run_inference(trajectories, roles, pairs, agent_mask)

        # Apply sigmoid and threshold
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        # Map pairs back to actual agent IDs
        interact_pair = []
        for i, (friendly_idx, unauth_idx) in enumerate(pairs_list):
            friendly_id = self.other_agents_ids[friendly_idx]
            unauth_id = self.other_agents_ids[unauth_idx]
            follow_flag = preds[i].item()
            interact_pair.append((friendly_id, unauth_id, follow_flag))

        return interact_pair


def main():
    """
    Main function to create a SimAgent, track trajectories, and predict interactions.
    """
    my_agent = SimAgent(my_id=5)
    agent_role, other_agents_traj = my_agent.sim_traj_tracker()
    interact_pair = my_agent.interact_predict(agent_role, other_agents_traj)

    print("\nPredicted interaction pairs (friendly_id, unauthorized_id, follow_flag):")
    if interact_pair:
        for pair in interact_pair:
            print(
                "  Friendly Agent "
                f"{pair[0]} <-> Unauthorized Agent "
                f"{pair[1]}, Prediction: "
                f"{pair[2]}"
            )
    else:
        print("  No interaction pairs predicted.")


if __name__ == "__main__":
    main()
