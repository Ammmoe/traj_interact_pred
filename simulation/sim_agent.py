import numpy as np


class SimAgent:
    SAMPLING_FREQ = 30
    TRAJ_DURATION = 2  # seconds
    TRAJ_LEN = TRAJ_DURATION * SAMPLING_FREQ  # -> 60 frames
    N_FEATURES = 6  # (x, y, z, vx, vy, vz)
    PRED_INTERVAL = 1  # Hz

    def __init__(self, my_id):
        self.my_id = my_id

        # agent_list: numpy array (N, 2) -> (agent_id:int, role:str)
        # Example: [[0, "friendly"], [3, "unauthorized"], ...]
        self.agent_list = np.empty((0, 2), dtype=object)

        # other_agents_traj: numpy array (N, TRAJ_LEN=60, N_FEATURES=6)
        self.other_agents_traj = np.empty((0, self.TRAJ_LEN, self.N_FEATURES))

        # interact_pair: list of (friendly_id, unauthorized_id, follow_flag)
        self.interact_pair = []

    def sim_traj_tracker(self, my_id):
        """
        Input:
            my_id (int): ID of this agent

        Output:
            agent_dict (dict):
                {
                    agent_id_1: "friendly",
                    agent_id_2: "unauthorized",
                    ...
                }

            other_agents_traj (np.ndarray):
                Trajectories of all agents except my_id.
                Shape: (N_OTHER_AGENTS, TRAJ_LEN=60, N_FEATURES=6)

        Description:
            Build agent metadata dictionary and collect all trajectory
            data from simulation or dataset. Excludes self.my_id.
            Called by interact_predict() at 1 Hz.
        """
        pass

    def interact_predict(self, other_agents_traj):
        """
        Input:
            other_agents_traj (np.ndarray):
                Shape: (N_OTHER_AGENTS, TRAJ_LEN, N_FEATURES)

        Output:
            interact_pair (list):
                List of (friendly_id, unauthorized_id, follow_flag)

        Description:
            Predict interaction pairs between friendly and unauthorized agents.
            Called by timer at PRED_INTERVAL = 1 Hz.
        """
        pass

    def traj_tracker(self, agent_list):
        """
        Input:
            my_id (int): ID of this agent

        Output:
            agent_dict (dict):
                {
                    agent_id_1: "friendly",
                    agent_id_2: "unauthorized",
                    ...
                }

            other_agents_traj (np.ndarray):
                Trajectories of all agents except my_id.
                Shape: (N_OTHER_AGENTS, TRAJ_LEN=60, N_FEATURES=6)

        Description:
            Pull trajectory frames from ROS /sim_nwu_pose.
            This updates the latest 60-frame window for each agent.
            Called by interact_predict() at 1 Hz.
        """
        pass


def main():
    my_agent = SimAgent(my_id=0)


if __name__ == "__main__":
    main()
