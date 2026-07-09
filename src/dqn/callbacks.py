import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from src.tabular.q_learner import create_checkpoint_steps
from src.dqn.qtable import create_qtable


class SnapshotCallback(BaseCallback):
    """
    Create snapshots of the agent as a SB3 callback
    """

    def __init__(self, checkpoints, n_states, n_actions):
        super().__init__()
        self.checkpoints = checkpoints
        self.n_states, self.n_actions = n_states, n_actions
        self.next_idx = 0
        self.snapshots = []
        self.rows = []
        self.visitation = np.zeros((n_states, n_actions))
        self.ep_count = 0

    def _on_step(self) -> bool:
        self.ep_count += int(np.sum(self.locals["dones"]))
        while (
            self.next_idx < len(self.checkpoints)
        ) and self.num_timesteps >= self.checkpoints[self.next_idx]:
            self.snapshots.append(create_qtable(self.model, self.n_states))
            ep_rew = [e["r"] for e in self.model.ep_info_buffer] or [np.nan]
            self.rows.append(
                {
                    "env_steps": self.checkpoints[self.next_idx],
                    "episode": self.ep_count,
                    "mean_td_error": np.nan,
                    "epsilon": self.model.exploration_rate,
                    "train_return_mean": float(np.mean(ep_rew)),
                }
            )
            self.next_idx += 1

        return True
