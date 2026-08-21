import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.deep.qtable import create_qtable


class SnapshotCallback(BaseCallback):
    """
    Create snapshots of the agent as a SB3 callback
    """

    def __init__(self, checkpoints, obs_type, n_states=None, n_actions=None):
        super().__init__()
        self.checkpoints = checkpoints
        self.n_states, self.n_actions = n_states, n_actions
        self.next_idx = 0
        self.snapshots = []
        self.rows = []

        if obs_type == "Discrete":
            self.visitation = np.zeros((n_states, n_actions))
        else:
            self.visitation = None
            self.obs_buffer = []
            self.obs_stride = 50
            self.next_obs_at = 0

        self.ep_count = 0
        self.last_loss = np.nan
        self.obs_type = obs_type

    def _on_step(self) -> bool:
        self.ep_count += int(np.sum(self.locals["dones"]))

        # discrete/box path for loss
        loss = self.logger.name_to_value.get("train/loss", np.nan)
        if not np.isnan(loss):
            self.last_loss = loss

        if self.obs_type == "Discrete":
            obs = np.ravel(self.model._last_obs)
            acts = np.ravel(self.locals["actions"])

            for o, a in zip(obs, acts):
                self.visitation[int(o), int(a)] += 1
        elif self.obs_type == "Box":
            if self.num_timesteps >= self.next_obs_at:
                self.obs_buffer.append(np.asarray(self.model._last_obs).copy())
                self.next_obs_at = self.num_timesteps + self.obs_stride

        while (
            self.next_idx < len(self.checkpoints)
        ) and self.num_timesteps >= self.checkpoints[self.next_idx]:
            if self.obs_type == "Discrete":
                self.snapshots.append(create_qtable(self.model, self.n_states))

            else:
                snap = {
                    k: v.cpu().clone() for k, v in self.model.q_net.state_dict().items()
                }
                self.snapshots.append(snap)

            ep_rew = [e["r"] for e in self.model.ep_info_buffer] or [np.nan]
            self.rows.append(
                {
                    "env_steps": self.checkpoints[self.next_idx],
                    "episode": self.ep_count,
                    "mean_td_error": float(self.last_loss),
                    "epsilon": self.model.exploration_rate,
                    "train_return_mean": float(np.mean(ep_rew)),
                }
            )
            self.next_idx += 1

        return True
