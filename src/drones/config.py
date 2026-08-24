from dataclasses import dataclass, field

import numpy as np

from src.config import BaseConfig

# Obs dim range meanings
# ang_vel 0:3 | quat 3:7 | lin_vel 7:10 | lin_pos 10:13 | prev_action 13:17 | aux 17:21
# pole_top_pos 21:24 | pole_bot_pos 24:27 | pole_top_vel 27:30 | pole_bot_vel 30:33
KEEP_A = (
    list(range(0, 7)) + list(range(10, 17)) + list(range(21, 27))
)  # Drop all velocities (except ang_vel, which is IMU readings and valid) and last actions

# attitude layout for quaternion angles
# ang_vel 0:3 | quaternion 3:7 | lin_vel 7:10 | lin_pos 10:13 | action 13:17 | aux 17:21
KEEP_B = list(range(0, 17))  # Keep lin_vel, fails without

# Dimension of vector of KIN for Task B
# len(KEEP_B) + our FlattenWaypointEnv context of 4 waypoints * 4 dim/waypoint
VECTOR_DIM = len(KEEP_B) + (4 * 4)  # = 33

# Different batch-size for RecurrentPPO for Task B
RPPO_TASKB_BS = {"episodelstm": 256}

DEFAULT_SEED = 1092026
EVAL_SEED = 28101995

def_fac = lambda x: field(default_factory=x)  # noqa


class FloorLRDecay:
    """
    LR decay with a minimum floor. Never anneals to 0
    """

    def __init__(self, initial: float, frac_floor: float = 0.1):
        self.initial = initial
        self.frac_floor = frac_floor

    def __call__(self, progress_remaining: float) -> float:
        current_frac = self.initial * progress_remaining
        floor = self.initial * self.frac_floor
        return max(current_frac, floor)

    def __repr__(self):
        return f"{self.__class__.__name__}([{self.initial}, {self.frac_floor}])"


@dataclass(frozen=True)
class Arm:
    name: str
    algo: str
    arch: str | None
    k_frames: int
    batch_size: int
    algo_speed: float  # relative to windowMLP, approx: set after running them. Purely used for queueing longest first for efficient resource use.


ARMS: dict[str, Arm] = {
    a.name: a
    for a in [
        # Pool (faster cos very few/no params)
        # -----------------------
        Arm("meanpool", "ppo", "meanpool", 8, 256, 1.09),
        Arm("cam", "ppo", "cam", 8, 256, 1.01),
        Arm("eca", "ppo", "eca", 8, 256, 1.02),
        Arm("se", "ppo", "se", 8, 256, 0.99),
        # Sequence over Window
        # ----------------------
        Arm("windowmlp", "ppo", "windowmlp", 8, 256, 1.0),
        Arm("transformer", "ppo", "transformer", 8, 256, 0.9),
        Arm("windowlstm", "ppo", "windowlstm", 8, 256, 0.85),  # estimated
        # K=1 no frame stack
        # --------------------------
        Arm("memoryless", "ppo", None, 1, 256, 1.07),
        Arm("episodelstm", "recurrentppo", None, 1, 2048, 0.86),
    ]
}


@dataclass
class DroneEnvConfig(BaseConfig):
    env_id: str = "PyFlyt/QuadX-Pole-Balance-v4"
    task: str = "pole-balance"
    axis2: str = "none"
    flight_mode: int = 0
    stripped: bool = True
    spread: float = 0.1
    norm_obs: bool = True
    norm_reward: bool = True  # Tested with False, gradients blow up
    keep_indices: list[int] = def_fac(lambda: list(KEEP_A))
    max_duration_seconds: float = 20.0  # in seconds
    n_obstacles: int = 0  # task a unaffected and honest


@dataclass
class DroneAlgoConfig(BaseConfig):
    d: int = 64
    net_arch_pi: list[int] = def_fac(lambda: [64, 64])
    net_arch_vf: list[int] = def_fac(lambda: [64, 64])

    learning_rate: float = 3e-4
    lr_floor_frac: float = 0.1
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None  # Dont want early stopping

    # recurrentppo only
    lstm_hidden_size: int = 64  # to ensure parameter matching with lstm window
    shared_lstm: bool = True  # one LSTM for actor + critic
    enable_critic_lstm: bool = (
        False  # Needs to be False for the shared_lstm to actually happen
    )


@dataclass
class DroneRunConfig(BaseConfig):
    master_seed: int = DEFAULT_SEED
    eval_seed: int = EVAL_SEED
    step_budget: int = 5_000_000
    n_runs: int = 10
    n_envs: int = 8  # values for running on 32-core RHUL CPU Compute VM
    n_concurrent: int = 24  # values for running on 32-core RHUL CPU Compute VM
    device: str = "cpu"

    checkpoint_every: int = 100_000  # linear checkpoints for chkpt/resume schedule to save the entire model + optimizer states
    keep_checkpoints: int = 2  # how many rolling checkpoints
    n_snapshots: int = 20  # for eval
    snapshot_start: int = 100_000  # log-spaced snapshots of just model state_dict for eval (learning from 0-100k negligible)

    wandb_project: str = "msc-diss-pyflyt-archs"
    wandb_mode: str = "online"

    eval_episodes: int = 20
    eval_n_jobs: int = 14


@dataclass
class DroneConfig(BaseConfig):
    run_config: DroneRunConfig = def_fac(DroneRunConfig)
    env_config: DroneEnvConfig = def_fac(DroneEnvConfig)
    algo_config: DroneAlgoConfig = def_fac(DroneAlgoConfig)


def snapshot_steps(cfg: DroneRunConfig) -> list[int]:
    """
    log-spaced points from snapshot_start to step_budget, inclusive.
    """

    pts = np.geomspace(cfg.snapshot_start, cfg.step_budget, cfg.n_snapshots)
    return sorted(set(int(round(p)) for p in pts))


def checkpoint_steps(cfg: DroneRunConfig) -> list[int]:
    """
    Linear, every checkpoint_every steps.
    """
    return list(range(cfg.checkpoint_every, cfg.step_budget + 1, cfg.checkpoint_every))


def run_key(arm: Arm, axis2: str = "none") -> str:
    """
    <algo>-<arch>-<axis>. Keep fixed.
    """
    return f"{arm.algo}-{arm.name}-{axis2}"


if __name__ == "__main__":
    fd = FloorLRDecay(1, 0.1)

    print(fd)

    for i in range(10, -1, -1):
        print(fd(i / 10))
