"""
Render champion videos in human mode for viewing and screen capture.
Works differently to figures.py in tabular and deep, because rendering rollouts through PyFlyt would render in egocentric camera view. For proper viewing, we want world-frame view.
Dont create gifs for drones, it would be too long and too big on disk.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.drones.config import ARMS, EVAL_SEED, Arm, DroneConfig, run_key
from src.drones.envs import apply_motor_noise_fix, make_env, make_env_b
from src.drones.eval import TASKS, config_from_meta
from src.rng_factory import SeededRNG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRONES_ROOT = PROJECT_ROOT / "runs" / "drones"

N_VIDEOS = 5
FPS = 30
MAX_STEPS = 1000

CHAMPION_METRIC = {"pole-balance": "eval_return_mean", "waypoint": "success_rate"}


def find_champion(task_dir: Path) -> Path:
    """
    Find which run dir is the champion based on the final snapshot
    """
    metric = CHAMPION_METRIC[task_dir.name]
    best = None
    best_score = -np.inf

    for arm in ARMS.values():
        arm_dir = task_dir / run_key(arm, "none")

        df = pd.read_parquet(arm_dir / "all_evals.parquet")
        last = df[df["env_steps"] == df["env_steps"].max()]
        row = last.loc[last[metric].idxmax()]

        if row[metric] > best_score:
            best = arm_dir / row["run_no."]
            best_score = row[metric]

    print(
        f"{task_dir.name}: champion {best.parent.name}/{best.name} ({metric}={best_score:.3f})"
    )
    return best


def create_venv(
    arm: Arm, cfg: DroneConfig, seed: int, render_mode=None
) -> VecNormalize:
    """
    Wrap drone env with human rendering, exactly how eval wraps it
    """
    apply_motor_noise_fix()

    if cfg.env_config.task == "pole-balance":
        env = make_env(
            seed,
            stripped=cfg.env_config.stripped,
            flight_mode=cfg.env_config.flight_mode,
            k_frames=arm.k_frames,
            spread=cfg.env_config.spread,
            render_mode=render_mode,
        )
        keys = None
    else:
        env = make_env_b(
            seed,
            n_obstacles=cfg.env_config.n_obstacles,
            k_frames=arm.k_frames,
            flight_mode=cfg.env_config.flight_mode,
            max_duration=cfg.env_config.max_duration_seconds,
            render_mode=render_mode,
        )
        keys = ["vector"]

    venv = VecNormalize(
        DummyVecEnv([env]),
        norm_obs=cfg.env_config.norm_obs,
        norm_reward=False,
        gamma=cfg.algo_config.gamma,
        norm_obs_keys=keys,
    )
    venv.training = False

    return venv


def render_one(model: PPO, venv: VecNormalize, seed: int):
    """
    human render 1 episode of champion flying. to be used with screen capture (to caputure the world frame, only possible through human rendering).
    """

    venv.seed(seed)
    obs = venv.reset()

    input("    Press enter to play out scene")

    state = None
    episode_start = np.ones(1, dtype=bool)
    length = 0

    while length < MAX_STEPS:
        action, state = model.predict(
            obs, state=state, episode_start=episode_start, deterministic=True
        )
        episode_start = np.zeros(1, dtype=bool)

        obs, _, dones, infos = venv.step(action)
        length += 1
        time.sleep(1 / FPS)

        if dones[0]:
            return infos[0]

    return {}


def render_task(task_name: str, n_videos: int = N_VIDEOS):
    task_dir = DRONES_ROOT / task_name
    champ_dir = find_champion(task_dir)

    meta = json.loads((champ_dir / "meta.json").read_text())
    cfg = config_from_meta(meta)
    cfg.algo_config.n_steps = 2
    arm = ARMS[meta["arm"]]
    task = TASKS[cfg.env_config.task]

    # Need to create a headless env first, Then load sb3 model Then load the render GUI
    # Otherwise it just crashes
    headless = create_venv(arm, cfg, meta["seed"], render_mode=None)

    try:
        model: PPO = task.build_model(arm, cfg, headless, meta["seed"])
        weights = torch.load(
            champ_dir / "final.pt", weights_only=False, map_location="cpu"
        )  # Run champion on cpu only
        model.policy.load_state_dict(weights["policy"], strict=True)
        model.policy.set_training_mode(False)

        venv = create_venv(arm, cfg, meta["seed"], render_mode="human")
        venv.obs_rms = weights["obs_rms"]
        venv.norm_reward = False
        model.set_env(venv)
        headless.close()

        rng = SeededRNG(EVAL_SEED)
        with torch.no_grad():
            for i in range(n_videos):
                info = render_one(model, venv, rng.next_seed())
                outcome = "success" if task.is_success(info) else "fail"
                print(f"  clip {i}: {outcome}")
                input("  Press Enter for the next clip")
    finally:
        venv.close()


def main():
    for task_name in ("pole-balance", "waypoint"):
        render_task(task_name)


if __name__ == "__main__":
    main()
