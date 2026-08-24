"""
Parallelised evaluation for Task A and Task B
"""

import os

# parallel eval, set to 1 to prevent thread thrashing
os.environ.setdefault("OMP_NUM_THREADS", "1")

import dataclasses
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from stable_baselines3.common.vec_env import VecNormalize
from tqdm.auto import tqdm

from src.drones import run_one, run_one_b
from src.drones.config import (
    ARMS,
    Arm,
    DroneAlgoConfig,
    DroneConfig,
    DroneEnvConfig,
    DroneRunConfig,
)
from src.drones.envs import make_vecenv, make_vecenv_b
from src.rng_factory import SeededRNG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANCHORS_ROOT = PROJECT_ROOT / "runs" / "drones" / "anchors"

HOVER_TARGET = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Task:
    name: str
    make_venv: Callable
    build_model: Callable
    is_success: Callable
    step_metrics: Callable
    episode_metrics: Callable


def reconstruct_dataclass(cls, d: dict):
    """
    Build a dataclass from a stored dict.
    """
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def config_from_meta(meta: dict) -> DroneConfig:
    cfg = meta["config"]

    return DroneConfig(
        run_config=reconstruct_dataclass(DroneRunConfig, cfg["run_config"]),
        env_config=reconstruct_dataclass(DroneEnvConfig, cfg["env_config"]),
        algo_config=reconstruct_dataclass(DroneAlgoConfig, cfg["algo_config"]),
    )


def load_anchors(task_name: str):
    return json.loads((ANCHORS_ROOT / f"{task_name}.json").read_text())


def get_unnorm_frame(venv: VecNormalize):
    obs = venv.get_original_obs()[0]
    return obs[-1] if obs.ndim == 2 else obs


def venv_a(arm: Arm, cfg: DroneConfig, seed: int) -> VecNormalize:
    venv = make_vecenv(
        seed,
        n_envs=1,
        use_subproc=False,
        stripped=cfg.env_config.stripped,
        flight_mode=cfg.env_config.flight_mode,
        k_frames=arm.k_frames,
        spread=cfg.env_config.spread,
    )
    venv = VecNormalize(
        venv,
        norm_obs=cfg.env_config.norm_obs,
        norm_reward=False,
        gamma=cfg.algo_config.gamma,
    )
    venv.training = False
    return venv


def venv_b(arm: Arm, cfg: DroneConfig, seed: int) -> VecNormalize:
    venv = make_vecenv_b(
        seed,
        n_envs=1,
        use_subproc=False,
        n_obstacles=cfg.env_config.n_obstacles,
        k_frames=arm.k_frames,
        flight_mode=cfg.env_config.flight_mode,
        max_duration=cfg.env_config.max_duration_seconds,
    )
    venv = VecNormalize(
        venv,
        norm_obs=cfg.env_config.norm_obs,
        norm_reward=False,
        gamma=cfg.algo_config.gamma,
        norm_obs_keys=["vector"],  # keep pixels unnormlised
    )
    venv.training = False
    return venv


def steps_a(venv: VecNormalize):
    """
    KEEP_A slices of obs
    """
    obs = get_unnorm_frame(venv)
    lin_pos, top, bot = obs[7:10], obs[14:17], obs[17:20]

    return {
        "leaningness": float(np.linalg.norm(top[:2] - bot[:2])),
        "hover_error": float(np.linalg.norm(lin_pos - HOVER_TARGET)),
    }


def steps_b(venv: VecNormalize):
    return {}


def episode_a(info: dict):
    return {}


def episode_b(info: dict):
    obstacle = bool(info.get("obstacle_collision", False))
    collision = bool(info.get("collision", False))
    complete = bool(info.get("env_complete", False))
    oob = bool(info.get("out_of_bounds", False))

    return {
        "targets_reached": float(info.get("num_targets_reached", 0)),
        "term_obstacle_collision": float(obstacle),
        "term_floor_collision": float(collision and not obstacle),
        "term_env_complete": float(complete),
        "term_timeout": float(not complete and not collision and not oob),
    }


TASKS: dict[str, Task] = {
    "pole-balance": Task(
        "pole-balance",
        venv_a,
        run_one.build_model,
        is_success=lambda info: bool(info.get("TimeLimit.truncated", False)),
        step_metrics=steps_a,
        episode_metrics=episode_a,
    ),
    "waypoint": Task(
        "waypoint",
        venv_b,
        run_one_b.build_model,
        is_success=lambda info: bool(info.get("env_complete", False)),
        step_metrics=steps_b,
        episode_metrics=episode_b,
    ),
}


def rollout(model, venv: VecNormalize, seed: int, gamma: float, task: Task):
    """
    Rollout one episode
    """
    venv.seed(seed)
    obs = venv.reset()

    state = None
    episode_start = np.ones(1, dtype=bool)
    r_undiscounted = 0
    r_discounted = 0
    discount = 1
    length = 0

    step_acc = defaultdict(list)

    while length < 1000:  # no episode is above 1000
        action, state = model.predict(
            obs, state=state, episode_start=episode_start, deterministic=True
        )
        episode_start = np.zeros(1, dtype=bool)

        obs, rewards, dones, infos = venv.step(action)
        reward = float(rewards[0])

        r_undiscounted += reward
        r_discounted += discount * reward
        discount *= gamma
        length += 1

        if not dones[0]:
            for k, v in task.step_metrics(venv).items():
                step_acc[k].append(v)
        else:
            info = infos[0]
            break

    row = {
        "return": r_undiscounted,
        "mc_return": r_discounted,
        "len": length,
        "success": float(task.is_success(info)),
        "term_collision": float(bool(info.get("collision", False))),
        "term_oob": float(bool(info.get("out_of_bounds", False))),
        **task.episode_metrics(info),
        **{f"{k}_mean": float(np.mean(v)) for k, v in step_acc.items()},
    }

    return row


def evaluate_snapshot(model, venv, blob: dict, task: Task, cfg: DroneConfig):
    venv.obs_rms = blob["obs_rms"]
    model.policy.load_state_dict(blob["policy"], strict=True)
    model.policy.set_training_mode(False)

    rng = SeededRNG(cfg.run_config.eval_seed)

    with torch.no_grad():
        rows = []
        for _ in range(cfg.run_config.eval_episodes):
            r = rollout(model, venv, rng.next_seed(), cfg.algo_config.gamma, task)
            rows.append(r)

    df = pd.DataFrame(rows)

    out = {
        "env_steps": blob["step"],
        "n_episodes": len(rows),
        "eval_return_mean": float(df["return"].mean()),
        "eval_return_std": float(df["return"].std(ddof=0)),
        "mc_return_mean": float(df["mc_return"].mean()),
        "episode_len_mean": float(df["len"].mean()),
        "episode_len_std": float(df["len"].std(ddof=0)),
        "success_rate": float(df["success"].mean()),
    }
    for col in df.columns:
        if col not in ("return", "mc_return", "len", "success"):
            out[col] = float(df[col].mean())
    return out


def evaluate_run(run_dir: Path) -> pd.DataFrame:
    out_path = run_dir / "eval_log.parquet"

    meta = json.loads((run_dir / "meta.json").read_text())
    cfg = config_from_meta(meta)
    cfg.algo_config.n_steps = 2

    arm = ARMS[meta["arm"]]
    task = TASKS[cfg.env_config.task]
    anchors = load_anchors(task.name)

    venv = task.make_venv(arm, cfg, meta["seed"])

    try:
        model = task.build_model(arm, cfg, venv, meta["seed"])
        rows = []
        for snap in sorted((run_dir / "snapshots").glob("step_*.pt")):
            blob = torch.load(
                snap, weights_only=False, map_location=cfg.run_config.device
            )
            row = evaluate_snapshot(model, venv, blob, task, cfg)
            row.update(
                {
                    "run_key": meta["run_key"],
                    "run_no.": run_dir.name,
                    "arm": meta["arm"],
                    "seed": meta["seed"],
                    "v_star_s0": anchors["v_star_s0"],
                    "v_random_s0": anchors["v_random_s0"],
                }
            )
            rows.append(row)

    finally:
        venv.close()

    df = pd.DataFrame(rows).sort_values("env_steps").reset_index(drop=True)
    df.to_parquet(out_path, index=False)

    return df


def evaluate(task_dir: Path, n_jobs: int):
    run_dirs = sorted(d for d in task_dir.glob("*/run_*"))

    results = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(evaluate_run)(d) for d in run_dirs
    )

    frames = list(tqdm(results, total=len(run_dirs), desc=f"eval {task_dir.name}"))

    per_arm = defaultdict(list)
    for run_dir, df in zip(run_dirs, frames):
        per_arm[run_dir.parent].append(df)

    for arm_dir, dfs in per_arm.items():
        merged = pd.concat(dfs).reset_index(drop=True)
        merged.to_parquet(arm_dir / "all_evals.parquet", index=False)
        print(f"{arm_dir.name} {len(dfs)} runs {len(merged)} rows")


def main():
    for task_name in ("pole-balance", "waypoint"):
        task_dir = PROJECT_ROOT / "runs" / "drones" / task_name

        print(f"  [TASK {task_name}]")

        evaluate(task_dir, DroneRunConfig().eval_n_jobs)


if __name__ == "__main__":
    main()
