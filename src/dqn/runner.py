import json
from datetime import datetime
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import DQN

from src.rng_factory import SeededRNG
from src.tabular.q_learner import create_checkpoint_steps
from src.dqn.envs import make_vecenv
from src.dqn.callbacks import SnapshotCallback


def run_one_seed(env_id, env_kwargs, run_cfg, algo_cfg, seed, out, n_states, n_actions):
    checkpoints = create_checkpoint_steps(
        run_cfg.step_budget, run_cfg.checkpoint_type, run_cfg.checkpoint_points, 10
    )
    venv = make_vecenv(env_id, run_cfg.n_envs, env_kwargs)

    algo = algo_cfg.to_dict()
    policy = algo.pop("policy")
    model = DQN(policy, venv, seed=int(seed), device=run_cfg.device, verbose=0, **algo)

    cb = SnapshotCallback(checkpoints, n_states, n_actions)
    model.learn(total_timesteps=run_cfg.step_budget, callback=cb, progress_bar=False)

    np.savez(
        out / "snapshots.npz",
        snapshots=np.array(cb.snapshots),
        steps=np.array(checkpoints),
        visitation=cb.visitation,
    )
    pd.DataFrame(cb.rows).to_parquet(out / "train_log.parquet")


def run_experiment(env_id, env_kwargs, run_cfg, algo_cfg, out=Path("./runs")):
    slippery = env_kwargs.get("is_slippery", False) or env_kwargs.get("is_rainy", False)
    ts = datetime.now().strftime(
        f"%Y-%m-%d-%I-%M-%S-%p-{'slip' if slippery else 'det'}"
    )
    exp_dir = out / "dqn" / env_id / ts
    exp_dir.mkdir(parents=True, exist_ok=True)

    probe = gym.make(env_id, **env_kwargs)
    n_states, n_actions = int(probe.observation_space.n), int(probe.action_space.n)

    meta = {
        "run_config": run_cfg.to_dict(),
        "algo_config": algo_cfg.to_dict(),
        "env_config": {
            "env_id": env_id,
            "env_kwargs": env_kwargs,
            "obs_space": n_states,
            "action_space": n_actions,
        },
        "versions": {
            p: version(p)
            for p in (
                "stable_baselines3",
                "torch",
                "numpy",
                "gymnasium",
            )
        },
    }
    (exp_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    rng = SeededRNG(run_cfg.master_seed)
    seeds = [rng.next_seed() for _ in range(run_cfg.n_runs)]

    for i, s in enumerate(seeds):
        run_dir = exp_dir / f"run_{i:02d}"
        run_dir.mkdir(exist_ok=True)
        run_one_seed(
            env_id, env_kwargs, run_cfg, algo_cfg, s, run_dir, n_states, n_actions
        )

    return exp_dir
