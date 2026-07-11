import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from stable_baselines3 import DQN
from tqdm.auto import tqdm

from src.dqn.callbacks import SnapshotCallback
from src.dqn.config import DQNConfig
from src.dqn.envs import make_vecenv
from src.rng_factory import SeededRNG
from src.tabular.q_learner import create_checkpoint_steps


def seed_worker(cfg: DQNConfig, seed: int, out: Path):
    torch.set_num_threads(1)

    run_one_seed(cfg, seed, out)


def run_one_seed(cfg: DQNConfig, seed: int, out: Path):
    run_cfg = cfg.run_config
    algo_cfg = cfg.algo_config
    env_id = cfg.env_config.env_id
    env_kwargs = cfg.env_config.env_kwargs
    n_states = cfg.env_config.obs_space
    n_actions = cfg.env_config.action_space

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


def run_experiment(cfg: DQNConfig, out=Path("./runs")):
    env_config = cfg.env_config
    env_id, env_kwargs = env_config.env_id, env_config.env_kwargs
    slippery = env_kwargs.get("is_slippery", False) or env_kwargs.get("is_rainy", False)
    ts = datetime.now().strftime(
        f"%Y-%m-%d-%I-%M-%S-%p-{'slip' if slippery else 'det'}"
    )
    exp_dir = out / env_id / "dqn" / ts
    exp_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        **cfg.to_dict(),
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

    rng = SeededRNG(cfg.run_config.master_seed)
    seeds = [rng.next_seed() for _ in range(cfg.run_config.n_runs)]

    run_dirs = [exp_dir / f"run_{i:02d}" for i in range(cfg.run_config.n_runs)]
    for d in run_dirs:
        d.mkdir(exist_ok=True)

    delayed_args = []
    for seed, dir_ in zip(seeds, run_dirs):
        delayed_args.append(delayed(seed_worker)(cfg, seed, dir_))

    jobs = Parallel(n_jobs=cfg.run_config.n_jobs, return_as="generator")(delayed_args)

    run_cfg = cfg.run_config

    total_steps = run_cfg.n_runs * run_cfg.step_budget

    with tqdm(
        total=total_steps,
        unit="step",
        unit_scale=True,
        desc=f"{env_id} {'slip' if slippery else 'det'}",
    ) as pbar:
        for _ in jobs:
            pbar.update(
                run_cfg.step_budget
            )  # one finished seed = step_budget env steps

    return exp_dir
