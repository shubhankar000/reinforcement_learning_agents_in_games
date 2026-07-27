import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from tqdm.auto import tqdm

from src.deep.envs import make_env
from src.rng_factory import SeededRNG

# Reference V values from literature
REFERENCE = {
    "CartPole-v1": {"v_random": 22.0, "v_star": 500.0},
    "LunarLander-v3": {"v_random": -180.0, "v_star": 200.0},
    "CarRacing-v3": {"v_random": -50.0, "v_star": 900.0},
}

ENVS = {
    "CartPole-v1": {"is_success": lambda term, trunc, r, G, L: L >= 500},
    "LunarLander-v3": {"is_success": lambda term, trunc, r, G, L: G > 200},
    "CarRacing-v3": {"is_success": lambda term, trunc, r, G, L: G > 900},
}
M = 20


def evaluate(exp_dir: Path, eval_master_seed: int = 67):
    meta = json.loads((exp_dir / "meta.json").read_text())
    env_id = meta["env_config"]["env_id"]
    env_kwargs = meta["env_config"]["env_kwargs"]
    gamma = meta["algo_config"]["gamma"]
    ref, cfg = REFERENCE[env_id], ENVS[env_id]

    policy = meta["algo_config"]["policy"]
    device = meta["run_config"]["device"]
    all_evals = []
    for dir_ in tqdm(sorted(exp_dir.glob("run_*")), desc="Evaluating 1 seed"):
        blob = torch.load(dir_ / "snapshots.pt", weights_only=False)
        env = make_env(env_id, env_kwargs)
        cap = env.spec.max_episode_steps or 1000

        shell = DQN(
            policy,
            env,
            policy_kwargs={"net_arch": blob["net_arch"]},
            device=device,
            buffer_size=1,
        )

        rows = []
        for sd, step in zip(blob["snapshots"], blob["steps"]):
            shell.q_net.load_state_dict(sd)
            shell.policy.set_training_mode(False)

            eval_rng = SeededRNG(eval_master_seed)
            returns, succ, lens, qpreds, mcs = [], [], [], [], []

            for _ in range(M):
                obs, _ = env.reset(seed=eval_rng.next_seed())
                obs_t, _ = shell.policy.obs_to_tensor(obs)
                with torch.no_grad():
                    qpreds.append(float(shell.q_net(obs_t).max()))
                done, G_undisc, G_disc, disc, L = False, 0, 0, 1, 0
                while not done and L < cap:
                    act, _ = shell.predict(obs, deterministic=True)
                    obs, r, term, trunc, _ = env.step(int(act))
                    done = term or trunc
                    G_undisc += r
                    G_disc += disc * r
                    disc *= gamma
                    L += 1

                returns.append(G_undisc)
                lens.append(L)
                mcs.append(G_disc)
                succ.append(cfg["is_success"](term, trunc, r, G_undisc, L))

            rows.append(
                {
                    "run_no.": dir_.name,
                    "env_steps": step,
                    "eval_return_mean": float(np.mean(returns)),
                    "eval_return_std": float(np.std(returns)),
                    "success_rate": float(np.mean(succ)),
                    "episode_len_mean": float(np.mean(lens)),
                    "v_star_s0": ref["v_star"],
                    "v_random_s0": ref["v_random"],
                    "q_pred_mean": float(np.mean(qpreds)),
                    "mc_return_mean": float(np.mean(mcs)),
                }
            )

        eval_df = pd.DataFrame(rows)
        all_evals.append(eval_df)
        eval_df.to_parquet(dir_ / "eval_log.parquet", index=False)
        env.close()

    pd.concat(all_evals).reset_index(drop=True).to_parquet(
        exp_dir / "all_evals.parquet", index=False
    )
