import json
from pathlib import Path

import gymnasium as gym
import imageio.v3 as imageio
import pandas as pd
import torch
from stable_baselines3 import DQN

from src.dqn.eval import ENVS


def find_champion(exp_dir, metric="eval_return_mean"):
    df = pd.read_parquet(exp_dir / "all_evals.parquet")
    best = df.loc[df[metric].idxmax()]
    return best["run_no."], int(best["env_steps"]), float(best[metric])


def load_champion_box(exp_dir, env):
    run_no, champ_step, _ = find_champion(exp_dir)
    snaps = torch.load(exp_dir / run_no / "snapshots.pt", weights_only=False)
    idx = list(snaps["steps"]).index(champ_step)
    model = DQN(
        "MlpPolicy", env, policy_kwargs={"net_arch": snaps["net_arch"]}, buffer_size=1
    )
    model.q_net.load_state_dict(snaps["snapshots"][idx])
    model.policy.set_training_mode(False)
    return model, run_no, champ_step


def record_champion_box(
    exp_dir: Path, env_id: str, env_kwargs: dict, out_path: Path, fps=30, tries=20
):
    env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    model, _, _ = load_champion_box(exp_dir, env)
    is_success = ENVS[env_id]["is_success"]
    cap = env.spec.max_episode_steps or 1000
    best = None

    for t in range(tries):
        obs, _ = env.reset(seed=t)
        frames = [env.render()]
        done, L, r, term, trunc, G = False, 0, 0.0, False, False, 0.0
        while not done and L < cap:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(act))
            frames.append(env.render())
            G += r
            done = term or trunc
            L += 1
        if best is None:
            best = frames
        if is_success(term, trunc, r, G, L):
            best = frames
            break

    imageio.imwrite(
        out_path.with_suffix(".mp4"), best, fps=fps, macro_block_size=1, plugin="FFMPEG"
    )
    imageio.imwrite(out_path.with_suffix(".gif"), best, duration=1000 / fps, loop=0)
    env.close()


def save_champion_box(exp_dir: Path, env_id, env_kwargs):
    env = gym.make(env_id, **env_kwargs)
    model, run_no, champ_step = load_champion_box(exp_dir, env)

    model.save(exp_dir / "champion_model.zip")
    (exp_dir / "champion_meta.json").write_text(
        json.dumps(
            {"run_no": run_no, "env_steps": champ_step, "metric": "eval_return_mean"}
        )
    )
    env.close()
