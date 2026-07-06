"""
Heatmap figures file, helper functions to generate heatmaps
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GRID = {"FrozenLake-v1": (4, 4), "CliffWalking-v1": (4, 12)}
ARROW = {
    "FrozenLake-v1": {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)},  # L D R U
    "CliffWalking-v1": {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)},  # U R D L
}
ACTIONS = {
    "FrozenLake-v1": ["Left", "Down", "Right", "Up"],
    "CliffWalking-v1": ["Up", "Right", "Down", "Left"],
    "Taxi-v4": ["South", "North", "East", "West", "Pickup", "Dropoff"],
}
SPECIAL = {
    "FrozenLake-v1": {"start": [0], "goal": [15], "block": [5, 7, 11, 12]},  # Holes
    "CliffWalking-v1": {
        "start": [36],
        "goal": [47],
        "block": list(range(37, 47)),
    },  # Cliffs
}


def best_run(exp_dir: Path) -> str:
    df = pd.read_parquet(exp_dir / "all_evals.parquet")
    last = df[df.env_steps == df.env_steps.max()]
    return last.loc[last.eval_return_mean.idxmax(), "run_no."]


def champion_snap(exp_dir: Path):
    snap = np.load(exp_dir / best_run(exp_dir) / "snapshots.npz")
    return snap["snapshots"][-1], snap["visitation"]


def initial_frame(env_id: str, env_kwargs: dict):
    """Render the env's starting frame as an (H, W, 3) array for overlays."""
    env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    env.reset(seed=0)
    frame = env.render()
    env.close()
    return frame


def plot_visitation_spatial(
    variant_dirs: dict, env_id: str, env_kwargs: dict, out_path: Path
):
    rows, cols = GRID[env_id]
    frame = initial_frame(env_id, env_kwargs)
    extent = [-0.5, cols - 0.5, rows - 0.5, -0.5]  # map frame onto grid-index coords
    fig, axes = plt.subplots(1, len(variant_dirs), figsize=(6 * len(variant_dirs), 4))
    for ax, (label, d) in zip(np.atleast_1d(axes), variant_dirs.items()):
        _, vis = champion_snap(d)
        grid = vis.sum(axis=1).reshape(rows, cols)  # sum over actions
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(alpha=0)  # holes/unvisited transparent -> show map beneath
        ax.imshow(frame, extent=extent)  # starting frame as backdrop
        im = ax.imshow(
            np.ma.masked_less(grid, 1),  # mask never-visited
            norm=mcolors.LogNorm(vmin=1),
            cmap=cmap,
            alpha=0.6,
            extent=extent,
            interpolation="nearest",
        )  # LOG scale, critical
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, label="state visits")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_visitation_sa(variant_dirs: dict, env_id: str, out_path: Path):
    fig, axes = plt.subplots(1, len(variant_dirs), figsize=(4 * len(variant_dirs), 6))
    for ax, (label, d) in zip(np.atleast_1d(axes), variant_dirs.items()):
        _, vis = champion_snap(d)  # (n_states, n_actions)
        im = ax.imshow(
            np.ma.masked_less(vis, 1), norm=mcolors.LogNorm(vmin=1), aspect="auto"
        )
        ax.set_xticks(range(len(ACTIONS[env_id])))
        ax.set_xticklabels(ACTIONS[env_id], rotation=45, ha="right")
        ax.set_ylabel("state")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, label="visits")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_policy_arrows(
    variant_dirs: dict, env_id: str, env_kwargs: dict, out_path: Path
):
    rows, cols = GRID[env_id]
    sp = SPECIAL[env_id]
    frame = initial_frame(env_id, env_kwargs)
    extent = [-0.5, cols - 0.5, rows - 0.5, -0.5]  # map frame onto grid-index coords
    fig, axes = plt.subplots(1, len(variant_dirs), figsize=(6 * len(variant_dirs), 4))
    for ax, (label, d) in zip(np.atleast_1d(axes), variant_dirs.items()):
        Q, _ = champion_snap(d)
        V = Q.max(axis=1).reshape(rows, cols)
        policy = Q.argmax(axis=1)
        ax.imshow(frame, extent=extent)  # starting frame as backdrop
        im = ax.imshow(V, cmap="coolwarm", alpha=0.45, extent=extent)  # value tint
        skip = set(sp["goal"]) | set(sp["block"])  # no arrow on terminal cells
        for s in range(rows * cols):
            if s in skip:
                continue
            r, c = divmod(s, cols)
            u, v = ARROW[env_id][int(policy[s])]
            ax.quiver(
                c,
                r,
                u,
                v,
                angles="xy",
                scale_units="xy",
                scale=2.5,
                color="black",
                width=0.01,
            )  # scale>1 → arrow < 1 cell
        ax.set_title(label)
        ax.axis("off")
        fig.colorbar(im, ax=ax, label=r"$V(s)=\max_a Q$")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def record_champion(
    exp_dir, env_id, env_kwargs, out_path, seed=0, cap=200, fps=3, tries=20
):
    Q, _ = champion_snap(exp_dir)
    env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    best_frames = None
    for t in range(tries):
        obs, _ = env.reset(seed=seed + t)
        frames, done, L, r, term, trunc = [env.render()], False, 0, 0.0, False, False
        while not done and L < cap:
            obs, r, term, trunc, _ = env.step(int(np.argmax(Q[obs])))
            frames.append(env.render())
            done, L = term or trunc, L + 1
        # FrozenLake terminates on holes too, so require reward; CW/Taxi only end at goal
        reached = term and (r == 1 if env_id == "FrozenLake-v1" else True)
        if best_frames is None:
            best_frames = frames  # fallback: keep first attempt
        if reached:
            best_frames = frames
            break
    # write both: mp4 (scrubbable) and gif (frame-exact; VSCode's mp4 preview
    # motion-smooths short clips into fake diagonal motion). Both land in videos/.
    out_path = Path(out_path)
    imageio.mimsave(out_path.with_suffix(".mp4"), best_frames, fps=fps, macro_block_size=1)
    imageio.mimsave(out_path.with_suffix(".gif"), best_frames, fps=fps)
