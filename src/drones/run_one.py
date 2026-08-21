"""
Run one seed of one arm. Idempotent

python run_one.py --arm windowmlp --run-index 3
python run_one.py --arm windowmlp --run-index 3 --fresh    # wipe and restart

lockfile based state machine, decided by what is on disk:
    lockfile present -> the previous attempt died. RESUME from newest checkpoint.
    no lockfile, final.pt there -> already finished. SKIP.
    neither -> fresh run.

The lockfile carries a process PID, so a second launcher hitting the same run directory is refused
"""

import os

# Prevent core thrashing for multiple parallel runs
# Must run before torch is imported
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import shutil
import sys
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import wandb
from src.drones.architectures import ARCHITECTURES
from src.drones.callbacks import ArtifactCallback, CustomMetricsCallback
from src.drones.config import (
    ARMS,
    Arm,
    DroneConfig,
    FloorLRDecay,
    checkpoint_steps,
    run_key,
    snapshot_steps,
)
from src.drones.envs import make_vecenv
from src.drones.extractors import FoldedExtractor, PerFrameMLP
from src.drones.scene import scene_params
from src.drones.wandb_logger import patch_wandb
from src.rng_factory import SeededRNG

ALGOS: dict[str, type[PPO] | type[RecurrentPPO]] = {
    "ppo": PPO,
    "recurrentppo": RecurrentPPO,
}
POLICIES = {"ppo": "MlpPolicy", "recurrentppo": "MlpLstmPolicy"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "runs" / "drones"


def seed_for(master_seed: int, run_index: int) -> int:
    """Recover the seed the in-process runner would have produced for this run.

    Have to rebuild the seed sequence in memory by calling it run_index times to restore to its original value for resume.
    """
    rng = SeededRNG(master_seed)
    for _ in range(run_index + 1):
        seed = rng.next_seed()
    return int(seed)


def read_lock(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"pid": -1}  # corrupt lock still means "died mid-run"


def pid_alive(pid: int) -> bool:
    """
    os.kill(pid, 0) sends no signal, it only checks.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def build_model(arm: Arm, cfg: DroneConfig, venv, seed: int):
    """
    Build the model with the hyperparams for each arm.
    Attach the encoder and arch classes
    """
    arch = ARCHITECTURES[arm.arch] if arm.arch else None

    policy_kwargs = dict(
        net_arch=dict(pi=cfg.algo_config.net_arch_pi, vf=cfg.algo_config.net_arch_vf),
        features_extractor_class=FoldedExtractor,
        features_extractor_kwargs=dict(
            encoder_cls=PerFrameMLP,
            architecture_cls=arch.cls if arch else None,
            architecture_kwargs=arch.kwargs if arch else None,
            d=cfg.algo_config.d,
        ),
    )
    if arm.algo == "recurrentppo":
        policy_kwargs.update(
            lstm_hidden_size=cfg.algo_config.lstm_hidden_size,
            shared_lstm=cfg.algo_config.shared_lstm,
            enable_critic_lstm=cfg.algo_config.enable_critic_lstm,
        )

    # Create the SB3 model. works if fresh or resumed.
    return ALGOS[arm.algo](
        POLICIES[arm.algo],
        venv,
        seed=seed,
        device=cfg.run_config.device,
        verbose=1,
        learning_rate=FloorLRDecay(
            cfg.algo_config.learning_rate, cfg.algo_config.lr_floor_frac
        ),
        n_steps=cfg.algo_config.n_steps,
        batch_size=arm.batch_size,
        n_epochs=cfg.algo_config.n_epochs,
        gamma=cfg.algo_config.gamma,
        gae_lambda=cfg.algo_config.gae_lambda,
        clip_range=cfg.algo_config.clip_range,
        ent_coef=cfg.algo_config.ent_coef,
        vf_coef=cfg.algo_config.vf_coef,
        max_grad_norm=cfg.algo_config.max_grad_norm,
        target_kl=cfg.algo_config.target_kl,
        policy_kwargs=policy_kwargs,
    )


def build_venv(arm, cfg: DroneConfig, seed: int, vecnorm_path: Path | None):
    """
    Raw vec env, then either fresh normalisation or the saved statistics.

    Build without normalisation so a resume can wrap it with VecNormalize.load
    """
    raw = make_vecenv(
        seed,
        n_envs=cfg.run_config.n_envs,
        use_subproc=True,
        stripped=cfg.env_config.stripped,
        flight_mode=cfg.env_config.flight_mode,
        k_frames=arm.k_frames,
        spread=cfg.env_config.spread,
    )

    if vecnorm_path is not None and vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), raw)
        venv.training = True
        return venv

    return VecNormalize(
        raw,
        norm_obs=cfg.env_config.norm_obs,
        norm_reward=cfg.env_config.norm_reward,
        gamma=cfg.algo_config.gamma,
    )


EXPERIMENT_RUN_FIELDS = (
    "master_seed",
    "step_budget",
    "n_envs",
    "checkpoint_every",
    "n_snapshots",
    "snapshot_start",
)


def experiment_fields(config: dict) -> dict:
    """The parts of a stored config that change RESULTS."""
    return {
        "algo": config["algo_config"],
        "env": config["env_config"],
        "run": {k: config["run_config"][k] for k in EXPERIMENT_RUN_FIELDS},
    }


def build_meta(arm: Arm, cfg: DroneConfig, seed: int, run_index: int) -> dict:
    return {
        "arm": arm.name,
        "arch": arm.arch,
        "k_frames": arm.k_frames,
        "batch_size": arm.batch_size,
        "run_key": run_key(arm, cfg.env_config.axis2),
        "run_index": run_index,
        "seed": seed,
        "config": cfg.to_dict(),
        "versions": {
            p: version(p)
            for p in (
                "stable_baselines3",
                "sb3_contrib",
                "torch",
                "numpy",
                "gymnasium",
                "PyFlyt",
            )
        },
        "wandb_run_id": None,
        # -1 means never resumed. Replaced by the actual step on the first resume.
        "resumed_at_steps": [-1],
        "scene_params": scene_params() if cfg.env_config.n_obstacles else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--run-index", type=int, required=True)
    ap.add_argument("--fresh", action="store_true", help="wipe the run dir first")
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the step budget, for tests only.",
    )
    args = ap.parse_args()

    torch.set_num_threads(1)

    cfg = DroneConfig()
    if args.steps is not None:
        # scale all the schedulers with the step budget
        cfg.run_config.step_budget = args.steps
        cfg.run_config.snapshot_start = max(1, args.steps // 10)
        cfg.run_config.checkpoint_every = max(1, args.steps // 4)

    arm = ARMS[args.arm]
    seed = seed_for(cfg.run_config.master_seed, args.run_index)

    run_dir = (
        Path(RUNS_ROOT)
        / cfg.env_config.task
        / run_key(arm, cfg.env_config.axis2)
        / f"run_{args.run_index:02d}"
    )
    lock_path = run_dir / ".running"
    meta_path = run_dir / "meta.json"
    final_path = run_dir / "final.pt"

    if args.fresh:
        shutil.rmtree(run_dir, ignore_errors=True)

    lock = read_lock(lock_path)

    # another process is live on this directory - do not corrupt it
    if lock and pid_alive(lock.get("pid", -1)):
        print(f"[skip] {run_dir} is locked by live pid {lock['pid']}")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    meta = build_meta(arm, cfg, seed, args.run_index)

    # guard against config drift
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if experiment_fields(old["config"]) != experiment_fields(meta["config"]):
            raise SystemExit(f"config differs from {meta_path}. Re-run with --fresh")
        meta = old

    # clean finish, no lock present
    if lock is None and final_path.exists():
        print(f"[skip] {run_dir} already complete")
        return

    ckpts = sorted((run_dir / "checkpoints").glob("step_*.zip"))
    resume_from = ckpts[-1] if ckpts else None

    venv = build_venv(
        arm,
        cfg,
        seed,
        resume_from.with_name(f"{resume_from.stem}_vecnorm.pkl")
        if resume_from
        else None,
    )

    if resume_from is not None:
        model = ALGOS[arm.algo].load(
            resume_from, env=venv, device=cfg.run_config.device
        )
        done = model.num_timesteps
        steps = [s for s in meta["resumed_at_steps"] if s != -1] + [done]
        meta["resumed_at_steps"] = steps
        print(f"[resume] {run_dir} from {resume_from.name} at {done:,} steps")
    else:
        model = build_model(arm, cfg, venv, seed)
        done = 0
        print(f"[start] {run_dir}  arm={arm.name}  seed={seed}")

    remaining = cfg.run_config.step_budget - done
    if remaining <= 0:
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        finalise(run_dir, model, venv, lock_path, final_path)
        print(f"[done] {run_dir} already at {done:,} of {cfg.run_config.step_budget:,}")
        venv.close()
        return

    n_params = sum(p.numel() for p in model.policy.parameters())

    wandb_run = wandb.init(
        project=cfg.run_config.wandb_project,
        mode=cfg.run_config.wandb_mode,
        group=arm.name,  # use wandb groups by arm
        name=f"{arm.name}-seed{args.run_index:02d}",
        tags=[
            cfg.env_config.task,
            f"mode{cfg.env_config.flight_mode}",
            f"{cfg.run_config.step_budget // 1_000_000}M",
        ],
        id=meta["wandb_run_id"],
        resume="must" if meta["wandb_run_id"] else None,
        config=None
        if meta["wandb_run_id"]
        else {
            "arm": arm.name,
            "run_index": args.run_index,
            "seed": seed,
            "run_key": run_key(arm, cfg.env_config.axis2),
            "k_frames": arm.k_frames,
            "batch_size": arm.batch_size,
            "n_params": n_params,
            "python": sys.executable,
            **meta["versions"],
            **cfg.to_dict(),
        },
    )

    meta["wandb_run_id"] = wandb_run.id
    patch_wandb(model)

    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    lock_path.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))

    try:
        callback = ArtifactCallback(
            run_dir,
            snapshot_steps(cfg.run_config),
            checkpoint_steps(cfg.run_config),
            cfg.run_config.keep_checkpoints,
        )
        model.learn(
            remaining,
            callback=[callback, CustomMetricsCallback()],
            reset_num_timesteps=(resume_from is None),
            progress_bar=False,
        )

        finalise(run_dir, model, venv, lock_path, final_path)
        print(f"[done] {run_dir} at {model.num_timesteps:,} steps")
    finally:
        venv.close()
        wandb.finish()


def finalise(
    run_dir: Path,
    model: PPO | RecurrentPPO,
    venv: VecNormalize,
    lock_path: Path,
    final_path: Path,
):
    obs_rms = venv.obs_rms if isinstance(venv, VecNormalize) else None
    ret_rms = venv.ret_rms if isinstance(venv, VecNormalize) else None
    torch.save(
        {
            "step": model.num_timesteps,
            "policy": model.policy.state_dict(),
            "obs_rms": obs_rms,
            "ret_rms": ret_rms,
        },
        final_path,
    )
    venv.save(str(run_dir / "final_vecnorm.pkl"))

    # jsonl was the crash-safe write format; parquet is the artifact schema
    log = run_dir / "train_log.jsonl"
    if log.exists():
        pd.read_json(log, lines=True).to_parquet(run_dir / "train_log.parquet")

    lock_path.unlink(missing_ok=True)  # ONLY on success: the lock IS the crash flag


if __name__ == "__main__":
    main()
