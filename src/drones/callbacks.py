"""
Snapshot, rolling checkpoints and incremental metrics logging.
"""

import math
import json
import time
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import BaseCallback

# patch grad norm in PPO
_GRAD: dict[str, float] = {}


def patch_grad_norm() -> None:
    if getattr(torch.nn.utils, "_drone_patched", False):
        return

    original = torch.nn.utils.clip_grad_norm_

    def patched(parameters, max_norm, *args, **kwargs):
        total = original(parameters, max_norm, *args, **kwargs)
        value = float(total)
        _GRAD["sum"] = _GRAD.get("sum", 0.0) + value
        _GRAD["n"] = _GRAD.get("n", 0) + 1
        _GRAD["clipped"] = _GRAD.get("clipped", 0) + int(value > max_norm)
        _GRAD["nonfinite"] = _GRAD.get("nonfinite", 0) + int(not math.isfinite(value))
        return total

    torch.nn.utils.clip_grad_norm_ = patched
    torch.nn.utils._drone_patched = True


GROUP_PATHS = {
    "encoder": ("features_extractor", "encoder"),
    "architecture": ("features_extractor", "architecture"),
    "lstm": ("lstm_actor",),  # recurrent arm only
    "mlp": ("mlp_extractor",),
    "action_head": ("action_net",),
    "value_head": ("value_net",),
}


def resolve(policy, path):
    obj = policy
    for attr in path:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def grad_norm(module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm(2)) ** 2
    return total**0.5


class CustomMetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._reasons = {}
        self._episodes = 0

    def _on_training_start(self) -> None:
        patch_grad_norm()

        for name, path in GROUP_PATHS.items():
            module = resolve(self.model.policy, path)
            if module is None:
                continue
            n = sum(p.numel() for p in module.parameters())
            self.logger.record(f"params/{name}", n)
        total = sum(p.numel() for p in self.model.policy.parameters())
        self.logger.record("params/total", total)
        assert total > 0, "policy has no parameters"

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            self._episodes += 1
            for key in ("collision", "out_of_bounds", "env_complete"):
                if info.get(key):
                    self._reasons[key] = self._reasons.get(key, 0) + 1
            if info.get("TimeLimit.truncated"):
                self._reasons["truncated"] = self._reasons.get("truncated", 0) + 1
        return True

    def _on_rollout_end(self) -> None:
        if self._episodes:
            for key in ("collision", "out_of_bounds", "env_complete", "truncated"):
                self.logger.record(
                    f"term/{key}", self._reasons.get(key, 0) / self._episodes
                )
            self.logger.record("term/n_episodes", self._episodes)
        self._reasons.clear()
        self._episodes = 0

        n = _GRAD.get("n", 0)
        if n:
            self.logger.record("grad/pre_clip_norm", _GRAD["sum"] / n)
            self.logger.record("grad/clipped_frac", _GRAD["clipped"] / n)
            self.logger.record("grad/nonfinite", _GRAD["nonfinite"])
        _GRAD.clear()

        for name, path in GROUP_PATHS.items():
            module = resolve(self.model.policy, path)
            if module is not None:
                self.logger.record(f"grad/{name}", grad_norm(module))


class ArtifactCallback(BaseCallback):
    """
    Writes snapshots, checkpoints and the metric log for 1 run
    """

    def __init__(
        self,
        run_dir: Path,
        snapshot_at,
        checkpoint_at,
        keep_checkpoints: int,
    ):
        super().__init__()

        self.run_dir = Path(run_dir)
        self.snapshot_at = list(snapshot_at)
        self.checkpoint_at = list(checkpoint_at)
        self.keep_checkpoints = keep_checkpoints

        self.snaps_dir = self.run_dir / "snapshots"
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.log_path = self.run_dir / "train_log.jsonl"

        self._snap_i = 0
        self._ckpt_i = 0
        self._t0 = None

    def _on_training_start(self) -> None:
        self.snaps_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.perf_counter()

        # Resume: advance state for everything already done.
        # 1/3 advance pointers to everything already written
        done = self.num_timesteps
        self._snap_i = sum(1 for s in self.snapshot_at if s <= done)
        self._ckpt_i = sum(1 for s in self.checkpoint_at if s <= done)

        kept = []
        if self.log_path.exists():
            # 2/3 drop rows past the checkpoint being resumed from.
            # without this those steps appear twice in the log with diff elapsed_s
            for line in self.log_path.read_text().splitlines():
                if line.strip() and json.loads(line)["step"] <= done:
                    kept.append(line)

            self.log_path.write_text("\n".join(kept) + ("\n" if kept else ""))

        if kept:
            # 3/3 move wall clock forward
            self._t0 -= json.loads(kept[-1])["elapsed_s"]

    def _vn_stats(self):
        """
        Statistics of VecNormalizer's current state, saved with every snapshot so that eval uses it properly
        """
        venv = self.model.get_vec_normalize_env()
        if venv is None:
            return None, None

        return venv.obs_rms, venv.ret_rms

    def _save_snapshot(self, step: int):
        """
        Save state dict only, not full model with optimizer states.
        Purely for eval
        """
        obs_rms, ret_rms = self._vn_stats()
        torch.save(
            {
                "step": step,
                "policy": self.model.policy.state_dict(),
                "obs_rms": obs_rms,
                "ret_rms": ret_rms,
            },
            self.snaps_dir / f"step_{step:08d}.pt",
        )

    def _save_checkpoint(self, step: int):
        """
        Save full SB3 model. This is for checkpoint/resume.
        """
        self.model.save(self.ckpt_dir / f"step_{step:08d}.zip")

        venv = self.model.get_vec_normalize_env()
        if venv is not None:
            venv.save(str(self.ckpt_dir / f"step_{step:08d}_vecnorm.pkl"))

        # keep only newest N rolling checkpoints
        zips = sorted(self.ckpt_dir.glob("step_*.zip"))
        for old in zips[: -self.keep_checkpoints]:
            old.unlink(missing_ok=True)
            (self.ckpt_dir / f"{old.stem}_vecnorm.pkl").unlink(missing_ok=True)

    def _on_step(self) -> bool:
        now = self.num_timesteps

        while (
            self._snap_i < len(self.snapshot_at)
            and now >= self.snapshot_at[self._snap_i]
        ):
            self._save_snapshot(self.snapshot_at[self._snap_i])
            self._snap_i += 1

        while (
            self._ckpt_i < len(self.checkpoint_at)
            and now >= self.checkpoint_at[self._ckpt_i]
        ):
            self._save_checkpoint(self.checkpoint_at[self._ckpt_i])
            self._ckpt_i += 1

        return True

    def _on_rollout_end(self) -> None:
        """
        One metric row per rollout, append immediately rather than keep in memory, to allow for proper checkpoint/resume.

        Have to use jsonl since parquet is not append friendly. jsonl->parquet conversion done at the end after successful completion
        """
        buf = self.model.ep_info_buffer

        if not buf:
            return

        row = {
            "step": self.num_timesteps,
            "ep_rew_mean": sum(e["r"] for e in buf) / len(buf),
            "ep_len_mean": sum(e["l"] for e in buf) / len(buf),
            "n_episodes": len(buf),
            "elapsed_s": round(time.perf_counter() - self._t0, 1),
        }

        with self.log_path.open("a") as f:
            # jsonl is just multi line json
            f.write(json.dumps(row) + "\n")
