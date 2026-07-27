"""
Cleanup model .pt artefacts after full eval is done and champion is chosen, to preserve disk space.
"""

import json
from pathlib import Path
import pandas as pd

RUNS = Path("runs")


def cleanup_experiment(
    exp_dir: Path, keep_champion_snapshot=True, drop_coverage=False, dry_run=True
):
    """Safe to call unconditionally. Deletes a seed's snapshots.pt ONLY after proving
    its info is preserved: its rows are in all_evals.parquet AND champion_model.zip
    exists. Idempotent, and never raises -- a cleanup failure can't break the run."""
    try:
        evals = exp_dir / "all_evals.parquet"
        if not evals.exists() or not (exp_dir / "champion_model.zip").exists():
            return 0  # unfinished -> no-op
        df = pd.read_parquet(evals)  # corrupt -> except -> skip
        evaluated = set(df["run_no."].unique())  # seeds actually distilled

        keep_run = None
        if keep_champion_snapshot:
            keep_run = json.loads((exp_dir / "champion_meta.json").read_text())[
                "run_no"
            ]

        freed = 0
        for snap in exp_dir.glob("run_*/snapshots.pt"):
            run = snap.parent.name
            if run not in evaluated:  # eval never covered it -> KEEP
                print(f"keep (not in eval parquet): {run}")
                continue
            if keep_run and run == keep_run:
                continue
            freed += snap.stat().st_size
            if not dry_run:
                snap.unlink()

        if drop_coverage:
            for cov in exp_dir.glob("run_*/obs_coverage.npy"):
                if cov.parent.name in evaluated:
                    freed += cov.stat().st_size
                    if not dry_run:
                        cov.unlink()

        print(
            f"{exp_dir.name}: {'would free' if dry_run else 'freed'} {freed / 1e6:.1f} MB"
        )
        return freed
    except Exception as e:
        print(
            f"cleanup skipped ({exp_dir.name}): {e}"
        )  # swallow -> never abort pipeline
        return 0


def cleanup_all(dry_run=True, **kw):
    total = 0
    for exp_dir in RUNS.glob("*/dqn/*"):
        if not exp_dir.is_dir() or exp_dir.name in ("plots", "videos"):
            continue
        total += cleanup_experiment(exp_dir, dry_run=dry_run, **kw)
    print(f"\nTOTAL {'would free' if dry_run else 'freed'}: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    import sys

    apply = "--apply" in sys.argv  # dry-run unless you pass --apply
    cleanup_all(dry_run=not apply, keep_champion_snapshot=False)
