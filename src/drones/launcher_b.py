"""
The orchestrator that runs the whole ablation for Task B

It does this by repeatedly calling run_one.py

Uses subprocess.Popen so that if 1 arm fails, the whole thing doesnt fail.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

from src.drones.config import ARMS, DroneConfig, run_key, Arm

POLL_RATE = 5
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_queue(arms, runs):
    """
    Queue is sorted longest first, so the long arms can keep running from the start to optimize time and VM resources.
    """
    jobs = []
    for a in arms:
        for r in runs:
            jobs.append((ARMS[a], r))

    return sorted(jobs, key=lambda x: x[0].algo_speed)


def spawn_arm(arm: Arm, run_index: int, log_dir: Path, *args):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{arm.name}_run{run_index:02d}.log"

    handler = log_path.open("a")
    now = datetime.now()
    handler.write(f"\n===== attempt {now.strftime('%Y-%m-%d %H:%M:%S')}  =====\n")
    handler.flush()

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.drones.run_one_b",
        "--arm",
        arm.name,
        "--run-index",
        str(run_index),
        *args,
    ]
    process = subprocess.Popen(cmd, stdout=handler, stderr=subprocess.STDOUT)
    return process, log_path, handler


def main():
    cfg = DroneConfig()

    # ======== Overrides for Task B ======== #
    cfg.env_config.task = "waypoint"
    cfg.run_config.n_runs = 5  # Task A was 10, but Task B is expensive
    cfg.run_config.n_concurrent = (
        4  # TODO change this when running on GPU VM. THis is for laptop
    )
    # ======== End Overrides ======== #

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print the queue and exit")
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="pass through to run_one, for testing purposes only",
    )
    args = ap.parse_args()

    arms = sorted(ARMS)
    runs = list(range(cfg.run_config.n_runs))
    queue = build_queue(arms, runs)
    n_concurrent = cfg.run_config.n_concurrent

    child_args = ["--steps", str(args.steps)] if args.steps is not None else []
    log_dir = PROJECT_ROOT / "runs" / "drones" / cfg.env_config.task / "_logs"

    print(f"{len(queue)} jobs, {n_concurrent} concurrency, longest first")

    for arm, r in queue[:3] + queue[-3:]:
        print(
            f"    {run_key(arm, cfg.env_config.axis2)} run_{r:02d}  speed {arm.algo_speed}"
        )

    # if dry run print queue and exit
    if args.dry_run:
        return

    pending = list(queue)
    running = {}
    failed = []
    finished = 0
    t0 = time.perf_counter()

    try:
        while pending or running:
            while pending and len(running) < n_concurrent:
                arm, run_index = pending.pop(0)
                process, log_path, handler = spawn_arm(
                    arm, run_index, log_dir, *child_args
                )
                running[process] = (
                    arm,
                    run_index,
                    log_path,
                    handler,
                    time.perf_counter(),
                )

            finished_procs = [p for p in running if p.poll() is not None]
            for process in finished_procs:
                arm, run_index, log_path, handler, started = running.pop(process)
                handler.close()
                finished += 1
                mins = (time.perf_counter() - started) / 60

                if process.returncode == 0:
                    log_path.unlink(missing_ok=True)
                    print(
                        f"[{finished}/{len(queue)}] ok   {arm.name} run_{run_index:02d}  ({mins:.0f}m)"
                    )
                else:
                    failed.append((arm.name, run_index, process.returncode, log_path))
                    print(
                        f"[{finished}/{len(queue)}] FAIL {arm.name} run_{run_index:02d}  rc={process.returncode}  {log_path}"
                    )

            time.sleep(POLL_RATE)

    finally:
        for process in running:
            handler.close()
            process.terminate()

    print(f"\nfinished in {(time.perf_counter() - t0) / 3600:.1f} h")

    if failed:
        print(
            f"{len(failed)} failed. rerun launcher, it will automatically discover failed runs and rerun only them from their checkpoints"
        )
        for name, run_index, rc, log_path in failed:
            print(f"   {name} run_{run_index:02d}  rc={rc}  {log_path}")
    else:
        print("All runs completed!")


if __name__ == "__main__":
    main()
