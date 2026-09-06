# Reinforcement Learning Architectures in Drone Simulation

This is a dissertation for MSc Machine Learning at Royal Holloway, University of London. 

Submitted 6th September 2026

This project is an architecture ablation (9 arms) for PPO on two PyFlyt drone tasks, preceded by TQL/DQN validation on 6 gym environments.

This readme complements Section 10 of the report

## Requirements
- [uv](https://docs.astral.sh/uv/getting-started/installation/). `uv` installs Python 3.13
- Nvidia GPU only for Regime B training (launcher falls back to CPU if GPU not available, needs to be explicitly passed in as `--device cpu` when using `run_one_b.py`)
- Developed on WSL2. Should run fine on Linux and macOS, Windows is untested.

## Setup
- Run `uv sync` to install python and all packages
- Run `uv run wandb login` for the drones pipeline or set `wandb_mode="disabled"` in the config

## Gymnasium pipeline (TQL + DQN)
- `uv run src/tabular/run_validation.py` - computes the V*/Q* anchors, must be run first
- `uv run src/run_dqn_tabular.py` - trains, evaluates, generate plots and champion videos.

## Drones pipeline
Run in order:
- `uv run supplementary/taskA_V.py` - Regime A anchors (V* analytic, V_random measured)
- `uv run supplementary/taskB_V.py` - Regime B anchors (via flight_mode 7)
- `uv run src/drones/launcher.py` - Run Regime A fully; 90 runs, 24 concurrent, CPU-only
- `uv run src/drones/launcher_b.py` - Run Regime B fully; 45 runs, 12 concurrent, needs a GPU
- `uv run src/drones/eval.py` - Parallel eval of Regime A and B, lighter than training (14 worker processes by default)
- `uv run src/drones/plots.py` - Generate all rliable plots
- `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA LIBGL_ALWAYS_SOFTWARE=1 uv run src/drones/champion_video.py` - Find the champion and run it in eval, meant for screen capture (environment variables are WSL2 specific)


## Small-machine notes
- Both launchers accept `--dry-run`. This prints the queue and exits, for testing.

If individual arms are to be run standalone, use this example (change whats needed)
- `uv run src/drones/run_one.py --arm memoryless --run-index 0` - Run Regime A with memoryless arm, run index 0
- `uv run src/drones/run_one_b.py --arm windowmlp --run-index 0 --device cpu` - Run Regime B with windowmlp arm, run index 0, device cpu (default device is `cuda`)

## Layout
- `src/` (tabular, deep, drones + shared config/rng/runner)
- `supplementary/`
- `runs/` (all run artifacts)
- `vendored_libs` (changzy00/pytorch-attention, MIT, license kept)

## Acknowledgements
- gymnasium
- Stable Baselines 3
- sb3-contrib
- Weights and Biases
- PyFlyt
- [Pytorch Attention](https://github.com/changzy00/pytorch-attention) (license in the folder)