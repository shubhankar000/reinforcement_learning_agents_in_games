from pathlib import Path

from src.tabular import plots

RUNS, ALGOS, VARIANTS = Path("runs"), ["tabular", "dqn"], ["det", "slip"]

ENV_IDS = ["FrozenLake-v1", "CliffWalking-v1", "Taxi-v4"]


def find_latest(env_id, algo, variant) -> Path:
    cands = sorted((RUNS / env_id / algo).glob(f"*-{variant}"))
    if not cands:
        raise FileNotFoundError(f"no {algo} {variant} for {env_id}")
    return cands[-1]


def make_comparison(env_id: str):
    out = RUNS / env_id / "comparison" / "plots"
    out.mkdir(parents=True, exist_ok=True)

    methods = {f"{a}-{v}": find_latest(env_id, a, v) for a in ALGOS for v in VARIANTS}
    final = plots.final_scores_normalized(methods)
    pairs = {
        f"DQN,Tabular ({v})": (final[f"dqn-{v}"], final[f"tabular-{v}"])
        for v in VARIANTS
    }
    plots.plot_probability_of_improvement(pairs, out / "probability_of_improvement.png")


def main():
    for env_id in ENV_IDS:
        try:
            make_comparison(env_id)
            print(f"{env_id}: comparison done")
        except FileNotFoundError as e:
            print(f"{env_id}: skipped ({e})")  # e.g. a variant not trained yet


if __name__ == "__main__":
    main()
