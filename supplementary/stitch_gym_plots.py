import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

RUNS = Path("runs")
OUT = RUNS / "gym_stitched"

OUT.mkdir(exist_ok=True)


def stitch(paths, titles, nrows, ncols, out, width_in=6.3):
    imgs = [mpimg.imread(p) for p in paths]
    ratio = imgs[0].shape[0] / imgs[0].shape[1]

    fig, ax = plt.subplots(
        nrows, ncols, figsize=(width_in, width_in / ncols * ratio * nrows)
    )

    for ax, im, t in zip(ax.flat, imgs, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=7)
        ax.axis("off")

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / out, dpi=200, bbox_inches="tight")
    plt.close(fig)


GRIDS = ["FrozenLake-v1", "CliffWalking-v1", "Taxi-v4"]
stitch(
    [
        RUNS / e / "comparison/plots" / f"{v}_sample_efficiency.png"
        for v in ("det", "slip")
        for e in GRIDS
    ],
    [f"{e} {v}" for v in ("det", "slip") for e in GRIDS],
    2,
    3,
    "r_grid_efficiency.png",
)

stitch(
    [RUNS / e / "comparison/plots/probability_of_improvement.png" for e in GRIDS],
    GRIDS,
    1,
    3,
    "r_grid_poi.png",
)

stitch(
    [
        RUNS / "CliffWalking-v1" / a / "plots/policy_arrows.png"
        for a in ("tabular", "dqn")
    ],
    ["TQL", "DQN"],
    2,
    1,
    "r_cliff_arrows.png",
)

FA = ["CartPole-v1", "LunarLander-v3", "CarRacing-v3"]
stitch(
    [RUNS / e / "dqn/plots/sample_efficiency.png" for e in FA],
    FA,
    1,
    3,
    "r_fa_efficiency.png",
)
