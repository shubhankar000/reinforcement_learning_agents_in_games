from pathlib import Path
import pandas as pd

d = Path("runs") / "drones" / "pole-balance"
cols = ["eval_return_mean", "episode_len_mean"]

row = []
for arm in d.glob("*ppo*"):
    df = pd.read_parquet(str(arm / "all_evals.parquet"))
    df = df[df["env_steps"] == df["env_steps"].max()]
    m = df[cols]
    row.append((arm.name, m.mean().values.tolist()))
    # print(arm.name, round((df[cols[1]].mean() / 800).item(), 2))


final = pd.DataFrame(row)
final.columns = ["arm", "ret,len"]
print(final)

d = Path("runs") / "drones" / "waypoint"
cols = ["eval_return_mean", "episode_len_mean"]

row = []
for arm in d.glob("*ppo*"):
    df = pd.read_parquet(str(arm / "all_evals.parquet"))
    df = df[df["env_steps"] == df["env_steps"].max()]
    m = df[cols]
    row.append((arm.name, m.mean().values.tolist()))
    # print(arm.name, round((df[cols[1]].mean() / 800).item(), 2))


final = pd.DataFrame(row)
final.columns = ["arm", "ret,len"]
print(final)
