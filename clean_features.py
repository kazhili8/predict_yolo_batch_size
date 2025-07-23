import pandas as pd, numpy as np, pathlib

RAW = pathlib.Path(r"scripts\outputs\dataframe\features_raw.csv")
OUT = RAW.with_name("features_clean.csv")

df = pd.read_csv(RAW)

num_cols = df.select_dtypes(float).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

for suf in ("e1","e10","e100"):
    t_col = f"avg_step_time_{suf}"
    if t_col in df:
        df[f"throughput_{suf}"] = df["batch"] / df[t_col]

df.sort_values(["model","batch"]).to_csv(OUT, index=False)
print("saved →", OUT)