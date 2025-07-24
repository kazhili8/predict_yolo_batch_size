import pandas as pd, numpy as np, pathlib

DF = pathlib.Path(r"scripts/outputs/dataframe/features_v4.csv")
OUT_X = "rank_features.npy"
OUT_y = "rank_labels.npy"
OUT_group = "rank_group.npy"
df = pd.read_csv(DF)

MAP_CANDIDATES = ["avg_map50", "map50", "mAP50", "mAP@0.5"]
map_cols = [c for c in MAP_CANDIDATES if c in df.columns]

if map_cols:
    map_col = map_cols[0]
    if df[map_col].notna().any():
        df["delta_map"] = (df[map_col].max() - df[map_col]).abs()
    else:
        df["delta_map"] = 0.0
else:
    print("Warning: no map50 column found; delta_map set to 0.")
    df["delta_map"] = 0.0

for need in ["throughput", "avg_power", "avg_mem"]:
    if need not in df.columns:
        df[need] = np.nan

df[["throughput", "avg_power", "avg_mem"]] = \
    df[["throughput", "avg_power", "avg_mem"]].fillna(0.0)

df["score"] = (
    0.5 * df["throughput"] -
    0.3 * df["avg_power"] -
    0.1 * df["avg_mem"] -
    0.1 * df["delta_map"]
)
df["is65W"] = ((df.get("tag") == "65W").astype(int))
FEATS_TARGET = ["batch", "throughput", "avg_mem",
                "pwr_mean", "pwr_std", "energy_per_img","is65W"]
FEATS = [c for c in FEATS_TARGET if c in df.columns]

if len(FEATS) == 0:
    raise ValueError("None of the target feature columns exist in the dataframe.")

X = df[FEATS].fillna(0.0).to_numpy()
y = df["score"].to_numpy(dtype=float)

grp_cols = [c for c in ["model", "epochs", "tag"] if c in df.columns]
df[grp_cols] = df[grp_cols].fillna("unk")
g = df.groupby(grp_cols, sort=False).size().to_numpy()
assert g.sum() == len(df), f"group sum {g.sum()} != samples {len(df)}"

np.save(OUT_X, X)
np.save(OUT_y, y)
np.save(OUT_group, g)

print(f"Saved ranking data:"
      f"\n  samples = {len(df)}"
      f"\n  features = {len(FEATS)} -> {FEATS}"
      f"\n  groups = {len(g)}, sum(groups) = {g.sum()}")
