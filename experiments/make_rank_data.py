import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
import config
from scoring import add_true_score

CANDIDATE_FEATURES = [
    "batch", "throughput", "avg_mem",
    "pwr_mean", "pwr_std", "pwr_p95",
    "power_peak_to_mean", "power_range", "power_slope",
    "energy_per_img",
    "gpu_util_mean", "gpu_util_std", "gpu_util_p95", "gpu_util_slope",
    "mem_util_mean", "mem_util_std", "mem_util_p95",
    "temp_mean", "temp_max", "temp_slope",
    "sm_clock_mean", "mem_clock_mean",
    "step_time_std", "step_time_p95", "step_time_cv", "step_time_slope",
    "throughput_var_ratio",
    "power_limit_w", "vram_total_mb",
    "is65W",
]
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare ranker data bundle")
    p.add_argument("--features", default=config.FEATURES_CSV,
                   help="Path to raw features CSV")
    p.add_argument("--weights", nargs=4, type=float,
                   default=config.DEFAULT_WEIGHTS,
                   metavar=("T", "P", "M", "D"),
                   help="Weights T P M Δ (sum not forced to 1)")
    p.add_argument("--group-cols", nargs="+",
                   default=config.GROUP_COLS,
                   help="Columns used to define a group/query")
    p.add_argument("--map-col", default=config.MAP_COL,
                   help="Column name for mAP (used in delta_map)")
    p.add_argument("--cv", type=int, default=config.N_FOLDS,
                   help="0 = no CV, >0 = GroupKFold with that many folds")
    p.add_argument("--out", default="scripts/outputs/rank_data_v1.pkl",
                   help="Output pickle file")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.features)
    print("[DEBUG] args.group_cols =", args.group_cols)
    print("[DEBUG] df.columns =", df.columns.tolist())
    available = set(df.columns)
    group_cols = [c for c in args.group_cols if c in available]
    if len(group_cols) < len(args.group_cols):
        missing = set(args.group_cols) - available
        print(f"[WARN] Missing group cols {missing}. Use {group_cols} instead.")
    if len(group_cols) == 0:
        raise ValueError(
            f"No valid group cols found in CSV. Got {args.group_cols}, "
            f"but df has {list(df.columns)}"
        )
    df = add_true_score(
        df,
        map_col=args.map_col,
        weights=tuple(args.weights),
        group_cols=group_cols,
    )
    df["rel"] = (
            df.groupby(group_cols, sort=False)["true_score"]
            .rank(method="dense", ascending=True)
            .astype(int) - 1
    )
    df["rel"] = df["rel"].clip(upper=31)
    feats = [c for c in CANDIDATE_FEATURES if c in df.columns]

    X = df[feats].fillna(0.0).to_numpy(dtype=float)

    y = df["rel"].to_numpy(dtype=int)

    group_key = df[group_cols].astype(str).agg("|".join, axis=1)
    groups = pd.factorize(group_key, sort=False)[0]

    bundle = {"df": df, "X": X, "y": y, "groups": groups, "features": feats, "group_cols": group_cols}

    n_groups = int(np.unique(groups).size)
    if args.cv and args.cv > 1:
        n_splits = min(args.cv, n_groups)
        if n_splits < 2:
            print(f"Not enough groups for CV (groups={n_groups}). Skipping CV.")
        else:
            gkf = GroupKFold(n_splits=n_splits)
            folds = [(tr.astype(int), va.astype(int)) for tr, va in gkf.split(X, y, groups)]
            bundle["folds"] = folds
            print(f"Generated {n_splits} GroupKFold splits (groups={n_groups}).")
    else:
        print("No CV folds generated (args.cv <= 1).")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"[make_rank_data] saved bundle to:  {args.out}")


if __name__ == "__main__":
    main()
