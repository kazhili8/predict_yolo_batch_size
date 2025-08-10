import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold

CANDIDATE_FEATURES = [
    "batch","throughput","avg_mem","pwr_mean","pwr_std","pwr_p95","power_peak_to_mean","power_range","power_slope",
    "energy_per_img","gpu_util_mean","gpu_util_std","gpu_util_p95","gpu_util_slope",
    "mem_util_mean","mem_util_std","mem_util_p95","mem_util_slope",
    "sm_clock_mean","sm_clock_std","sm_clock_p95","sm_clock_slope",
    "mem_clock_mean","mem_clock_std","mem_clock_p95","mem_clock_slope",
    "temp_mean","temp_std","temp_p95","temp_slope",
    "step_time_mean","step_time_std","step_time_p95","step_time_cv","step_time_slope",
    "thr_mean","thr_std","thr_p95","throughput_var_ratio",
    "eff_tp_watt","eff_tp_mem","inv_step_time","pwr_cv","vram_total_mb","power_limit_w","n_logs"
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--weights-table", default="")
    p.add_argument("--weights", nargs=4, type=float, default=[0.6,0.2,0.1,0.1])
    p.add_argument("--group-cols", nargs="+", default=["model","epochs","tag"])
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--weight_alpha", type=float, default=1.0)
    p.add_argument("--out", required=True)
    return p.parse_args()

def pick_best_weights(path, fallback):
    csv = Path(path)
    if not csv.exists():
        return tuple(fallback)
    try:
        df = pd.read_csv(csv)
        r = df.sort_values("Top1", ascending=False).iloc[0]
        return float(r["T"]), float(r["P"]), float(r["M"]), float(r["Δ"])
    except Exception:
        return tuple(fallback)

def build_true_score(df, w):
    T,P,M,D = w
    if "throughput" not in df.columns and "avg_step_time" in df.columns:
        df = df.copy()
        df["throughput"] = 1.0 / df["avg_step_time"].astype(float)
    df["delta_map"] = df["map50"].max() - df["map50"]
    return T*df["throughput"] - P*df["avg_power"] - M*df["avg_mem"] - D*df["delta_map"]

def make_groups(df, group_cols):
    keys = df[group_cols].astype(str).agg("|".join, axis=1).values
    uniq, inv = np.unique(keys, return_inverse=True)
    return inv

def domain_balance_weights(df, group_cols, alpha):
    gcols = [c for c in group_cols if c != "tag"]
    df = df.copy()
    if "tag" not in df.columns:
        return np.ones(len(df), dtype=float)
    w = np.ones(len(df), dtype=float)
    grouped = df.groupby(gcols, sort=False) if gcols else [(None, df)]
    for _, sub in grouped:
        counts = sub["tag"].value_counts()
        total = float(len(sub))
        for t, cnt in counts.items():
            idx = sub.index[sub["tag"]==t]
            base = total / (len(counts)*float(cnt))
            w[idx] = base
    w = w / np.mean(w)
    instab = []
    for c in ["pwr_cv","step_time_cv"]:
        if c in df.columns:
            v = df[c].fillna(0).to_numpy(dtype=float)
            v = (v - v.min()) / (v.max() - v.min() + 1e-12)
            instab.append(v)
    if instab:
        s = np.mean(np.vstack(instab), axis=0)
        w = w * (1.0 / (1.0 + alpha*s))
    w = np.clip(w, 0.2, 5.0)
    w = w / np.mean(w)
    return w

def main():
    args = parse_args()
    df = pd.read_csv(args.features)
    best_w = pick_best_weights(args.weights_table, args.weights)
    print(f"[DEBUG] group_cols = {args.group_cols}")
    print(f"[DEBUG] df.shape = {df.shape}")
    feats = [c for c in CANDIDATE_FEATURES if c in df.columns]
    if "throughput" not in feats and "avg_step_time" in df.columns:
        df["throughput"] = 1.0 / df["avg_step_time"].astype(float)
        feats.append("throughput")
    y = build_true_score(df, best_w).to_numpy(dtype=float)
    X = df[feats].to_numpy(dtype=float)
    groups = make_groups(df, args.group_cols)
    sample_weight = domain_balance_weights(df, args.group_cols, args.weight_alpha).astype(float)
    bundle = {"df": df, "X": X, "y": y, "groups": groups, "features": feats, "group_cols": args.group_cols, "sample_weight": sample_weight}
    n_groups = int(np.unique(groups).size)
    if args.cv and args.cv > 1 and n_groups >= args.cv:
        gkf = GroupKFold(n_splits=args.cv)
        folds = [(tr.astype(int), va.astype(int)) for tr, va in gkf.split(X, y, groups)]
        bundle["folds"] = folds
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"[make_rank_data] saved bundle to:  {args.out}")

if __name__ == "__main__":
    main()
