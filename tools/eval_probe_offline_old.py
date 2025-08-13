import argparse, pandas as pd, numpy as np
from pathlib import Path

def pick_map_col(df):
    for c in ["map50","avg_map50","mAP50","mAP@0.5"]:
        if c in df.columns: return c
    raise RuntimeError("no mAP column")

def make_true_score(df, weights, oracle, map_col):
    if oracle=="throughput": return df["throughput"].astype(float)
    T,P,M,A = weights
    dmap = df[map_col].max() - df[map_col]
    return T*df["throughput"] - P*df["avg_power"] - M*df["avg_mem"] - A*dmap

def feasible_mask(df, cap_w, delta_map, map_col):
    b1 = df[df["batch"]==1]
    if len(b1)==0: return pd.Series(False, index=df.index)
    base = float(b1[map_col].iloc[0])
    return (df["pwr_mean"]<=cap_w) & (df[map_col] >= base*(1.0-delta_map))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--tag", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--cap_w", type=float, required=True)
    ap.add_argument("--delta_map", type=float, default=0.01)
    ap.add_argument("--oracle", choices=["score","throughput"], default="score")
    ap.add_argument("--weights", nargs=4, type=float, default=[0.60,0.20,0.10,0.10])
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)
    if args.tag:
        df = df[df["tag"].astype(str).isin(args.tag)].copy()
    df = df[df["epochs"]==args.epochs].copy()
    map_col = pick_map_col(df)

    rows = []
    for (m,t), g0 in df.groupby(["model","tag"], sort=False):
        g = g0.copy()
        fm = feasible_mask(g, args.cap_w, args.delta_map, map_col)
        g = g[fm].copy()
        if g.empty: continue
        b1 = float(g[g["batch"]==1][map_col].iloc[0])
        g["oracle_score"] = make_true_score(g, args.weights, args.oracle, map_col)
        b_orc = int(g.sort_values("oracle_score", ascending=False)["batch"].iloc[0])
        g2 = g[g[map_col] >= b1*(1.0-args.delta_map)].copy()
        if g2.empty: g2 = g
        b_probe = int(g2.sort_values("pwr_mean", ascending=True)["batch"].iloc[0])
        rows.append({"model":m,"tag":t,"batch_probe":b_probe,"batch_oracle":b_orc,"hit":float(b_probe==b_orc)})
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    if len(out)>0:
        print("N=", len(out), "Top1_probe_vs_oracle=", float(out["hit"].mean()))
    else:
        print("N=0")

if __name__=="__main__":
    main()
