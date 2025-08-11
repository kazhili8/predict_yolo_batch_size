import argparse, pandas as pd, numpy as np
from pathlib import Path

def pick_map_col(df):
    for c in ["map50","avg_map50","mAP50","mAP@0.5"]:
        if c in df.columns: return c
    raise RuntimeError("no mAP column")

def make_true_score(df, weights, oracle, map_col):
    if oracle=="throughput":
        s = df["throughput"].astype(float)
        return s
    T,P,M,A = weights
    dmap = df[map_col].max() - df[map_col]
    s = T*df["throughput"] - P*df["avg_power"] - M*df["avg_mem"] - A*dmap
    return s

def feasible_mask(df, cap_w, delta_map, map_col):
    if cap_w is None: return pd.Series(True, index=df.index)
    b1 = df[df["batch"]==1]
    if len(b1)==0: return pd.Series(False, index=df.index)
    base = float(b1[map_col].iloc[0])
    return (df["pwr_mean"]<=cap_w) & (df[map_col] >= base*(1.0-delta_map))

def topk_set(df, score_col, k):
    x = df.sort_values(score_col, ascending=False)["batch"].astype(int).tolist()
    return set(x[:min(k,len(x))])

def spearman(a, b):
    if len(a)<2: return np.nan
    ra = pd.Series(a).rank(method="average")
    rb = pd.Series(b).rank(method="average")
    da = ra - ra.mean()
    db = rb - rb.mean()
    num = float((da*db).sum())
    den = float(np.sqrt((da*da).sum())*np.sqrt((db*db).sum()))
    return np.nan if den==0.0 else num/den

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--tag", nargs="*", default=[])
    ap.add_argument("--ref_epoch", type=int, default=1)
    ap.add_argument("--tgt_epochs", nargs="+", type=int, required=True)
    ap.add_argument("--cap_w", type=float, default=None)
    ap.add_argument("--delta_map", type=float, default=0.01)
    ap.add_argument("--weights", nargs=4, type=float, default=[0.60,0.20,0.10,0.10])
    ap.add_argument("--oracle", choices=["score","throughput"], default="score")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)
    if args.tag:
        df = df[df["tag"].astype(str).isin(args.tag)].copy()
    map_col = pick_map_col(df)
    rows = []

    for (m,t), gmt in df.groupby(["model","tag"], sort=False):
        gre = gmt[gmt["epochs"]==args.ref_epoch].copy()
        if gre.empty: continue
        fm_re = feasible_mask(gre, args.cap_w, args.delta_map, map_col)
        gre = gre[fm_re].copy()
        if gre.empty: continue
        gre["score"] = make_true_score(gre, args.weights, args.oracle, map_col)
        if gre.empty: continue
        best_re = int(gre.sort_values("score", ascending=False)["batch"].iloc[0])

        for e in args.tgt_epochs:
            gte = gmt[gmt["epochs"]==e].copy()
            if gte.empty:
                rows.append({"model":m,"tag":t,"ref":args.ref_epoch,"tgt":e,"hit":np.nan,"rho":np.nan,"top3_overlap":np.nan,"n_common":0})
                continue
            fm_tg = feasible_mask(gte, args.cap_w, args.delta_map, map_col)
            gte = gte[fm_tg].copy()
            if gte.empty:
                rows.append({"model":m,"tag":t,"ref":args.ref_epoch,"tgt":e,"hit":0.0,"rho":np.nan,"top3_overlap":0.0,"n_common":0})
                continue
            gte["score"] = make_true_score(gte, args.weights, args.oracle, map_col)
            best_tg = int(gte.sort_values("score", ascending=False)["batch"].iloc[0])

            common = pd.merge(gre[["batch","score"]].rename(columns={"score":"s_re"}),
                              gte[["batch","score"]].rename(columns={"score":"s_tg"}),
                              on="batch", how="inner")
            r = spearman(common["s_re"].values, common["s_tg"].values) if len(common)>=2 else np.nan
            o = len(topk_set(gre,"score",3).intersection(topk_set(gte,"score",3)))/3.0
            rows.append({"model":m,"tag":t,"ref":args.ref_epoch,"tgt":e,"hit":float(best_re==best_tg),"rho":r,"top3_overlap":o,"n_common":int(len(common))})

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    for e in args.tgt_epochs:
        sub = out[out["tgt"]==e]
        sub = sub[np.isfinite(sub["hit"])]
        if len(sub)==0:
            print(f"epoch {args.ref_epoch}→{e}: N=0")
        else:
            print(f"epoch {args.ref_epoch}→{e}: N={len(sub)}  Top1_stability={sub['hit'].mean():.3f}  Spearman={sub['rho'].mean():.3f}  Top3_overlap={sub['top3_overlap'].mean():.3f}")

if __name__=="__main__":
    main()
