import argparse, numpy as np, pandas as pd, pathlib

def parse_batches(s):
    if s == "auto": return None
    xs = []
    for t in s.split(","):
        t=t.strip()
        if t: xs.append(int(t))
    xs = sorted(set(xs))
    return xs or None

def true_score(g, w):
    T,P,M,D = w
    mmax = g["map50"].max()
    return T*g["throughput"] - P*g["avg_power"] - M*g["avg_mem"] - D*(mmax - g["map50"])

def pick_oracle(g, w, cap_w, delta_map):
    gg = g.copy()
    if cap_w>0: gg = gg[gg["avg_power"]<=cap_w]
    if gg.empty: return None
    b1 = gg.loc[gg["batch"]==1]
    if b1.empty or b1["map50"].isna().all(): return None
    thr = float(b1["map50"].max())*(1.0-delta_map)
    feas = gg[gg["map50"]>=thr].copy()
    if feas.empty: feas = gg
    feas = feas.assign(ts=true_score(feas,w))
    i = int(feas["ts"].idxmax())
    return int(gg.loc[i,"batch"])

def pick_oracle_tp(g, cap_w, delta_map):
    gg = g.copy()
    if cap_w>0: gg = gg[gg["avg_power"]<=cap_w]
    if gg.empty: return None
    b1 = gg.loc[gg["batch"]==1]
    if b1.empty or b1["map50"].isna().all(): return None
    thr = float(b1["map50"].max())*(1.0-delta_map)
    feas = gg[gg["map50"]>=thr].copy()
    if feas.empty: feas = gg
    i = int(feas["throughput"].idxmax())
    return int(gg.loc[i,"batch"])

def pick_probe(g, delta_map, cand):
    b1 = g.loc[g["batch"]==1]
    if b1.empty or b1["map50"].isna().all(): return None
    thr = float(b1["map50"].max())*(1.0-delta_map)
    gg = g.copy()
    if cand is not None:
        gg = gg[gg["batch"].isin(cand)]
        if gg.empty: return None
    feas = gg[gg["map50"]>=thr].copy()
    if feas.empty: feas = gg
    i = int(feas["avg_power"].idxmin())
    return int(gg.loc[i,"batch"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--cap_w", type=float, default=0)
    ap.add_argument("--delta_map", type=float, default=0.01)
    ap.add_argument("--weights", nargs=4, type=float, default=[0.6,0.2,0.1,0.1])
    ap.add_argument("--group_cols", nargs="+", default=["model","epochs","tag"])
    ap.add_argument("--probe_candidates", default="2,4,8,16")
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)
    if "throughput" not in df.columns and "avg_step_time" in df.columns:
        df["throughput"] = 1.0/df["avg_step_time"].astype(float)
    cols = ["batch","throughput","avg_power","avg_mem","map50"] + args.group_cols
    df = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
    cand = parse_batches(args.probe_candidates)

    rows=[]
    hit_score=hit_tp=n=0
    for k,g in df.groupby(args.group_cols, dropna=False):
        g = g.sort_values("batch")
        if not set(["batch","throughput","avg_power","avg_mem","map50"]).issubset(g.columns): continue
        b_orc = pick_oracle(g, tuple(args.weights), args.cap_w, args.delta_map)
        b_tp  = pick_oracle_tp(g, args.cap_w, args.delta_map)
        b_prb = pick_probe(g, args.delta_map, cand)
        if b_orc is None or b_tp is None or b_prb is None: continue
        n+=1
        hit_score += int(b_prb==b_orc)
        hit_tp    += int(b_prb==b_tp)
        rows.append({"group":"|".join(map(str,k)),"probe":b_prb,"oracle_score":b_orc,"oracle_tp":b_tp})
    if n==0:
        print("N=0")
        return
    acc_score = hit_score/n
    acc_tp    = hit_tp/n
    print(f"N= {n}  Top1_probe_vs_score_oracle= {acc_score:.3f}  Top1_probe_vs_tp_oracle= {acc_tp:.3f}")
    if args.out_csv:
        pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

if __name__=="__main__":
    main()
