import argparse, pandas as pd, numpy as np

def load_features(path):
    df = pd.read_csv(path)
    need = ["model","epochs","tag","batch","map50","pwr_mean"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"missing column {c} in {path}")
    if "throughput" in df.columns:
        thr = df["throughput"].astype(float)
    else:
        if "step_time_mean" not in df.columns:
            raise ValueError("need either throughput or step_time_mean")
        thr = 1.0 / df["step_time_mean"].astype(float)
    df = df.assign(thr=thr)
    return df

def load_recs(path):
    df = pd.read_csv(path)
    cols = list(df.columns)
    km = "model" if "model" in cols else None
    ke = "epochs" if "epochs" in cols else None
    kt = "tag" if "tag" in cols else None
    kb = "recommended_batch" if "recommended_batch" in cols else ("batch" if "batch" in cols else None)
    if not (km and ke and kt and kb):
        return {}
    rec = {}
    for _,r in df.iterrows():
        rec[(str(r[km]), int(r[ke]), str(r[kt]))] = int(r[kb])
    return rec

def feasible(g, delta_map, cap_w):
    b1 = g[g["batch"]==1]
    if len(b1)==0:
        return g.assign(feasible=False)
    base = float(b1["map50"].iloc[0])
    min_map = base*(1.0-delta_map)
    ok = (g["map50"]>=min_map) & (g["pwr_mean"]<=cap_w)
    return g.assign(feasible=ok)

def oracle_thr(g):
    gg = g[g["feasible"]]
    if len(gg)==0:
        return None
    i = gg["thr"].values.argmax()
    return int(gg.iloc[i]["batch"])

def oracle_score(g, wT, wP, wM, wD):
    gg = g[g["feasible"]]
    if len(gg)==0:
        return None
    dm = gg["map50"].max() - gg["map50"]
    sc = wT*gg["thr"] - wP*gg["pwr_mean"] - wM*gg["avg_mem"] - wD*dm
    j = int(np.argmax(sc.values))
    return int(gg.iloc[j]["batch"])

def pick_zeus(g):
    return oracle_thr(g)

def pick_greedy(g):
    gg = g[g["feasible"]].sort_values("batch", ascending=False)
    if len(gg)==0:
        return None
    return int(gg.iloc[0]["batch"])

def pick_random(g, rnd):
    gg = g[g["feasible"]]
    if len(gg)==0:
        return None
    j = rnd.integers(0, len(gg))
    return int(gg.iloc[j]["batch"])

def pick_table(g, rec):
    k = (str(g["model"].iloc[0]), int(g["epochs"].iloc[0]), str(g["tag"].iloc[0]))
    return rec.get(k, None)

def regret(g, chosen, oracle_batch):
    if oracle_batch is None or chosen is None:
        return np.nan
    ot = float(g[g["batch"]==oracle_batch]["thr"].iloc[0])
    ct = float(g[g["batch"]==chosen]["thr"].iloc[0])
    if ot <= 0:
        return np.nan
    return (ot-ct)/ot

def violated(g, chosen, delta_map, cap_w):
    if chosen is None:
        return np.nan
    row = g[g["batch"]==chosen]
    if len(row)==0:
        return np.nan
    b1 = g[g["batch"]==1]
    if len(b1)==0:
        return np.nan
    base = float(b1["map50"].iloc[0])
    ok_map = float(row["map50"].iloc[0]) >= base*(1.0-delta_map)
    ok_pwr = float(row["pwr_mean"].iloc[0]) <= cap_w
    return 0.0 if (ok_map and ok_pwr) else 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--recs_csv", required=True)
    ap.add_argument("--cap_w", type=float, required=True)
    ap.add_argument("--delta_map", type=float, default=0.01)
    ap.add_argument("--oracle", choices=["throughput","score"], default="throughput")
    ap.add_argument("--weights", nargs=4, type=float, default=[0.6,0.2,0.1,0.1])
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    f = load_features(args.features_csv)
    rec = load_recs(args.recs_csv)
    rnd = np.random.default_rng(args.seed)
    wT,wP,wM,wD = args.weights

    rows = []
    for (m,e,t), g0 in f.groupby(["model","epochs","tag"], sort=False):
        g = feasible(g0.copy(), args.delta_map, args.cap_w)
        if args.oracle == "throughput":
            o = oracle_thr(g)
        else:
            o = oracle_score(g, wT, wP, wM, wD)

        z  = pick_zeus(g)
        gm = pick_greedy(g)
        rd = pick_random(g, rnd)
        tb = pick_table(g, rec)

        reg_z  = regret(g, z,  o)
        reg_gm = regret(g, gm, o)
        reg_rd = regret(g, rd, o)
        reg_tb = regret(g, tb, o)

        t1_z  = int(z  == o) if (o is not None and z  is not None) else 0
        t1_gm = int(gm == o) if (o is not None and gm is not None) else 0
        t1_rd = int(rd == o) if (o is not None and rd is not None) else 0
        t1_tb = int(tb == o) if (o is not None and tb is not None) else 0

        vio_z  = violated(g, z,  args.delta_map, args.cap_w)
        vio_gm = violated(g, gm, args.delta_map, args.cap_w)
        vio_rd = violated(g, rd, args.delta_map, args.cap_w)
        vio_tb = violated(g, tb, args.delta_map, args.cap_w)

        rows.append({
            "model":m,"epochs":int(e),"tag":t,
            "oracle_batch":o,"zeus_batch":z,"greedy_batch":gm,"random_batch":rd,"table_batch":tb,
            "regret_zeus":reg_z,"regret_greedy":reg_gm,"regret_random":reg_rd,"regret_table":reg_tb,
            "top1_zeus":t1_z,"top1_greedy":t1_gm,"top1_random":t1_rd,"top1_table":t1_tb,
            "vio_zeus":vio_z,"vio_greedy":vio_gm,"vio_random":vio_rd,"vio_table":vio_tb
        })

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
