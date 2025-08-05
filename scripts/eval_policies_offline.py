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

def build_table_rec(path):
    if path is None:
        return {}
    df = pd.read_csv(path)
    cols = list(df.columns)
    key_m = "model" if "model" in cols else None
    key_e = "epochs" if "epochs" in cols else None
    key_t = "tag" if "tag" in cols else None
    key_b = "recommended_batch" if "recommended_batch" in cols else ("batch" if "batch" in cols else None)
    if not (key_m and key_e and key_t and key_b):
        return {}
    rec = {}
    for _,r in df.iterrows():
        rec[(str(r[key_m]), int(r[key_e]), str(r[key_t]))] = int(r[key_b])
    return rec

def feasible_set(g, delta_map, cap_w):
    b1 = g[g["batch"]==1]
    if len(b1)==0:
        return g.assign(feasible=False)
    base_map = float(b1["map50"].iloc[0])
    map_min = base_map * (1.0 - delta_map)
    feas = (g["map50"] >= map_min) & (g["pwr_mean"] <= cap_w)
    return g.assign(feasible=feas)

def pick_oracle(g):
    gg = g[g["feasible"]]
    if len(gg)==0:
        return None
    i = gg["thr"].values.argmax()
    return int(gg.iloc[i]["batch"])

def pick_zeus_like(g):
    return pick_oracle(g)

def pick_greedy_mem(g):
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

def regret_vs_oracle(g, chosen):
    o = pick_oracle(g)
    if o is None or chosen is None:
        return np.nan, o
    thr_o = float(g[g["batch"]==o]["thr"].iloc[0])
    thr_c = float(g[g["batch"]==chosen]["thr"].iloc[0])
    if thr_o <= 0:
        return np.nan, o
    reg = (thr_o - thr_c) / thr_o
    return reg, o

def violated(g, chosen, delta_map, cap_w):
    if chosen is None:
        return True
    b1 = g[g["batch"]==1]
    if len(b1)==0:
        return True
    base_map = float(b1["map50"].iloc[0])
    map_min = base_map * (1.0 - delta_map)
    row = g[g["batch"]==chosen]
    if len(row)==0:
        return True
    ok = (float(row["map50"].iloc[0]) >= map_min) and (float(row["pwr_mean"].iloc[0]) <= cap_w)
    return (not ok)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--recs_csv", required=True)
    ap.add_argument("--cap_w", type=float, required=True)
    ap.add_argument("--delta_map", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = load_features(args.features_csv)
    rec = build_table_rec(args.recs_csv)
    rnd = np.random.default_rng(args.seed)

    rows = []
    for (m,e,t), g0 in df.groupby(["model","epochs","tag"], sort=False):
        g = feasible_set(g0.copy(), args.delta_map, args.cap_w)
        o = pick_oracle(g)
        z = pick_zeus_like(g)
        gm = pick_greedy_mem(g)
        rd = pick_random(g, rnd)
        tb = pick_table(g, rec)

        reg_o, _ = regret_vs_oracle(g, o)
        reg_z, _ = regret_vs_oracle(g, z)
        reg_gm,_ = regret_vs_oracle(g, gm)
        reg_rd,_ = regret_vs_oracle(g, rd)
        reg_tb,_ = regret_vs_oracle(g, tb)

        top1_z  = int(z==o) if o is not None and z is not None else 0
        top1_gm = int(gm==o) if o is not None and gm is not None else 0
        top1_rd = int(rd==o) if o is not None and rd is not None else 0
        top1_tb = int(tb==o) if o is not None and tb is not None else 0

        vio_z  = violated(g, z,  args.delta_map, args.cap_w)
        vio_gm = violated(g, gm, args.delta_map, args.cap_w)
        vio_rd = violated(g, rd, args.delta_map, args.cap_w)
        vio_tb = violated(g, tb, args.delta_map, args.cap_w)

        rows.append({
            "model":m,"epochs":int(e),"tag":t,
            "oracle_batch":o,"zeus_batch":z,"greedy_batch":gm,"random_batch":rd,"table_batch":tb,
            "regret_zeus":reg_z,"regret_greedy":reg_gm,"regret_random":reg_rd,"regret_table":reg_tb,
            "top1_zeus":top1_z,"top1_greedy":top1_gm,"top1_random":top1_rd,"top1_table":top1_tb,
            "vio_zeus":vio_z,"vio_greedy":vio_gm,"vio_random":vio_rd,"vio_table":vio_tb
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
