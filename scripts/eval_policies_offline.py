import argparse, pathlib, numpy as np, pandas as pd, random

def read_csv(fp):
    return pd.read_csv(fp)

def maybe_add_throughput(df):
    if "throughput" not in df.columns:
        if "avg_step_time" in df.columns:
            df = df.copy()
            df["throughput"] = 1.0 / df["avg_step_time"].astype(float)
    return df

def load_recs(rec_csv):
    if not rec_csv:
        return {}
    p = pathlib.Path(rec_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    mdl = "model" if "model" in df.columns else None
    epc = "epochs" if "epochs" in df.columns else None
    tag = "tag" if "tag" in df.columns else None
    bch = "batch" if "batch" in df.columns else ("recommended_batch" if "recommended_batch" in df.columns else None)
    if not mdl or not bch:
        return {}
    out = {}
    for _,r in df.iterrows():
        k = (str(r[mdl]), int(r[epc]) if epc in df.columns else 1, str(r[tag]) if tag in df.columns else "NA")
        out[k] = int(r[bch])
    return out

def feasible_mask(g, cap_w, delta_map):
    bb = g[g["batch"]==1]
    if bb.empty:
        return pd.Series(False, index=g.index)
    thr = float(bb["map50"].max())*(1.0 - delta_map)
    ok_map = g["map50"] >= thr
    ok_pwr = True if cap_w<=0 else (g["avg_power"]<=cap_w)
    return ok_map & ok_pwr

def true_score_view(g, w):
    T,P,M,D = w
    dm = g["map50"].max() - g["map50"]
    return T*g["throughput"] - P*g["avg_power"] - M*g["avg_mem"] - D*dm

def pick_oracle(g, oracle, w, feas):
    gg = g[feas]
    if gg.empty:
        gg = g
    if oracle=="throughput":
        i = int(gg["throughput"].idxmax())
        return int(g.loc[i,"batch"])
    sv = true_score_view(gg, w)
    j = int(sv.idxmax())
    return int(g.loc[j,"batch"])

def eval_one_group(g, table_rec, zeus_rec, oracle, w, cap_w, delta_map, rng):
    feas = feasible_mask(g, cap_w, delta_map)
    b_orc = pick_oracle(g, oracle, w, feas)

    def regret_of(b):
        gg = g.copy()
        sv = true_score_view(gg, w)
        sv_orc = float(sv.loc[g["batch"]==b_orc].iloc[0])
        sv_b   = float(sv.loc[g["batch"]==b].iloc[0]) if (g["batch"]==b).any() else -np.inf
        if sv_orc <= 0:
            return float(0.0 if sv_b>=sv_orc else 1.0)
        r = max(0.0, (sv_orc - sv_b)/abs(sv_orc))
        return float(min(r, 1.0))

    def violated(b):
        if (g["batch"]==b).sum()==0:
            return 1
        return int((~feas).loc[g["batch"]==b].iloc[0])

    # table policy
    k = (str(g["model"].iloc[0]), int(g["epochs"].iloc[0]), str(g["tag"].iloc[0]))
    b_table = table_rec.get(k, None)

    # zeus policy
    b_zeus = zeus_rec.get(k, None)

    # greedy: min power among feasible; fallback to global min power
    if feas.any():
        i = int(g.loc[feas, "avg_power"].idxmin())
        b_greedy = int(g.loc[i,"batch"])
    else:
        j = int(g["avg_power"].idxmin())
        b_greedy = int(g.loc[j,"batch"])

    # random: uniform over feasible; fallback to all
    choices = g.loc[feas, "batch"].tolist() if feas.any() else g["batch"].tolist()
    b_random = int(rng.choice(choices))

    rows = {
        "model": g["model"].iloc[0],
        "epochs": int(g["epochs"].iloc[0]),
        "tag": g["tag"].iloc[0],
        "oracle_batch": b_orc,
        "table_batch": b_table,
        "zeus_batch": b_zeus,
        "greedy_batch": b_greedy,
        "random_batch": b_random,
        "top1_table": float(b_table==b_orc) if b_table is not None else 0.0,
        "top1_zeus": float(b_zeus==b_orc) if b_zeus is not None else 0.0,
        "top1_greedy": float(b_greedy==b_orc),
        "top1_random": float(b_random==b_orc),
        "regret_table": regret_of(b_table) if b_table is not None else 1.0,
        "regret_zeus": regret_of(b_zeus) if b_zeus is not None else 1.0,
        "regret_greedy": regret_of(b_greedy),
        "regret_random": regret_of(b_random),
        "vio_table": float(violated(b_table)) if b_table is not None else 1.0,
        "vio_zeus": float(violated(b_zeus)) if b_zeus is not None else 1.0,
        "vio_greedy": float(violated(b_greedy)),
        "vio_random": float(violated(b_random)),
    }
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--recs_csv", required=True)
    ap.add_argument("--zeus_csv", default="")
    ap.add_argument("--cap_w", type=float, required=True)
    ap.add_argument("--delta_map", type=float, default=0.01)
    ap.add_argument("--oracle", choices=["score","throughput"], default="score")
    ap.add_argument("--weights", nargs=4, type=float, default=[0.6,0.2,0.1,0.1])
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = read_csv(args.features_csv)
    df = maybe_add_throughput(df)
    need = ["model","epochs","tag","batch","avg_power","avg_mem","map50","throughput"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing columns: {missing}")

    table_rec = load_recs(args.recs_csv)
    zeus_rec  = load_recs(args.zeus_csv)
    rng = np.random.default_rng(2025)
    W = tuple(args.weights)

    rows=[]
    for k,g in df.groupby(["model","epochs","tag"], sort=False):
        g = g.sort_values("batch")
        rows.append(eval_one_group(g, table_rec, zeus_rec, args.oracle, W, args.cap_w, args.delta_map, rng))
    out = pd.DataFrame(rows)
    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[done] wrote {args.out_csv}  n_groups={len(out)}")

if __name__ == "__main__":
    main()
