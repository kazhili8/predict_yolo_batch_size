import argparse, subprocess, sys, pandas as pd, numpy as np, re
from pathlib import Path

def parse_candidates(s: str):
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b)+1))
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return sorted(set(out))
def load_recs(path):
    if not path: return {}
    p = Path(path)
    if not p.exists(): return {}
    df = pd.read_csv(p)
    mdl = "model" if "model" in df.columns else None
    epc = "epochs" if "epochs" in df.columns else None
    tag = "tag"   if "tag"   in df.columns else None
    bch = "batch" if "batch" in df.columns else ("recommended_batch" if "recommended_batch" in df.columns else None)
    if not mdl or not bch: return {}
    out = {}
    for _, r in df.iterrows():
        k = (str(r[mdl]), int(r[epc]) if epc in df.columns else 1, str(r[tag]) if tag in df.columns else "NA")
        out[k] = int(r[bch])
    return out

def ensure_throughput(df):
    if "throughput" not in df.columns and "avg_step_time" in df.columns:
        df = df.copy()
        df["throughput"] = 1.0 / df["avg_step_time"].astype(float)
    return df

def feasible_mask(g, cap_w, delta_map):
    b1 = g[g["batch"] == 1]
    if b1.empty: return pd.Series(False, index=g.index)
    thr = float(b1["map50"].max()) * (1.0 - delta_map)
    ok_map = g["map50"] >= thr
    ok_pwr = True if cap_w <= 0 else (g["avg_power"] <= cap_w)
    return ok_map & ok_pwr

def true_score(g, w):
    T, P, M, D = w
    dm = g["map50"].max() - g["map50"]
    return T * g["throughput"] - P * g["avg_power"] - M * g["avg_mem"] - D * dm

def pick_oracle(g, oracle, w, feas, candmask=None):
    gg = g[feas] if feas.any() else g
    if candmask is not None:
        gg = gg[gg["batch"].isin(candmask)]
        if gg.empty:
            gg = g[g["batch"].isin(candmask)]
            if gg.empty:
                gg = g
    if oracle == "throughput":
        i = int(gg["throughput"].idxmax())
        return int(g.loc[i, "batch"])
    sv = true_score(gg, w)
    j = int(sv.idxmax())
    return int(g.loc[j, "batch"])

def regret_of(g, w, best_b, chosen_b):
    sv = true_score(g, w)
    sv_best = float(sv.loc[g["batch"] == best_b].iloc[0])
    sv_ch = float(sv.loc[g["batch"] == chosen_b].iloc[0]) if (g["batch"] == chosen_b).any() else -np.inf
    if sv_best <= 0:
        return float(0.0 if sv_ch >= sv_best else 1.0)
    r = max(0.0, (sv_best - sv_ch) / abs(sv_best))
    return float(min(r, 1.0))

def violated(g, feas, b):
    if (g["batch"] == b).sum() == 0: return 1
    return int((~feas).loc[g["batch"] == b].iloc[0])

def _parse_last_int(s):
    lines = s.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.isdigit():
            return int(line)
        m = re.search(r'(\d+)\s*$', line)
        if m:
            return int(m.group(1))
    raise RuntimeError("no integer found in output")

def run_probe(predict_py, model, tag, rec65, rec115):
    cmd = [
        sys.executable, predict_py,
        "--model", model, "--tag", tag,
        "--rec65", rec65, "--rec115", rec115,
        "--force_probe", "1", "--print_chosen_only", "1"
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
        return _parse_last_int(out)
    except Exception:
        cmd = [
            sys.executable, predict_py,
            "--model", model, "--tag", tag,
            "--rec65", rec65, "--rec115", rec115,
            "--print_chosen_only", "1"
        ]
        try:
            out = subprocess.check_output(cmd, text=True)
            return _parse_last_int(out)
        except Exception:
            return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--rec65", required=True)
    ap.add_argument("--rec115", required=True)
    ap.add_argument("--predict_py", required=True)
    ap.add_argument("--tag", choices=["65W", "115W"], required=True)
    ap.add_argument("--oracle", choices=["score", "throughput"], default="score")
    ap.add_argument("--weights", nargs=4, type=float, default=[0.6, 0.2, 0.1, 0.1])
    ap.add_argument("--cap_w", type=float, required=True)
    ap.add_argument("--delta_map", type=float, default=0.01)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--force_all", type=int, default=0)
    ap.add_argument("--candidates", default="2,4,8,16")
    args = ap.parse_args()

    f = pd.read_csv(args.features_csv)
    f = ensure_throughput(f)

    rec = {} if args.force_all == 1 else load_recs(args.rec65 if args.tag == "65W" else args.rec115)

    rows = []
    n = n_hit = n_vio0 = n_reg0 = 0
    for (m, e, t), g0 in f.groupby(["model", "epochs", "tag"], sort=False):
        if int(e) != 1 or str(t) != args.tag:
            continue
        g = g0.sort_values("batch").copy()
        feas = feasible_mask(g, args.cap_w, args.delta_map)
        C = parse_candidates(args.candidates)
        b_orc = pick_oracle(g, args.oracle, tuple(args.weights), feas, C)
        k = (str(m), int(e), str(t))
        b_tbl = rec.get(k, None)
        if args.force_all != 1 and b_tbl is not None and b_tbl == b_orc:
            continue
        b_prb = run_probe(args.predict_py, str(m), args.tag, args.rec65, args.rec115)
        if b_prb is None:
            continue
        hit = int(b_prb == b_orc)
        vio = violated(g, feas, b_prb)
        g_cand = g[g["batch"].isin(C)] if len(C) > 0 else g
        reg = regret_of(g_cand, tuple(args.weights), b_orc, b_prb)
        rows.append({
            "model": str(m), "epochs": int(e), "tag": str(t),
            "oracle_batch": int(b_orc), "table_batch": (int(b_tbl) if b_tbl is not None else None),
            "probe_batch": int(b_prb), "hit": int(hit), "vio": int(vio), "regret": float(reg)
        })
        n += 1; n_hit += hit; n_vio0 += int(vio == 0); n_reg0 += int(reg == 0.0)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    if n == 0:
        print("N=0")
    else:
        print(f"N={n}  Top1={n_hit/n:.3f}  Violation-free={n_vio0/n:.3f}  Zero-regret={n_reg0/n:.3f}")

if __name__ == "__main__":
    main()
