import argparse, pandas as pd, numpy as np
from pathlib import Path
def feasible(g, delta_map, cap_w):
    b1 = g[g["batch"] == 1]
    if len(b1) == 0:
        g["feasible"] = False
        return g
    base = float(b1["map50"].iloc[0])
    g = g.assign(
        feasible=(g["map50"] >= base * (1.0 - delta_map)) & (g["avg_power"] <= cap_w)
    )
    return g

def best_batch(g, oracle, w):
    gg = g[g["feasible"]]
    if len(gg) == 0:
        gg = g
    if oracle == "throughput":
        i = gg["throughput"].values.argmax()
        return int(gg.iloc[i]["batch"])
    dm = gg["map50"].max() - gg["map50"]
    sc = w[0] * gg["throughput"] - w[1] * gg["avg_power"] - w[2] * gg["avg_mem"] - w[3] * dm
    j = int(np.argmax(sc.values))
    return int(gg.iloc[j]["batch"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--cap_w", type=float, required=True)
    ap.add_argument("--delta_map", type=float, default=0.01)
    ap.add_argument("--oracle", choices=["throughput", "score"], default="score")
    ap.add_argument("--weights", nargs=4, type=float, default=[0.6, 0.2, 0.1, 0.1])
    ap.add_argument("--epochs", nargs="+", type=int, default=[1, 5, 10, 20])
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary", default="")
    args = ap.parse_args()

    f = pd.read_csv(args.features_csv)
    if "throughput" not in f.columns and "avg_step_time" in f.columns:
        f["throughput"] = 1.0 / f["avg_step_time"].astype(float)

    rows = []
    for (m, t), gmt in f.groupby(["model", "tag"], sort=False):
        best = {}
        for e in args.epochs:
            g = gmt[gmt["epochs"] == e].copy()
            if g.empty:
                best[e] = None
                continue
            g = feasible(g, args.delta_map, args.cap_w)
            b = best_batch(g, args.oracle, args.weights)
            best[e] = b
        base_e = min(args.epochs)
        base_b = best.get(base_e)
        for e in args.epochs:
            if e == base_e:
                continue
            rows.append({
                "model": m, "tag": t,
                "e_base": base_e, "b_base": base_b,
                "e_other": e, "b_other": best.get(e),
                "same": float(base_b == best.get(e))
            })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    if df.empty:
        print("No comparable entries for the given epochs/cap_w.")
        return
    summary = (
        df.groupby(["tag", "e_base", "e_other"])
          .agg(n=("same", "size"), pct_same=("same", "mean"))
          .reset_index()
    )
    for _, r in summary.iterrows():
        print(f"[{r['tag']}] {int(r['e_base'])}→{int(r['e_other'])}  "
              f"same_best_batch = {r['pct_same']*100:.1f}%  (n={int(r['n'])})")

    out_sum = args.out_summary
    if not out_sum:
        out_sum = str(out_path.with_name(out_path.stem + "_summary.csv"))
    Path(out_sum).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_sum, index=False)

if __name__ == "__main__":
    main()
