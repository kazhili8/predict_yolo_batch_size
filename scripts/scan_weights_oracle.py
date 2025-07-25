import os, sys
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import argparse
import itertools
import pandas as pd
from pathlib import Path
from experiments import config
from experiments.scoring import add_true_score

def parse_args():
    p = argparse.ArgumentParser("Oracle scan for (T,P,M,Δ) without any model")
    p.add_argument("--features", default=config.FEATURES_CSV)
    p.add_argument("--map-col", default=config.MAP_COL)
    p.add_argument("--group-cols", nargs="+", default=config.GROUP_COLS)
    p.add_argument("--t-grid", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    p.add_argument("--p-grid", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    p.add_argument("--m-grid", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    p.add_argument("--d-grid", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    p.add_argument("--out", default="scripts/outputs/weights_oracle_scan.csv")
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.features)

    records = []
    combos = list(itertools.product(args.t_grid, args.p_grid, args.m_grid, args.d_grid))
    base_df = add_true_score(df, map_col=args.map_col,
                             weights=config.DEFAULT_WEIGHTS,
                             group_cols=args.group_cols)
    base_best = (
        base_df.loc[base_df.groupby(args.group_cols)["true_score"].idxmax(), :]
        .set_index(args.group_cols)["batch"]
    )

    for T, P, M, D in combos:
        cur = add_true_score(df, map_col=args.map_col,
                             weights=(T, P, M, D),
                             group_cols=args.group_cols)
        best = (
            cur.loc[cur.groupby(args.group_cols)["true_score"].idxmax(), :]
            .set_index(args.group_cols)["batch"]
        )
        aligned = (best == base_best).sum()
        total = len(best)
        change_rate = 1 - aligned / max(total, 1)

        records.append({
            "T": T, "P": P, "M": M, "D": D,
            "num_groups": total,
            "change_rate_vs_default": change_rate
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"[oracle] saved → {out}  (rows={len(records)})")

if __name__ == "__main__":
    main()