import argparse
import numpy as np
import pandas as pd
from itertools import product
from experiments.config import FEATURES_CSV, GROUP_COLS, MAP_COL
from experiments.scoring import add_true_score

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=FEATURES_CSV)
    p.add_argument("--group-cols", nargs="+", default=GROUP_COLS)
    p.add_argument("--map-col", default=MAP_COL)
    p.add_argument("--step", type=float, default=0.1, help="grid step, e.g. 0.1")
    return p.parse_args()

def top1_acc(df, group_cols):
    hit = 0
    total = 0
    for _, sub in df.groupby(group_cols, sort=False):
        gt_idx = sub["true_score"].idxmax()
        pred_idx = gt_idx
        hit += int(gt_idx == pred_idx)
        total += 1
    return hit / max(total, 1)

def main():
    args = parse_args()
    df = pd.read_csv(args.features)

    best = (-1, None)
    step = args.step
    w_list = np.arange(0, 1 + 1e-9, step)

    for T, P, M in product(w_list, repeat=3):
        D = 1 - (T + P + M)
        if D < -1e-9:
            continue
        w = (T, P, M, D)
        df_scored = add_true_score(df, map_col=args.map_col, weights=w, group_cols=args.group_cols)
        acc = top1_acc(df_scored, args.group_cols)
        if acc > best[0]:
            best = (acc, w)

    print(f"Best acc={best[0]:.3f} at weights={best[1]}")

if __name__ == "__main__":
    main()
