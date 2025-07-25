from __future__ import annotations
import os
import sys
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import argparse
import itertools
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
import xgboost as xgb

from experiments import config
from experiments.scoring import add_true_score
from scripts.metrics import top1_accuracy, ndcg_at_1

CANDIDATE_FEATURES = [
    "batch", "throughput", "avg_mem", "pwr_mean", "pwr_std", "energy_per_img", "is65W"
]

def parse_args():
    p = argparse.ArgumentParser("Grid-scan weights (T,P,M,Δ) with ranker CV")
    p.add_argument("--features", default=config.FEATURES_CSV)
    p.add_argument("--map-col", default=config.MAP_COL)
    p.add_argument("--group-cols", nargs="+", default=config.GROUP_COLS)
    p.add_argument("--cv", type=int, default=config.N_FOLDS)
    p.add_argument("--t-grid", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    p.add_argument("--p-grid", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    p.add_argument("--m-grid", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    p.add_argument("--d-grid", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--num_round", type=int, default=300)
    p.add_argument("--early_stop", type=int, default=30)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--out", default="scripts/outputs/weights_model_scan.csv")
    return p.parse_args()

def _dmatrix(X: np.ndarray, y: np.ndarray, group: np.ndarray) -> xgb.DMatrix:
    dmat = xgb.DMatrix(X, label=y)
    _, counts = np.unique(group, return_counts=True)
    dmat.set_group(counts.tolist())
    return dmat

def prepare_matrix(df: pd.DataFrame, feats: list[str],
                   group_cols: list[str]):
    X = df[feats].fillna(0.0).to_numpy(dtype=float)
    y = df["rel"].to_numpy(dtype=int)
    group_key = df[group_cols].astype(str).agg("|".join, axis=1)
    groups = pd.factorize(group_key, sort=False)[0]
    return X, y, groups

def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.features)
    feats = [c for c in CANDIDATE_FEATURES if c in df_raw.columns]

    combos = list(itertools.product(args.t_grid, args.p_grid, args.m_grid, args.d_grid))
    rows = []

    params = dict(
        objective="rank:pairwise",
        eval_metric=["ndcg@1"],
        eta=args.eta,
        max_depth=args.max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=args.seed,
    )

    for T, P, M, D in combos:
        df = add_true_score(
            df_raw, map_col=args.map_col,
            weights=(T, P, M, D),
            group_cols=args.group_cols
        ).copy()

        df["rel"] = (
            df.groupby(args.group_cols, sort=False)["true_score"]
            .rank(method="dense", ascending=True)
            .astype(int) - 1
        ).clip(upper=31)

        X, y, groups = prepare_matrix(df, feats, args.group_cols)

        gkf = GroupKFold(n_splits=args.cv)
        top1_list, ndcg1_list, best_iters = [], [], []

        for tr_idx, va_idx in gkf.split(X, y, groups):
            dtr = _dmatrix(X[tr_idx], y[tr_idx], groups[tr_idx])
            dva = _dmatrix(X[va_idx], y[va_idx], groups[va_idx])

            bst = xgb.train(
                params,
                dtr,
                num_boost_round=args.num_round,
                evals=[(dva, "valid")],
                early_stopping_rounds=args.early_stop,
                verbose_eval=False,
            )

            preds = bst.predict(dva)
            df_fold = df.iloc[va_idx].copy()
            df_fold["pred_score"] = preds

            top1 = top1_accuracy(df_fold)
            ndcg1 = bst.best_score

            top1_list.append(top1)
            ndcg1_list.append(ndcg1)
            best_iters.append(bst.best_iteration)

        rows.append({
            "T": T, "P": P, "M": M, "D": D,
            "features": "|".join(feats),
            "top1_mean": float(np.mean(top1_list)),
            "top1_std": float(np.std(top1_list)),
            "ndcg1_mean": float(np.mean(ndcg1_list)),
            "ndcg1_std": float(np.std(ndcg1_list)),
            "best_iter_mean": float(np.mean(best_iters)),
        })
        print(f"[scan] (T,P,M,D)=({T},{P},{M},{D})  "
              f"Top-1={np.mean(top1_list):.3f}  nDCG@1={np.mean(ndcg1_list):.4f}")

    out_df = pd.DataFrame(rows).sort_values("top1_mean", ascending=False)
    out_df.to_csv(args.out, index=False)
    print(f"[scan] saved → {args.out}")
    print(out_df.head(10))

if __name__ == "__main__":
    main()