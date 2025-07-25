from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Dict, Tuple

GROUP_COLS = ["model", "epochs", "tag"]

def top1_accuracy(df: pd.DataFrame,
                  group_cols: Iterable[str] = GROUP_COLS,
                  true_col: str = "true_score",
                  pred_col: str = "pred_score") -> float:
    hit, total = 0, 0
    for _, g in df.groupby(list(group_cols), sort=False):
        if g.empty:
            continue
        gt_idx = g[true_col].idxmax()
        pd_idx = g[pred_col].idxmax()
        hit += int(gt_idx == pd_idx)
        total += 1
    return hit / max(total, 1)

def ndcg_at_1(df: pd.DataFrame,
              group_cols: Iterable[str] = GROUP_COLS,
              rel_col: str = "rel",
              pred_col: str = "pred_score") -> float:
    ndcgs, n_groups = 0.0, 0
    for _, g in df.groupby(list(group_cols), sort=False):
        if g.empty:
            continue
        ideal = g[rel_col].max()
        if ideal <= 0:
            continue
        best_pred_idx = g[pred_col].idxmax()
        ndcgs += g.loc[best_pred_idx, rel_col] / ideal
        n_groups += 1
    return ndcgs / max(n_groups, 1)

def mae_best_batch_index(df: pd.DataFrame,
                         group_cols: Iterable[str] = GROUP_COLS,
                         true_col: str = "true_score",
                         pred_col: str = "pred_score") -> float:
    diffs, n = [], 0
    for _, g in df.groupby(list(group_cols), sort=False):
        g = g.sort_values("batch").reset_index(drop=True)
        if g.empty:
            continue
        true_best_batch = g.loc[g[true_col].idxmax(), "batch"]
        pred_best_batch = g.loc[g[pred_col].idxmax(), "batch"]
        diffs.append(abs(int(true_best_batch) - int(pred_best_batch)))
        n += 1
    return float(np.mean(diffs)) if n > 0 else np.nan