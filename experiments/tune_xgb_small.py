import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold

import config

def _dmatrix(X: np.ndarray, y: np.ndarray, group: np.ndarray) -> xgb.DMatrix:
    dmat = xgb.DMatrix(X, label=y)
    _, counts = np.unique(group, return_counts=True)
    dmat.set_group(counts.tolist())
    return dmat

def _top1_accuracy(df_fold) -> float:
    """Per-group Top-1 accuracy using true_score vs pred_score."""
    hit = total = 0
    for _, g in df_fold.groupby(["model", "epochs", "tag"], sort=False):
        gt_idx = g["true_score"].idxmax()
        pred_idx = g["pred_score"].idxmax()
        hit += int(gt_idx == pred_idx)
        total += 1
    return hit / max(total, 1)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small hyperparameter search for XGB ranker")
    p.add_argument("--data", default="scripts/outputs/rank_data_v3.pkl",
                   help="Path to bundle produced by make_rank_data.py")
    p.add_argument("--cv", type=int, default=config.N_FOLDS,
                   help="Requested CV folds; will be capped by #groups.")
    p.add_argument("--num_round", type=int, default=300)
    p.add_argument("--early_stop", type=int, default=30)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--max_trials", type=int, default=36,
                   help="Max combinations to try (randomly sampled from grid)")
    p.add_argument("--out_dir", default="scripts/outputs/tune_small",
                   help="Where to save results and the best model")
    p.add_argument("--eta_grid", nargs="+", type=float, default=[0.05, 0.1])
    p.add_argument("--max_depth_grid", nargs="+", type=int, default=[4, 6, 8])
    p.add_argument("--subsample_grid", nargs="+", type=float, default=[0.8, 1.0])
    p.add_argument("--colsample_bytree_grid", nargs="+", type=float, default=[0.8, 1.0])
    p.add_argument("--min_child_weight_grid", nargs="+", type=float, default=[1.0, 5.0])
    p.add_argument("--reg_lambda_grid", nargs="+", type=float, default=[1.0, 5.0])
    return p.parse_args()

def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.data)
    X, y, groups = bundle["X"], bundle["y"], bundle["groups"]
    feats = bundle["features"]
    df_all = bundle["df"]

    uniq_groups = np.unique(groups)
    n_groups = uniq_groups.size
    if "folds" in bundle and args.cv and args.cv > 1:
        folds = bundle["folds"]
        print(f"[tune] using {len(folds)} folds from bundle (groups={n_groups})")
    else:
        n_splits = min(max(args.cv, 2), n_groups)
        if n_splits < 2:
            raise ValueError(f"Not enough groups for CV (groups={n_groups}). Add more groups.")
        gkf = GroupKFold(n_splits=n_splits)
        folds = [(tr.astype(int), va.astype(int)) for tr, va in gkf.split(X, y, groups)]
        print(f"[tune] generated {n_splits} GroupKFold splits (groups={n_groups})")

    grid_all = list(itertools.product(
        args.eta_grid,
        args.max_depth_grid,
        args.subsample_grid,
        args.colsample_bytree_grid,
        args.min_child_weight_grid,
        args.reg_lambda_grid,
    ))
    rng = np.random.RandomState(args.seed)
    if len(grid_all) > args.max_trials:
        sel_idx = rng.choice(len(grid_all), size=args.max_trials, replace=False)
        grid = [grid_all[i] for i in sel_idx]
    else:
        grid = grid_all
    print(f"[tune] trying {len(grid)} / {len(grid_all)} combinations")

    results = []
    best = None  # (avg_top1, avg_ndcg1, params, best_iters)

    for (eta, max_depth, subsample, colsample, min_child_weight, reg_lambda) in grid:
        params = dict(
            objective="rank:pairwise",
            eval_metric=["ndcg@1"],
            eta=eta,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            seed=args.seed,
        )

        top1_list, ndcg1_list, best_iter_list = [], [], []
        for (tr_idx, va_idx) in folds:
            dtrain = _dmatrix(X[tr_idx], y[tr_idx], groups[tr_idx])
            dvalid = _dmatrix(X[va_idx], y[va_idx], groups[va_idx])

            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=args.num_round,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=args.early_stop,
                verbose_eval=False,
            )
            preds = bst.predict(dvalid)
            df_fold = df_all.iloc[va_idx].copy()
            df_fold["pred_score"] = preds
            top1 = _top1_accuracy(df_fold)
            ndcg1 = float(bst.best_score)
            top1_list.append(top1)
            ndcg1_list.append(ndcg1)
            best_iter_list.append(int(bst.best_iteration))

        avg_top1 = float(np.mean(top1_list))
        avg_ndcg1 = float(np.mean(ndcg1_list))
        avg_best_iter = int(np.mean(best_iter_list)) if len(best_iter_list) else args.num_round

        results.append({
            "eta": eta,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "avg_top1": avg_top1,
            "avg_ndcg1": avg_ndcg1,
            "avg_best_iter": avg_best_iter,
        })

        if (best is None or
            (avg_top1, avg_ndcg1) > (best[0], best[1])):
            best = (avg_top1, avg_ndcg1,
                    dict(eta=eta, max_depth=max_depth, subsample=subsample,
                         colsample_bytree=colsample, min_child_weight=min_child_weight,
                         reg_lambda=reg_lambda),
                    avg_best_iter)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(results).sort_values(
        ["avg_top1", "avg_ndcg1"], ascending=[False, False]
    ).to_csv(out_dir / "tune_results.csv", index=False)

    if best is None:
        raise RuntimeError("No successful trials.")

    best_top1, best_ndcg1, best_params, best_iter = best
    print(f"[tune] best avg_top1={best_top1:.3f}  avg_ndcg1={best_ndcg1:.4f}  "
          f"params={best_params}  best_iter≈{best_iter}")

    dtrain_full = _dmatrix(X, y, groups)
    final_params = dict(
        objective="rank:pairwise",
        eval_metric=["ndcg@1"],
        seed=args.seed,
        **best_params,
    )
    final_model = xgb.train(
        final_params, dtrain_full, num_boost_round=max(best_iter, 1), verbose_eval=False
    )
    joblib.dump({"model": final_model, "features": feats}, out_dir / "xgb_ranker_tuned.pkl")
    (out_dir / "best_params.json").write_text(json.dumps({
        "best_top1": best_top1,
        "best_ndcg1": best_ndcg1,
        "best_params": best_params,
        "best_iter": best_iter,
        "features": feats,
    }, indent=2), encoding="utf-8")
    print(f"[tune] saved model → {out_dir/'xgb_ranker_tuned.pkl'}")
    print(f"[tune] saved results → {out_dir/'tune_results.csv'}")
    print(f"[tune] saved params  → {out_dir/'best_params.json'}")

if __name__ == "__main__":
    main()
