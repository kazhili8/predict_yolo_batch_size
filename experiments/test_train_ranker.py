import argparse
import joblib
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
import config
from scoring import add_true_score
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost Ranker")
    p.add_argument("--data", default="scripts/outputs/rank_data_v1.pkl",
                   help="Path to bundle produced by make_rank_data.py")
    p.add_argument("--cv", type=int, default=config.N_FOLDS,
                   help="0 = train on full data, >0 = GroupKFold CV")
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--num_round", type=int, default=300)
    p.add_argument("--early_stop", type=int, default=30)
    p.add_argument("--save-model", default="scripts/outputs/xgb_ranker.pkl")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()

def _dmatrix(X: np.ndarray, y: np.ndarray, group: np.ndarray) -> xgb.DMatrix:
    dmat = xgb.DMatrix(X, label=y)
    _, counts = np.unique(group, return_counts=True)
    dmat.set_group(counts.tolist())
    return dmat


def _top1_accuracy(df_fold) -> float:
    """Per-group Top-1 accuracy using true_score vs pred_score"""
    hit = total = 0
    for _, g in df_fold.groupby(["model", "epochs", "tag"], sort=False):
        gt_idx = g["true_score"].idxmax()
        pred_idx = g["pred_score"].idxmax()
        hit += int(gt_idx == pred_idx)
        total += 1
    return hit / max(total, 1)

def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.data)
    X, y, groups = bundle["X"], bundle["y"], bundle["groups"]
    feats = bundle["features"]
    df_all = bundle["df"]

    params = dict(
        objective="rank:pairwise",
        eval_metric=["ndcg@1", "ndcg@5"],
        eta=args.eta,
        max_depth=args.max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=args.seed,
    )

    if args.cv and "folds" in bundle and args.cv > 1:
        print(f"Running {args.cv}-fold GroupKFold CV ...")
        top1_list, ndcg1_list, best_iter_list = [], [], []

        for fold_id, (tr_idx, va_idx) in enumerate(bundle["folds"]):
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
            ndcg1 = bst.best_score

            top1_list.append(top1)
            ndcg1_list.append(ndcg1)
            best_iter_list.append(bst.best_iteration)
            print(f"[fold {fold_id}] best_iter={bst.best_iteration:3d} | "
                  f"nDCG@1={ndcg1:.4f} | Top-1={top1:.3f}")

        print("-" * 60)
        print(f"Avg  nDCG@1 = {np.mean(ndcg1_list):.4f}")
        print(f"Avg  Top-1   = {np.mean(top1_list):.3f}")

        dtrain_full = _dmatrix(X, y, groups)
        final = xgb.train(
            params,
            dtrain_full,
            num_boost_round=int(np.mean(best_iter_list)),
            verbose_eval=False,
        )
        model = final
    else:
        dtrain = _dmatrix(X, y, groups)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=args.num_round,
            verbose_eval=True,
        )

    Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feats}, args.save_model)
    print(f"[train_ranker] model saved to {args.save_model}")

if __name__ == "__main__":
    main()
