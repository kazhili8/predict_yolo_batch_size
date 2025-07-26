from __future__ import annotations
import argparse, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
import xgboost as xgb
from metrics import top1_accuracy

def parse_args():
    p = argparse.ArgumentParser("Evaluate ranker with out-of-fold predictions (Booster-safe)")
    p.add_argument("--data", required=True, help="rank_data.pkl from make_rank_data.py")
    p.add_argument("--model", required=True, help="trained model .pkl (train_ranker or tune_xgb)")
    p.add_argument("--out_csv", default="scripts/outputs/oof_preds.csv")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--num_round", type=int, default=300)
    p.add_argument("--early_stop", type=int, default=50)
    p.add_argument("--params_json", default="", help="optional: JSON file with tuned xgb params")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def _dmatrix_with_group(X_, y_, g_):
    order = np.argsort(g_, kind="stable")
    Xs, ys, gs = X_[order], y_[order], g_[order]
    _, counts = np.unique(gs, return_counts=True)
    d = xgb.DMatrix(Xs, label=ys)
    d.set_group(counts.tolist())
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return d, inv

def load_params(model_bundle, params_json, seed):
    params = dict(
        objective="rank:pairwise",
        eval_metric="ndcg@1",
        eta=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        reg_lambda=1.0,
        seed=seed,
    )
    if isinstance(model_bundle, dict) and "params" in model_bundle:
        params.update(model_bundle["params"])
    if params_json:
        try:
            with open(params_json, "r", encoding="utf-8") as f:
                best = json.load(f)
            params.update(best)
        except Exception:
            pass
    for k in ["eta", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda"]:
        if k in params: params[k] = float(params[k])
    if "max_depth" in params: params["max_depth"] = int(params["max_depth"])
    return params

def main():
    args = parse_args()
    bundle = joblib.load(args.data)
    model_bundle = joblib.load(args.model)
    X = bundle["X"]; y = bundle["y"]; df = bundle["df"].copy()
    feats = model_bundle.get("features", bundle["features"])
    groups = bundle["groups"]
    uniq_g = np.unique(groups)
    cv = min(args.cv, len(uniq_g))
    print(f"[eval] groups={len(uniq_g)}, use cv={cv}")

    params = load_params(model_bundle, args.params_json, args.seed)

    gkf = GroupKFold(n_splits=cv)
    oof_pred = np.zeros(X.shape[0], dtype=float)
    fold_top1 = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        dtr, _ = _dmatrix_with_group(X[tr], y[tr], groups[tr])
        dva, inv = _dmatrix_with_group(X[va], y[va], groups[va])

        bst = xgb.train(
            params,
            dtr,
            num_boost_round=args.num_round,
            evals=[(dva, "valid")],
            early_stopping_rounds=args.early_stop,
            verbose_eval=False,
        )

        ntree_limit = getattr(bst, "best_ntree_limit", 0)
        if ntree_limit and ntree_limit > 0:
            pred_sorted = bst.predict(dva, ntree_limit=ntree_limit)
        else:
            pred_sorted = bst.predict(dva)

        preds = pred_sorted[inv]
        oof_pred[va] = preds

        df_fold = df.iloc[va].copy()
        df_fold["pred_score"] = preds
        t1 = top1_accuracy(df_fold)
        fold_top1.append(t1)
        print(f"[fold {fold}] Top-1={t1:.3f} (best_iter={getattr(bst,'best_iteration',None)})")

    df["pred_score"] = oof_pred
    df["rank_pred"] = df.groupby(["model","epochs","tag"], sort=False)["pred_score"] \
                        .rank(method="first", ascending=False).astype(int)

    overall_top1 = float(np.mean(fold_top1))
    print("--------------------------------------------------")
    print(f"[OOF] Avg Top-1 across folds = {overall_top1:.3f}")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["model","epochs","tag","rank_pred"], inplace=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[eval] saved OOF predictions → {args.out_csv}")

if __name__ == "__main__":
    main()
