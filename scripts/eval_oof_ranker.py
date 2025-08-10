import argparse, json, joblib, numpy as np, pandas as pd
from pathlib import Path
import xgboost as xgb

def parse_args():
    p = argparse.ArgumentParser("Evaluate ranker with out-of-fold predictions")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out_csv", default="scripts/outputs/oof_preds.csv")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--num_round", type=int, default=400)
    p.add_argument("--early_stop", type=int, default=40)
    p.add_argument("--params_json", default="")
    p.add_argument("--cv-group-cols", nargs="+", default=[])
    p.add_argument("--seed", type=int, default=2025)
    return p.parse_args()

def to_group_relevance(y_sorted, counts):
    rel = np.empty_like(y_sorted, dtype=np.float32)
    b = np.r_[0, np.cumsum(counts)]
    for i in range(len(counts)):
        s, e = b[i], b[i+1]
        seg = y_sorted[s:e]
        order = np.argsort(-seg, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.int32)
        ranks[order] = np.arange(e - s, dtype=np.int32)[::-1]
        rel[s:e] = ranks
    return rel

def group_by_sorted(X, y_float, groups, inst_w=None):
    order = np.argsort(groups, kind="mergesort")
    Xs = X[order]; ys_float = y_float[order]; gs = groups[order]
    ws = inst_w[order] if inst_w is not None else None
    b = np.r_[0, np.flatnonzero(np.diff(gs)) + 1, len(gs)]
    counts = np.diff(b)
    gw = None
    if ws is not None:
        gw = np.empty(len(counts), dtype=np.float32)
        for i in range(len(counts)):
            s, e = b[i], b[i+1]
            gw[i] = float(ws[s:e].mean())
    y_rel = to_group_relevance(ys_float, counts).astype(np.float32)
    return Xs, ys_float, y_rel, gs, ws, counts, gw

def dmatrix_grouped(Xs, y_rel, counts, gw=None):
    d = xgboost_DMatrix(Xs, label=y_rel)
    d.set_group(counts.tolist())
    if gw is not None:
        d.set_weight(gw)
    return d

def xgboost_DMatrix(X, label=None):
    try:
        return xgb.DMatrix(X, label=label)
    except TypeError:
        return xgb.DMatrix(X) if label is None else xgb.DMatrix(X, label=label)

def per_group_top1(y_true_float_sorted, y_pred_sorted, groups_sorted):
    b = np.r_[0, np.flatnonzero(np.diff(groups_sorted)) + 1, len(groups_sorted)]
    hits = 0
    for i in range(len(b)-1):
        s, e = b[i], b[i+1]
        a = y_true_float_sorted[s:e]; p = y_pred_sorted[s:e]
        hits += int(np.argmax(p) == np.argmax(a))
    return hits / (len(b)-1)

def predict_with_best(bst, dmat):
    bi = getattr(bst, "best_iteration", None)
    if isinstance(bi, (int, np.integer)) and bi >= 0:
        try:
            return bst.predict(dmat, iteration_range=(0, int(bi)+1))
        except Exception:
            pass
    bn = getattr(bst, "best_ntree_limit", None)
    if isinstance(bn, (int, np.integer)) and bn > 0:
        try:
            return bst.predict(dmat, ntree_limit=int(bn))
        except Exception:
            pass
    return bst.predict(dmat)

def main():
    args = parse_args()
    bundle = joblib.load(args.data)
    X = bundle["X"]; y = bundle["y"]; groups_query = bundle["groups"]; feats = bundle["features"]
    df = bundle["df"].copy()
    inst_w = bundle.get("sample_weight", None)
    orig_group_cols = bundle.get("group_cols", ["model","epochs","tag"])

    if args.cv_group_cols:
        keys = df[args.cv_group_cols].astype(str).agg("|".join, axis=1).values
        uniq, inv = np.unique(keys, return_inverse=True)
        groups_split = inv
        cv = len(np.unique(groups_split))
    else:
        groups_split = groups_query
        cv = args.cv

    print(f"[eval] groups_for_eval={int(np.unique(groups_query).size)}, groups_for_cv={int(np.unique(groups_split).size)}, use cv={cv}, features={len(feats)}")

    oof_pred = np.zeros(X.shape[0], dtype=float)
    fold_top1 = []

    uniq_cv = np.unique(groups_split)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(uniq_cv)
    folds = np.array_split(uniq_cv, cv)

    params = dict(objective="rank:pairwise", eval_metric=["ndcg@1"], tree_method="hist", seed=args.seed, eta=0.05, max_depth=6)

    for k in range(cv):
        va_g = set(folds[k].tolist())
        va = np.nonzero(np.isin(groups_split, list(va_g)))[0]
        tr = np.setdiff1d(np.arange(X.shape[0]), va, assume_unique=False)

        Xtr, ytr_f, ytr_rel, gtr, wtr, ctr, gwtr = group_by_sorted(X[tr], y[tr], groups_query[tr], inst_w[tr] if inst_w is not None else None)
        Xva, yva_f, yva_rel, gva, wva, cva, gwva = group_by_sorted(X[va], y[va], groups_query[va], inst_w[va] if inst_w is not None else None)

        dtr = dmatrix_grouped(Xtr, ytr_rel, ctr, gwtr)
        dva = dmatrix_grouped(Xva, yva_rel, cva, gwva)

        bst = xgb.train(params, dtr, num_boost_round=args.num_round, evals=[(dva,"va")], early_stopping_rounds=args.early_stop, verbose_eval=False)
        pred = predict_with_best(bst, dva)

        oof_pred[va] = pred
        df.loc[df.index[va], "rank_pred"] = pred

        t1 = per_group_top1(yva_f, pred, gva)
        print(f"[fold {k}] Top-1={t1:.3f} (best_iter={int(getattr(bst,'best_iteration',0) or 0)})")
        fold_top1.append(t1)

    overall_top1 = float(np.mean(fold_top1))
    print("--------------------------------------------------")
    print(f"[OOF] Avg Top-1 across folds = {overall_top1:.3f}")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[eval] saved OOF predictions → {args.out_csv}")

if __name__ == "__main__":
    main()
