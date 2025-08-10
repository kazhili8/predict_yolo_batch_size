import argparse, itertools, json
from pathlib import Path
import joblib, numpy as np, xgboost as xgb
class EnsemblePredictor:
    def __init__(self, models):
        self.models = models
    def predict(self, X):
        outs = []
        for m in self.models:
            try:
                import xgboost as _xgb
                outs.append(m.predict(_xgb.DMatrix(X)))
            except Exception:
                outs.append(m.predict(X))
        import numpy as _np
        return _np.mean(_np.column_stack(outs), axis=1)
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--num_round", type=int, default=400)
    p.add_argument("--early_stop", type=int, default=40)
    p.add_argument("--eta_grid", nargs="+", type=float, default=[0.05,0.1])
    p.add_argument("--max_depth_grid", nargs="+", type=int, default=[4,6,8])
    p.add_argument("--subsample_grid", nargs="+", type=float, default=[0.8,1.0])
    p.add_argument("--colsample_bytree_grid", nargs="+", type=float, default=[0.8,1.0])
    p.add_argument("--min_child_weight_grid", nargs="+", type=float, default=[1.0,5.0])
    p.add_argument("--reg_lambda_grid", nargs="+", type=float, default=[1.0,5.0])
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--out_dir", default="scripts/outputs/tune_small")
    p.add_argument("--max_trials", type=int, default=96)
    p.add_argument("--with_lgbm", action="store_true")
    return p.parse_args()

def to_group_relevance(y_sorted, counts):
    import numpy as np
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
    d = xgb.DMatrix(Xs, label=y_rel)
    d.set_group(counts.tolist())
    if gw is not None:
        d.set_weight(gw)
    return d

def fold_splits(groups, n_splits, seed):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_splits)
    idx = np.arange(groups.shape[0])
    out = []
    for k in range(n_splits):
        va_g = set(folds[k].tolist())
        va = np.nonzero(np.isin(groups, list(va_g)))[0]
        tr = np.setdiff1d(idx, va, assume_unique=False)
        out.append((tr, va))
    return out

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
    X = bundle["X"]; y = bundle["y"]; groups = bundle["groups"]; feats = bundle["features"]
    folds = bundle.get("folds")
    inst_w = bundle.get("sample_weight", None)

    if folds is None or len(folds)==0:
        folds = fold_splits(groups, args.cv, args.seed)

    grid_all = list(itertools.product(
        args.eta_grid, args.max_depth_grid, args.subsample_grid, args.colsample_bytree_grid, args.min_child_weight_grid, args.reg_lambda_grid
    ))
    grid = grid_all[:min(len(grid_all), args.max_trials)]

    print(f"[tune] using {args.cv} folds from bundle (groups={int(np.unique(groups).size)})")
    print(f"[tune] trying {len(grid)} / {len(grid_all)} combinations")

    params_tpl = dict(objective="rank:pairwise", eval_metric=["ndcg@1"], tree_method="hist", seed=args.seed)
    results = []
    best = None

    for (eta, md, sub, col, mcw, rl) in grid:
        params = dict(params_tpl)
        params.update(dict(eta=eta, max_depth=md, subsample=sub, colsample_bytree=col, min_child_weight=mcw, reg_lambda=rl))
        fold_scores = []
        fold_iters = []
        for (tr, va) in folds:
            Xtr, ytr_f, ytr_rel, gtr, wtr, ctr, gwtr = group_by_sorted(X[tr], y[tr], groups[tr], inst_w[tr] if inst_w is not None else None)
            Xva, yva_f, yva_rel, gva, wva, cva, gwva = group_by_sorted(X[va], y[va], groups[va], inst_w[va] if inst_w is not None else None)
            dtr = dmatrix_grouped(Xtr, ytr_rel, ctr, gwtr)
            dva = dmatrix_grouped(Xva, yva_rel, cva, gwva)
            bst = xgb.train(params, dtr, num_boost_round=args.num_round, evals=[(dva,"va")], early_stopping_rounds=args.early_stop, verbose_eval=False)
            pred = predict_with_best(bst, dva)
            top1 = per_group_top1(yva_f, pred, gva)
            fold_scores.append(top1); fold_iters.append(int(getattr(bst, "best_iteration", 0) or 0))
        avg_top1 = float(np.mean(fold_scores))
        results.append(dict(eta=eta,max_depth=md,subsample=sub,colsample_bytree=col,min_child_weight=mcw,reg_lambda=rl,avg_top1=avg_top1,avg_best_iter=float(np.mean(fold_iters))))
        if best is None or avg_top1 > best[0]:
            best = (avg_top1, dict(eta=eta,max_depth=md,subsample=sub,colsample_bytree=col,min_child_weight=mcw,reg_lambda=rl), int(np.round(np.mean(fold_iters))))

    best_top1, best_params, best_iter = best
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
        pd.DataFrame(results).to_csv(out_dir/"tune_results.csv", index=False)
    except Exception:
        pass

    final_params = dict(params_tpl); final_params.update(best_params)
    Xall, yall_f, yall_rel, gall, wall, call, gwall = group_by_sorted(X, y, groups, inst_w if inst_w is not None else None)
    dall = dmatrix_grouped(Xall, yall_rel, call, gwall)
    final_model = xgb.train(final_params, dall, num_boost_round=max(best_iter,1))
    joblib.dump({"model": final_model, "features": feats}, out_dir/"xgb_ranker_tuned.pkl")
    (out_dir/"best_params.json").write_text(json.dumps({
        "best_top1": best_top1, "best_params": best_params, "best_iter": best_iter, "features": feats
    }, indent=2), encoding="utf-8")
    print(f"[tune] best avg_top1={best_top1:.3f}  params={best_params}  best_iter≈{best_iter}")
    print(f"[tune] saved model → {out_dir/'xgb_ranker_tuned.pkl'}")

    if args.with_lgbm:
        import lightgbm as lgb
        yall_rel_int = yall_rel.astype(np.int32)
        max_rel = int(yall_rel_int.max())
        lg = list(range(max_rel + 1))
        lgbm = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=max(best_iter, 100),
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=args.seed,
            label_gain=lg
        )
        lgbm.fit(Xall, yall_rel_int, group=call.tolist(), sample_weight=wall)
        joblib.dump({"model": lgbm, "features": feats}, out_dir / "lgbm_ranker_tuned.pkl")

        ens = EnsemblePredictor([final_model, lgbm])
        joblib.dump({"model": ens, "features": feats}, out_dir / "ranker_ensemble.pkl")

        print(f"[tune] saved model → {out_dir / 'lgbm_ranker_tuned.pkl'}")
        print(f"[tune] saved model → {out_dir / 'ranker_ensemble.pkl'}")

if __name__ == "__main__":
    main()
