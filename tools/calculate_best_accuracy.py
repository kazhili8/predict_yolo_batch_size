import argparse, numpy as np, pandas as pd, joblib
from pathlib import Path
import xgboost as xgb
import pickle

MAP_CANDIDATES = ["map50","avg_map50","mAP50","mAP@0.5"]
FALLBACK_FEATS = [
    "throughput","inv_step_time","avg_power","avg_mem",
    "pwr_mean","pwr_std","pwr_cv","energy_per_img",
    "eff_tp_watt","eff_tp_mem","throughput_var_ratio","vram_total_mb"
]

def pick_map_col(df):
    for c in MAP_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("No mAP column found")

def ensure_feats(df, pref):
    cols = list(pref) if pref else FALLBACK_FEATS
    feats = [c for c in cols if c in df.columns]
    if not feats:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feats = [c for c in num_cols if c not in {"batch","map50","n_logs"}]
    return feats

def unwrap_model_object(obj):
    import lightgbm as lgb
    m = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    if isinstance(m, xgb.Booster):
        kind = "xgb_booster"
    elif hasattr(m, "get_booster") and callable(getattr(m, "get_booster")):
        kind = "xgb_sklearn"
    elif "lightgbm" in type(m).__module__ or isinstance(m, lgb.LGBMRanker):
        kind = "lgbm"
    else:
        raise RuntimeError("unsupported model object")
    feats = None
    if isinstance(obj, dict) and "features" in obj and isinstance(obj["features"], (list, tuple)):
        feats = list(obj["features"])
    return (m, kind, feats)

def load_models_list(pkl1, pkl2=""):
    models = []
    feats = None
    for p in [pkl1, pkl2] if pkl2 else [pkl1]:
        if not p:
            continue
        try:
            obj = joblib.load(p)
        except Exception:
            obj = pickle.load(open(p, "rb"))
        m, kind, f = unwrap_model_object(obj)
        models.append((m, kind))
        if feats is None and f is not None:
            feats = f
    return models, feats

def predict_one(model_tuple, X_df, feats):
    m, kind = model_tuple
    X_pd = X_df[feats] if feats is not None else X_df
    if kind == "xgb_booster":
        dm = xgb.DMatrix(X_pd.values, feature_names=(feats if feats is not None else None))
        return m.predict(dm)
    elif kind == "xgb_sklearn":
        try:
            dm = xgb.DMatrix(X_pd.values, feature_names=(feats if feats is not None else None))
            return m.predict(dm)
        except Exception:
            return m.predict(X_pd.values)
    else:
        return m.predict(X_pd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--model_pkl", required=True)
    ap.add_argument("--model_pkl2", default="")
    ap.add_argument("--weights", nargs=4, type=float, default=[0.6,0.2,0.1,0.1])
    ap.add_argument("--group_cols", nargs="+", default=["model","epochs","tag"])
    ap.add_argument("--cap_w", type=float, default=None)
    ap.add_argument("--delta_map", type=float, default=0.01)
    ap.add_argument("--out_md", default="metrics_ranker.md")
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)
    models, pref_feats = load_models_list(args.model_pkl, args.model_pkl2)
    feats = [c for c in ensure_feats(df, pref_feats) if c in df.columns]
    if not feats:
        raise RuntimeError("no usable features")
    map_col = pick_map_col(df)
    wT, wP, wM, wD = args.weights

    hits = 0
    total = 0

    for _, g0 in df.groupby(args.group_cols, dropna=False):
        need = list(set(feats + [map_col,"avg_power","avg_mem","throughput","pwr_mean","batch"]))
        g = g0.dropna(subset=[c for c in need if c in g0.columns]).copy()
        if g.empty:
            continue
        if args.cap_w is not None:
            b1 = g[g["batch"]==1]
            if len(b1)==0:
                continue
            base_map = float(b1[map_col].iloc[0])
            map_min = base_map*(1.0-args.delta_map)
            feas = (g[map_col] >= map_min) & (g["pwr_mean"] <= args.cap_w)
            g = g[feas].copy()
            if g.empty:
                continue

        g["delta_map"] = g[map_col].max() - g[map_col]
        g["true_score"] = wT*g["throughput"] - wP*g["avg_power"] - wM*g["avg_mem"] - wD*g["delta_map"]

        preds = []
        for mt in models:
            preds.append(predict_one(mt, g[feats], feats))
        if len(preds) == 1:
            g["pred_score"] = preds[0]
        else:
            g["pred_score"] = np.mean(np.column_stack(preds), axis=1)

        total += 1
        pred_best_batch = int(g.sort_values("pred_score", ascending=False)["batch"].iloc[0])
        true_best_batch = int(g.sort_values("true_score", ascending=False)["batch"].iloc[0])
        hits += int(pred_best_batch == true_best_batch)

    acc = np.nan if total == 0 else hits/total
    Path(args.out_md).write_text(f"Feasible-set Top-1 = {acc:.2%}  ({hits}/{total})\n", encoding="utf-8")
    print(f"written → {args.out_md}")

if __name__ == "__main__":
    main()
