import argparse, pandas as pd, numpy as np, joblib, pickle, os
from pathlib import Path
import xgboost as xgb

def load_models_with_fallback(pkl_path: str):
    p = Path(pkl_path)
    try:
        obj = joblib.load(pkl_path)
        return [unwrap_model_object(obj)], get_features_from_obj(obj)
    except Exception:
        try:
            obj = pickle.load(open(pkl_path, "rb"))
            return [unwrap_model_object(obj)], get_features_from_obj(obj)
        except Exception:
            pass
    models = []
    feats = None
    d = p.parent
    xgb_p = d / "xgb_ranker_tuned.pkl"
    lgb_p = d / "lgbm_ranker_tuned.pkl"
    for q in [xgb_p, lgb_p]:
        if q.exists():
            try:
                obj = joblib.load(str(q))
            except Exception:
                try:
                    obj = pickle.load(open(q, "rb"))
                except Exception:
                    obj = None
            if obj is not None:
                models.append(unwrap_model_object(obj))
                if feats is None:
                    feats = get_features_from_obj(obj)
    if models:
        return models, feats
    raise RuntimeError(f"cannot load model from {pkl_path} or its siblings")

def unwrap_model_object(obj):
    import lightgbm as lgb
    if isinstance(obj, dict) and "model" in obj:
        m = obj["model"]
    else:
        m = obj
    kind = None
    if isinstance(m, xgb.Booster):
        kind = "xgb_booster"
    elif hasattr(m, "get_booster") and callable(getattr(m, "get_booster")):
        kind = "xgb_sklearn"
    elif "lightgbm" in type(m).__module__ or isinstance(m, lgb.LGBMRanker):
        kind = "lgbm"
    else:
        if isinstance(obj, dict):
            if "booster" in obj and isinstance(obj["booster"], (bytes, bytearray)):
                bst = xgb.Booster()
                bst.load_model(obj["booster"])
                m = bst
                kind = "xgb_booster"
    if kind is None:
        raise RuntimeError("unsupported model object")
    return (m, kind)

def get_features_from_obj(obj):
    feats = None
    if isinstance(obj, dict) and "features" in obj and isinstance(obj["features"], (list, tuple)):
        feats = list(obj["features"])
    return feats

def predict_one(model_tuple, X_df, feats):
    m, kind = model_tuple
    X_pd = X_df[feats]
    if kind == "xgb_booster":
        dm = xgb.DMatrix(X_pd.values, feature_names=feats)
        return m.predict(dm)
    elif kind == "xgb_sklearn":
        try:
            dm = xgb.DMatrix(X_pd.values, feature_names=feats)
            return m.predict(dm)
        except Exception:
            return m.predict(X_pd.values)
    else:
        return m.predict(X_pd)

def pick_feature_names_fallback(df, models):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_meta = {"batch","map50","n_logs"}
    num_cols = [c for c in num_cols if c not in drop_meta]
    prefixes = ("avg_","gpu_util_","mem_clock_","mem_util_","power_","pwr_","sm_clock_","step_time_","temp_","thr_","throughput","throughput_var_ratio","vram_total_mb","energy_per_img","is65W","power_limit_w")
    cand = [c for c in num_cols if c.startswith(prefixes) or c in ["avg_mem","avg_power","avg_step_time","inv_step_time","throughput","map50","pwr_mean","vram_total_mb"]]
    cand = sorted(list(dict.fromkeys(cand)))
    if cand:
        return cand
    return num_cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--model_pkl", required=True)
    ap.add_argument("--cap_w", type=float, required=True)
    ap.add_argument("--delta_map", type=float, default=0.01)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model_pkl2", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)
    models, feats = load_models_with_fallback(args.model_pkl)
    if args.model_pkl2:
        m2, f2 = load_models_with_fallback(args.model_pkl2)
        models.extend(m2)
        if feats is None:
            feats = f2
    if feats is None:
        feats = pick_feature_names_fallback(df, models)
    feats = [c for c in feats if c in df.columns]
    if not feats:
        raise RuntimeError("no usable features")

    recs=[]
    for (m,e,t), g0 in df.groupby(["model","epochs","tag"], sort=False):
        b1 = g0[g0["batch"]==1]
        if len(b1)==0:
            continue
        base = float(b1["map50"].iloc[0])
        g = g0[(g0["map50"]>=base*(1.0-args.delta_map)) & (g0["avg_power"]<=args.cap_w)].copy()
        if len(g)==0:
            continue
        preds = []
        for mt in models:
            preds.append(predict_one(mt, g[feats], feats))
        if len(preds)==1:
            g["pred_score"] = preds[0]
        else:
            g["pred_score"] = np.mean(np.column_stack(preds), axis=1)
        g = g.sort_values("pred_score", ascending=False)
        top = g.iloc[0]
        recs.append({"model":m,"epochs":int(e),"tag":t,"recommended_batch":int(top["batch"]),"pred_score":float(top["pred_score"])})

    out = pd.DataFrame(recs)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
