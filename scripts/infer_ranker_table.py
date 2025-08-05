import argparse, pandas as pd, pickle, numpy as np
import xgboost as xgb
from pathlib import Path

def unwrap_model(obj):
    if isinstance(obj, xgb.Booster):
        return obj, "booster"
    if hasattr(obj, "predict") and hasattr(obj, "get_booster"):
        return obj, "sklearn"
    if isinstance(obj, dict):
        for k in ["model", "ranker", "xgb_model", "xgb", "booster"]:
            if k in obj:
                v = obj[k]
                if isinstance(v, xgb.Booster):
                    return v, "booster"
                if hasattr(v, "predict") and hasattr(v, "get_booster"):
                    return v, "sklearn"
        for k in ["booster_json", "booster_raw", "booster_str"]:
            if k in obj and isinstance(obj[k], (str, bytes)):
                bst = xgb.Booster()
                s = obj[k]
                if isinstance(s, bytes):
                    s = s.decode("utf-8")
                bst.load_config(s)
                if "booster" in obj and isinstance(obj["booster"], bytes):
                    bst.load_model(obj["booster"])
                return bst, "booster"
        if "model_path" in obj:
            p = Path(obj["model_path"])
            if p.exists():
                try:
                    bst = xgb.Booster()
                    bst.load_model(str(p))
                    return bst, "booster"
                except Exception:
                    pass
    raise RuntimeError("cannot unwrap xgboost model from pickle")

def pick_feature_names_fallback(df, model, kind):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_meta = {"batch","map50","n_logs"}
    num_cols = [c for c in num_cols if c not in drop_meta]
    prefixes = (
        "avg_","gpu_util_","mem_clock_","mem_util_","power_","pwr_",
        "sm_clock_","step_time_","temp_","thr_","throughput","throughput_var_ratio",
        "vram_total_mb","energy_per_img","is65W","power_limit_w"
    )
    cand = [c for c in num_cols if c.startswith(prefixes) or c in ("throughput","throughput_var_ratio","vram_total_mb","energy_per_img","is65W","power_limit_w")]
    cand = sorted(list(dict.fromkeys(cand)))
    nfeat = None
    try:
        if kind == "booster":
            nfeat = int(model.num_features())
        else:
            nfeat = int(model.get_booster().num_features())
    except Exception:
        nfeat = None
    if nfeat and len(cand) >= nfeat:
        return cand[:nfeat]
    if nfeat is None:
        return cand
    other = [c for c in num_cols if c not in cand]
    merged = cand + sorted(other)
    if len(merged) >= nfeat:
        return merged[:nfeat]
    return merged

def get_feature_names(df, model, kind):
    feats = None
    if kind == "booster":
        try:
            feats = model.feature_names
        except Exception:
            feats = None
    else:
        try:
            feats = model.get_booster().feature_names
        except Exception:
            feats = None
        if not feats:
            feats = getattr(model, "feature_names_in_", None)
    if feats and len(feats) > 0:
        return feats
    return pick_feature_names_fallback(df, model, kind)

def predict_scores(model, kind, X_df, feats):
    if kind == "booster":
        dm = xgb.DMatrix(X_df[feats].values, feature_names=feats)
        return model.predict(dm)
    else:
        return model.predict(X_df[feats].values)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--model_pkl", required=True)
    ap.add_argument("--cap_w", type=float, required=True)
    ap.add_argument("--delta_map", type=float, default=0.01)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)
    rk_raw = pickle.load(open(args.model_pkl,"rb"))
    rk, kind = unwrap_model(rk_raw)
    feats = get_feature_names(df, rk, kind)
    if not feats:
        raise RuntimeError("feature names not found or cannot be inferred")

    recs=[]
    for (m,e,t), g0 in df.groupby(["model","epochs","tag"], sort=False):
        b1 = g0[g0["batch"]==1]
        if len(b1)==0:
            continue
        base = float(b1["map50"].iloc[0])
        g = g0[(g0["map50"]>=base*(1.0-args.delta_map)) & (g0["pwr_mean"]<=args.cap_w)].copy()
        if len(g)==0:
            continue
        g["pred_score"] = predict_scores(rk, kind, g, feats)
        g = g.sort_values("pred_score", ascending=False)
        top = g.iloc[0]
        recs.append({"model":m,"epochs":int(e),"tag":t,"recommended_batch":int(top["batch"]),"pred_score":float(top["pred_score"])})

    out = pd.DataFrame(recs)
    out.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
