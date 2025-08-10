import argparse, pathlib, numpy as np, pandas as pd, joblib, xgboost as xgb

MAP_CANDIDATES = ["map50","avg_map50","mAP50","mAP@0.5"]
FALLBACK_FEATS = ["batch","throughput","avg_mem","pwr_mean","pwr_std","energy_per_img","eff_tp_watt","eff_tp_mem","inv_step_time","pwr_cv"]

def pick_map_col(df):
    for c in MAP_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("No mAP column found")

def parse_grid(spec):
    if spec == "auto":
        return [(0.6,0.2,0.1,0.1),(0.5,0.3,0.1,0.1),(0.5,0.2,0.2,0.1),(0.4,0.3,0.2,0.1),(0.7,0.2,0.05,0.05)]
    items = []
    for tok in spec.split(";"):
        t = [float(x) for x in tok.split(",")]
        if len(t) == 4:
            items.append(tuple(t))
    return items

def load_model_and_feats(pkl):
    obj = joblib.load(pkl)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        feats = obj.get("features")
        return model, feats
    return obj, None

def ensure_feats(df, pref):
    if pref:
        return [c for c in pref if c in df.columns]
    return [c for c in FALLBACK_FEATS if c in df.columns]

def predict_scores(model, X):
    try:
        return model.predict(xgb.DMatrix(X))
    except Exception:
        return model.predict(X)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--model_pkl", required=True)
    ap.add_argument("--grid", default="auto")
    ap.add_argument("--group_cols", nargs="+", default=["model","epochs","tag"])
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)
    model, pref_feats = load_model_and_feats(args.model_pkl)
    feats = ensure_feats(df, pref_feats)
    map_col = pick_map_col(df)

    rows = []
    grids = parse_grid(args.grid)
    for wT, wP, wM, wD in grids:
        hits = total = 0
        for _, g in df.groupby(args.group_cols, dropna=False):
            need = feats + [map_col,"avg_power","avg_mem","throughput"]
            g = g.dropna(subset=[c for c in need if c in g.columns]).copy()
            if g.empty:
                continue
            g["delta_map"] = g[map_col].max() - g[map_col]
            g["true_score"] = wT*g["throughput"] - wP*g["avg_power"] - wM*g["avg_mem"] - wD*g["delta_map"]
            X = g[feats].to_numpy()
            g["pred_score"] = predict_scores(model, X)
            hits += int(g.loc[g["pred_score"].idxmax(),"batch"] == g.loc[g["true_score"].idxmax(),"batch"])
            total += 1
        acc = (hits/total) if total else np.nan
        rows.append((wT,wP,wM,wD,acc))
    out = pd.DataFrame(rows, columns=["T","P","M","Δ","Top1"])
    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"saved → {args.out_csv}")

if __name__ == "__main__":
    main()
