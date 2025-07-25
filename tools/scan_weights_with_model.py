import itertools, numpy as np, pandas as pd, joblib, xgboost as xgb, pathlib

CSV   = pathlib.Path("scripts/outputs/dataframe/features_v2.csv")
MODEL = joblib.load("models/model_ranker.pkl")
FEATS = ["batch","throughput","avg_mem","pwr_mean","pwr_std","energy_per_img"]
MAP_CANDIDATES = ["map50","avg_map50","mAP50","mAP@0.5"]

df = pd.read_csv(CSV)
map_col = next(c for c in MAP_CANDIDATES if c in df.columns)

def top1_accuracy(weights):
    wt_T, wt_P, wt_M, wt_D = weights
    hits = total = 0
    for _, g in df.groupby(["model","epochs"], dropna=False):
        g = g.dropna(subset=FEATS+[map_col,"avg_power","avg_mem","throughput"]).copy()
        if g.empty: continue
        g["delta_map"] = g[map_col].max()-g[map_col]
        g["true_score"] = ( wt_T*g["throughput"]
                          - wt_P*g["avg_power"]
                          - wt_M*g["avg_mem"]
                          - wt_D*g["delta_map"] )
        d = xgb.DMatrix(g[FEATS])
        g["pred_score"] = MODEL.predict(d)
        hits += int(g.loc[g["pred_score"].idxmax(),"batch"] ==
                    g.loc[g["true_score"].idxmax(),"batch"])
        total += 1
    return hits/total if total else np.nan

grids = [(0.6,0.2,0.1,0.1),
         (0.5,0.3,0.1,0.1),
         (0.5,0.2,0.2,0.1),
         (0.4,0.3,0.2,0.1),
         (0.7,0.2,0.05,0.05)]
rows=[]
for w in grids:
    acc=top1_accuracy(w)
    rows.append((*w,acc))
    print(f"Weights {w} → Top-1 {acc:.2%}")

pd.DataFrame(rows,columns=["T","P","M","Δ","Top1"]).to_csv(
    "metrics_weight_sweep.csv",index=False)
print("Saved to metrics_weight_sweep.csv")