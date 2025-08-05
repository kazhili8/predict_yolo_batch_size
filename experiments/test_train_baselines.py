import pandas as pd, joblib, pathlib
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb

DATA = pathlib.Path(r"scripts/outputs/dataframe/features_v2.csv")
TARGET = "avg_power"
FEATS   = ["batch","throughput","avg_mem","energy_per_img",
           "pwr_mean","pwr_std","pwr_max"]

df   = pd.read_csv(DATA).dropna(subset=[TARGET])
X, y = df[FEATS], df[TARGET]
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=42),
    "XGBoost":    xgb.XGBRegressor(max_depth=6, n_estimators=400, learning_rate=0.1,
                                   objective="reg:squarederror", subsample=0.8),
    "LightGBM":   lgb.LGBMRegressor(max_depth=-1, n_estimators=400, learning_rate=0.05)
}

with open("results_baseline.md", "w", encoding="utf-8") as fout:
    fout.write("| Model |  MAE (5-fold) |  R² (5-fold) |\n|-------|--------------|-------------|\n")
    for name, mdl in models.items():
        neg_mae = cross_val_score(mdl, X, y, cv=cv5, scoring="neg_mean_absolute_error")
        r2      = cross_val_score(mdl, X, y, cv=cv5, scoring="r2")
        fout.write(f"| {name} | {(-neg_mae.mean()):.3f} | {r2.mean():.3f} |\n")
        mdl.fit(X, y)
        joblib.dump(mdl, f"models/model_{name}.pkl")
print("Baseline training done → results_baseline.md")
