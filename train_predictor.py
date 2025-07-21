from pathlib import Path
from typing import Tuple
import joblib,logging
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
N_TREES = 300
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def load_features(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"batch", "avg_power_e1", "avg_mem_e1", "avg_map50_e1"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing cols: {miss}")
    return df

def compute_baseline(df: pd.DataFrame) -> pd.Series:
    baseline = df[df.batch == 1].mean(numeric_only=True)
    logging.info("Baseline computed from %d samples (batch==1)", (df.batch == 1).sum())
    return baseline

def build_feature_matrix(df: pd.DataFrame, baseline: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_non1 = df[df.batch != 1].copy()
    df_non1["batch"] = df_non1["batch"].astype("category")
    df_non1["batch_code"] = df_non1["batch"].cat.codes

    df_non1["delta_power"] = df_non1.avg_power_e1 - baseline.avg_power_e1
    df_non1["delta_mem"]   = df_non1.avg_mem_e1   - baseline.avg_mem_e1
    df_non1["delta_map50"] = df_non1.avg_map50_e1 - baseline.avg_map50_e1

    X = df_non1[["batch_code", "delta_power", "delta_mem", "delta_map50"]]
    return X, df_non1

def train_and_eval(X, y, label) -> ExtraTreesRegressor:
    mask = y.notna()
    X, y = X[mask], y[mask]
    if len(y) < 3:
        logging.warning("%s skipped (<=2 valid samples)", label)
        return None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    regr = ExtraTreesRegressor(
        n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    regr.fit(X_tr, y_tr)
    y_pred = regr.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    logging.info("[%s]  MAE=%.4f  R2=%.4f  (n_test=%d)",
                 label, mae, r2, len(y_te))
    return regr

def main():
    csv_path = r"D:\Predict_YOLO_batch_size\scripts\outputs\dataframe\features.csv"
    df = load_features(csv_path)
    baseline = compute_baseline(df)
    X, df_non1 = build_feature_matrix(df, baseline)

    y_p1   = df_non1["avg_power_e1"]
    y_p10  = df_non1.get("avg_power_e10")
    y_p100 = df_non1.get("avg_power_e100")
    y_map  = df_non1["avg_map50_e1"]

    models = {
        "E1":   train_and_eval(X, y_p1,   "power_1ep"),
        "E10":  train_and_eval(X, y_p10,  "power_10ep")  if y_p10  is not None else None,
        "E100": train_and_eval(X, y_p100, "power_100ep") if y_p100 is not None else None,
        "map50": train_and_eval(X, y_map, "map50"),
        "baseline_power": float(baseline.avg_power_e1),
        "baseline_mem":   float(baseline.avg_mem_e1),
        "baseline_map50": float(baseline.avg_map50_e1),
        "feature_order":  ["batch_code", "delta_power", "delta_mem", "delta_map50"],
    }

    out_path = MODELS_DIR / "model.pkl"
    joblib.dump(models, out_path)
    logging.info("Model dict saved -> %s", out_path.resolve())

if __name__ == "__main__":
    main()