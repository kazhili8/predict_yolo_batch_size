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
    """Load the feature CSV and sanity‑check required columns."""
    df = pd.read_csv(csv_path)
    if "avg_map50" not in df.columns:
        if "avg_map" in df.columns:
            df["avg_map50"] = df["avg_map"]
        else:
            raise ValueError("Missing required column: avg_map50 or avg_map")

    expected_cols = {"batch", "avg_power", "avg_mem", "avg_map50"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def compute_baseline(df: pd.DataFrame) -> pd.Series:
    """Return baseline (mean of numeric columns) for batch == 1."""
    baseline = df[df.batch == 1].mean(numeric_only=True)
    logging.info("Baseline computed from %d samples (batch == 1)", (df.batch == 1).sum())
    return baseline

def build_feature_matrix(
    df: pd.DataFrame, baseline: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return design matrix X and targets (y_power, y_map50) for batches != 1."""
    non1 = df[df.batch != 1].copy()

    non1["batch"] = non1["batch"].astype("category")
    non1["batch_code"] = non1["batch"].cat.codes

    non1["delta_power"]  = non1.avg_power  - baseline.avg_power
    non1["delta_mem"]    = non1.avg_mem    - baseline.avg_mem
    non1["delta_map50"]  = non1.avg_map50  - baseline.avg_map50

    feature_cols = ["batch_code", "delta_power", "delta_mem", "delta_map50"]
    X = non1[feature_cols]
    y_power = non1["avg_power"]
    y_map50 = non1["avg_map50"]

    return X, y_power, y_map50

def train_and_eval(X: pd.DataFrame, y: pd.Series, label: str) -> ExtraTreesRegressor:
    """Train Extra‑Trees on (X, y) and log hold‑out performance."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    regr = ExtraTreesRegressor(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(
        "[%s]  MAE=%.4f  R2=%.4f  (n_test=%d)",
        label, mae, r2, len(y_test)
    )
    return regr


def main() -> None:
    df = load_features(r"D:\Predict_YOLO_batch_size\scripts\outputs\dataframe\features.csv")
    baseline = compute_baseline(df)
    X, y_power, y_map50 = build_feature_matrix(df, baseline)

    re_e   = train_and_eval(X, y_power,  "avg_power")
    re_a50 = train_and_eval(X, y_map50, "avg_map50")

    payload = {
        "E": re_e,
        "map50": re_a50,
        "baseline_power":  float(baseline.avg_power),
        "baseline_mem":    float(baseline.avg_mem),
        "baseline_map50":  float(baseline.avg_map50),
        "feature_order":   ["batch_code", "delta_power", "delta_mem", "delta_map50"],
        "meta": {
            "n_samples": int(len(X)),
            "n_trees":   N_TREES,
            "random_state": RANDOM_STATE,
        },
    }
    out_path = MODELS_DIR / "model.pkl"
    joblib.dump(payload, out_path)
    logging.info("Model dict saved -> %s", out_path.resolve())


if __name__ == "__main__":
    main()
