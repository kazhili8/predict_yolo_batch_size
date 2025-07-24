import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path

CSV_PATH = Path("scripts/outputs/dataframe/features_v2.csv")
MODEL_PATH = Path("models/model_ranker.pkl")

FEATS_TARGET = ["batch", "throughput", "avg_mem", "pwr_mean", "pwr_std", "energy_per_img"]

MAP_CANDIDATES = ["map50", "avg_map50", "mAP50", "mAP@0.5"]

METRICS_OUT = Path("metrics_ranker.md")


def pick_map_column(df: pd.DataFrame) -> str:
    for c in MAP_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"No mAP column found. Expected one of: {MAP_CANDIDATES}. "
        "Please include a valid mAP@0.5 column in the CSV."
    )


def verify_features(df: pd.DataFrame) -> list[str]:
    missing = [c for c in FEATS_TARGET if c not in df.columns]
    if missing:
        raise ValueError(
            f"These feature columns are missing from the CSV: {missing}. "
            "Please add them or retrain / re-export accordingly."
        )
    return FEATS_TARGET


def compute_true_scores_per_group(sub: pd.DataFrame, map_col: str) -> pd.DataFrame | None:

    if sub[map_col].notna().sum() == 0:
        return None
    needed_for_score = ["throughput", "avg_power", "avg_mem", map_col]
    keep_mask = sub[needed_for_score].notna().all(axis=1)
    sub = sub.loc[keep_mask].copy()
    if sub.empty:
        return None

    sub["delta_map"] = sub[map_col].max() - sub[map_col]
    sub["true_score"] = (
        0.5 * sub["throughput"]
        - 0.3 * sub["avg_power"]
        - 0.1 * sub["avg_mem"]
        - 0.1 * sub["delta_map"]
    )
    return sub


def main():
    df = pd.read_csv(CSV_PATH)
    model = joblib.load(MODEL_PATH)
    map_col = pick_map_column(df)
    feats = verify_features(df)
    hits, total = 0, 0

    for (m, ep), gdf in df.groupby(["model", "epochs"], dropna=False):
        gdf_feats_ok = gdf.dropna(subset=feats).copy()
        if gdf_feats_ok.empty:
            continue

        scored = compute_true_scores_per_group(gdf_feats_ok, map_col)
        if scored is None or scored.empty:
            continue

        dmat = xgb.DMatrix(scored[feats].to_numpy())
        scored["pred_score"] = model.predict(dmat)

        top_pred_batch = scored.loc[scored["pred_score"].idxmax(), "batch"]
        top_true_batch = scored.loc[scored["true_score"].idxmax(), "batch"]

        hits += int(top_pred_batch == top_true_batch)
        total += 1

    acc = np.nan if total == 0 else hits / total

    METRICS_OUT.write_text(
        f"Best recommend accuracy = {acc:.2%}  ({hits}/{total})\n",
        encoding="utf-8"
    )
    print(f"Best accuracy written → {METRICS_OUT}")


if __name__ == "__main__":
    main()