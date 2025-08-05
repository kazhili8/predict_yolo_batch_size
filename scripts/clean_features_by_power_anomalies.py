from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser("Filter aggregated features by removing power-anomalous rows.")
    p.add_argument("--features_agg", required=True,
                   help="Aggregated features CSV (e.g., features_v7_agg.csv).")
    p.add_argument("--avg_anoms", required=True,
                   help="CSV from audit_power_consistency.py (avg anomalies).")
    p.add_argument("--peak_anoms", required=True,
                   help="CSV from audit_power_consistency.py (peak anomalies).")
    p.add_argument("--out_csv", required=True,
                   help="Output cleaned features CSV (e.g., features_v7_agg_clean.csv).")
    p.add_argument("--drop_which", default="both", choices=["avg", "peak", "both"],
                   help="Which anomaly list to use for filtering.")
    p.add_argument("--assume_tag_if_missing", default="65W",
                   help="If anomaly CSV has no 'tag' column, assume anomalies belong to this tag.")
    return p.parse_args()

def load_anomaly_keys(path: str, assume_tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    def pick(colname):
        if colname in cols_lower:
            return cols_lower[colname]
        raise ValueError(f"Column '{colname}' not found in anomaly CSV {path}.")

    model_col  = pick("model")
    epochs_col = pick("epochs")
    batch_col  = pick("batch")

    tag_col = cols_lower.get("tag", None)

    out = pd.DataFrame({
        "model":  df[model_col].astype(str),
        "epochs": df[epochs_col].astype(int),
        "batch":  df[batch_col].astype(int),
    })
    if tag_col is not None:
        out["tag"] = df[tag_col].astype(str)
    else:
        out["tag"] = assume_tag
    out = out[["model","epochs","tag","batch"]].drop_duplicates()
    return out

def main():
    args = parse_args()

    feats = pd.read_csv(args.features_agg)
    required = ["model","epochs","tag","batch"]
    for c in required:
        if c not in feats.columns:
            raise ValueError(f"Column '{c}' not found in {args.features_agg}")

    avg_df  = load_anomaly_keys(args.avg_anoms,  assume_tag=args.assume_tag_if_missing)
    peak_df = load_anomaly_keys(args.peak_anoms, assume_tag=args.assume_tag_if_missing)

    if args.drop_which == "avg":
        blk = avg_df
    elif args.drop_which == "peak":
        blk = peak_df
    else:
        blk = pd.concat([avg_df, peak_df], ignore_index=True).drop_duplicates()

    feats["_key"] = (feats["model"].astype(str) + "|" + feats["epochs"].astype(str) +
                     "|" + feats["tag"].astype(str) + "|" + feats["batch"].astype(str))
    blk["_key"] = (blk["model"].astype(str) + "|" + blk["epochs"].astype(str) +
                   "|" + blk["tag"].astype(str) + "|" + blk["batch"].astype(str))

    before = len(feats)
    feats_clean = feats[~feats["_key"].isin(set(blk["_key"]))].drop(columns=["_key"])
    removed = before - len(feats_clean)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    feats_clean.to_csv(args.out_csv, index=False)
    print(f"[clean] input={before}, removed={removed}, output={len(feats_clean)}")
    print(f"[save] {args.out_csv}")

if __name__ == "__main__":
    main()