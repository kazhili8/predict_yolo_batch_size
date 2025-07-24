import argparse, joblib, pandas as pd, numpy as np, xgboost as xgb

FEATS = ["batch", "throughput", "avg_mem", "pwr_mean", "pwr_std", "energy_per_img"]
def to_int_safe(x):
    try:
        return int(float(x))
    except Exception:
        return np.nan

def main():
    ap = argparse.ArgumentParser(
        description="Show ranker-predicted best batch for each (model, epochs)."
    )
    ap.add_argument("--csv", default="scripts/outputs/dataframe/features_v2.csv")
    ap.add_argument("--model_path", default="models/model_ranker.pkl")
    ap.add_argument("--model_key", required=True,
                    help="Substring to match model name, e.g. 'yolo11n'")
    ap.add_argument("--epoch", type=int, default=None,
                    help="Specific epoch; if omitted, loop over all epochs")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["epochs_int"] = df["epochs"].apply(to_int_safe)
    df = df[df["model"].astype(str).str.contains(args.model_key, regex=False, na=False)]

    if df.empty:
        raise ValueError(f"No rows match model contains '{args.model_key}'")

    ranker = joblib.load(args.model_path)

    def run_group(sub):
        dmat = xgb.DMatrix(sub[FEATS].values)
        sub = sub.copy()
        sub["pred_score"] = ranker.predict(dmat)
        return sub.sort_values("pred_score", ascending=False)

    if args.epoch is not None:
        sub = df[df["epochs_int"] == args.epoch]
        if sub.empty:
            raise ValueError(f"No rows match epochs == {args.epoch}")
        print(f"\n=== model={args.model_key}  epochs={args.epoch} ===")
        print(run_group(sub)[["batch", "pred_score"]].head(args.topk))
        return

    for ep, sub in df.groupby("epochs_int", sort=True):
        print(f"\n=== model={args.model_key}  epochs={ep} ===")
        print(run_group(sub)[["batch", "pred_score"]].head(args.topk))

if __name__ == "__main__":
    main()
