from __future__ import annotations
import argparse
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
import config

def parse_args():
    p = argparse.ArgumentParser("SHAP explanation for trained XGB ranker")
    p.add_argument("--data", default="scripts/outputs/rank_data_v2.pkl",
                   help="Path to bundle made by make_rank_data.py")
    p.add_argument("--model", default="scripts/outputs/xgb_ranker_v2.pkl",
                   help="Path to saved ranker model (joblib)")
    p.add_argument("--out-dir", default="scripts/outputs/shap",
                   help="Directory to save plots")
    p.add_argument("--sample", type=int, default=2000,
                   help="Subsample rows for faster SHAP (0=all)")
    return p.parse_args()

def main():
    args = parse_args()
    bundle = joblib.load(args.data)
    model_bundle = joblib.load(args.model)
    model = model_bundle["model"]
    feat_names = model_bundle.get("features", bundle["features"])
    X = bundle["X"]
    df = bundle["df"]

    if args.sample > 0 and X.shape[0] > args.sample:
        idx = np.random.RandomState(42).choice(X.shape[0], size=args.sample, replace=False)
        X_sample = X[idx]
    else:
        idx = np.arange(X.shape[0])
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        shap_values, X_sample, feature_names=feat_names, plot_type="bar", show=False
    )
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bar.png", dpi=200)
    plt.close()

    shap.summary_plot(
        shap_values, X_sample, feature_names=feat_names, show=False
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=200)
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    fi = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
    fi.sort_values("mean_abs_shap", ascending=False).to_csv(out_dir / "feature_importance.csv", index=False)

    print(f"[shap] saved plots & csv to {out_dir}")

if __name__ == "__main__":
    main()