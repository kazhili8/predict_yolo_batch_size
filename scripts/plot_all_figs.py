import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def load_df(p):
    df = pd.read_csv(p)
    if "throughput" not in df.columns and "avg_step_time" in df.columns:
        df = df.copy(); df["throughput"] = 1.0/df["avg_step_time"].astype(float)
    return df

def bar_from_policy(eval_csv, title, out_png):
    df = pd.read_csv(eval_csv)
    agg = {}
    for who in ["zeus","greedy","random","table"]:
        agg[who] = dict(
            top1 = df[f"top1_{who}"].mean(),
            regret = df[f"regret_{who}"].mean(),
            vio = df[f"vio_{who}"].mean()
        )
    xs = list(agg.keys())
    top1 = [agg[x]["top1"]*100 for x in xs]
    regret = [agg[x]["regret"]*100 for x in xs]
    vio = [agg[x]["vio"]*100 for x in xs]
    plt.figure(figsize=(7.5,3.2))
    plt.subplot(1,3,1); plt.bar(xs, top1); plt.title("Top-1 (%)"); plt.ylim(0,100)
    plt.subplot(1,3,2); plt.bar(xs, regret); plt.title("Regret (%)"); plt.ylim(0,100)
    plt.subplot(1,3,3); plt.bar(xs, vio); plt.title("Violation (%)"); plt.ylim(0,100)
    plt.suptitle(title); plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180); plt.close()

def scatter_tp_power(df, title, out_png):
    plt.figure(figsize=(5.5,4.2))
    for b,g in df.groupby("batch"):
        plt.scatter(g["avg_power"], g["throughput"], s=12, label=str(b), alpha=0.6)
    plt.xlabel("Average Power (W)"); plt.ylabel("Throughput (img/s)"); plt.title(title)
    plt.tight_layout(); Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180); plt.close()

def plot_weight_sweep(csv_path, out_png):
    df = pd.read_csv(csv_path)
    if "top1" in df.columns:
        plt.figure(figsize=(6,3.6))
        plt.plot(range(len(df)), df["top1"].values, marker="o", linewidth=1)
        plt.xlabel("Weight set index"); plt.ylabel("Top-1")
        plt.title("Weight sweep (Top-1)")
        plt.tight_layout(); Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=180); plt.close()

def plot_oof(oof_csv, title, out_png):
    df = pd.read_csv(oof_csv)
    if "top1" in df.columns:
        v = float(df["top1"].mean()) if pd.api.types.is_numeric_dtype(df["top1"]) else None
        plt.figure(figsize=(4.8,3))
        plt.bar([title], [100.0*v if v is not None else 0.0])
        plt.ylabel("Top-1 (%)"); plt.tight_layout()
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=180); plt.close()

def plot_epoch_consistency(csv_path, title, out_png):
    df = pd.read_csv(csv_path)
    if {"from","to","same_best_rate"}.issubset(df.columns):
        piv = df.pivot(index="from", columns="to", values="same_best_rate").sort_index()
        plt.figure(figsize=(4.6,4))
        plt.imshow(piv.values, aspect="auto")
        plt.xticks(range(len(piv.columns)), piv.columns)
        plt.yticks(range(len(piv.index)), piv.index)
        plt.title(title); plt.colorbar(label="Same best batch (%)")
        plt.tight_layout(); Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=180); plt.close()

def plot_carbon(csv_path, out_png):
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.lower().endswith("gco2")]
    if len(cols)==0: return
    plt.figure(figsize=(6,3.6))
    plt.bar(range(len(df)), df[cols[0]].values)
    plt.title("Estimated CO2 by run"); plt.ylabel("gCO2")
    plt.tight_layout(); Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--policy_csv_65", required=True)
    ap.add_argument("--policy_csv_115", required=True)
    ap.add_argument("--metrics_weight_sweep", default="")
    ap.add_argument("--oof_all", default="")
    ap.add_argument("--oof_leave_model", default="")
    ap.add_argument("--oof_leave_tag", default="")
    ap.add_argument("--oof_leave_epochs", default="")
    ap.add_argument("--epoch_csv_65", default="")
    ap.add_argument("--epoch_csv_115", default="")
    ap.add_argument("--carbon_csv", default="")
    ap.add_argument("--shap_dir", default="")
    ap.add_argument("--out_dir", default="scripts/outputs/figs")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    f = load_df(args.features_csv)

    bar_from_policy(args.policy_csv_65, "Policies @65W (score-oracle)", str(out/"policies_65W.png"))
    bar_from_policy(args.policy_csv_115, "Policies @115W (score-oracle)", str(out/"policies_115W.png"))

    for m in sorted(f["model"].unique().tolist()):
        dfm = f[f["model"]==m]
        scatter_tp_power(dfm[dfm["tag"]=="65W"], f"{m} @65W", str(out/f"tp_power_{m}_65W.png"))
        scatter_tp_power(dfm[dfm["tag"]=="115W"], f"{m} @115W", str(out/f"tp_power_{m}_115W.png"))

    if args.metrics_weight_sweep and Path(args.metrics_weight_sweep).exists():
        plot_weight_sweep(args.metrics_weight_sweep, str(out/"weight_sweep_top1.png"))

    if args.oof_all and Path(args.oof_all).exists():
        plot_oof(args.oof_all, "OOF (All)", str(out/"oof_all.png"))
    if args.oof_leave_model and Path(args.oof_leave_model).exists():
        plot_oof(args.oof_leave_model, "OOF (Leave-Model)", str(out/"oof_leave_model.png"))
    if args.oof_leave_tag and Path(args.oof_leave_tag).exists():
        plot_oof(args.oof_leave_tag, "OOF (Leave-Tag)", str(out/"oof_leave_tag.png"))
    if args.oof_leave_epochs and Path(args.oof_leave_epochs).exists():
        plot_oof(args.oof_leave_epochs, "OOF (Leave-Epochs)", str(out/"oof_leave_epochs.png"))

    if args.epoch_csv_65 and Path(args.epoch_csv_65).exists():
        plot_epoch_consistency(args.epoch_csv_65, "Epoch consistency @65W", str(out/"epoch_consistency_65W.png"))
    if args.epoch_csv_115 and Path(args.epoch_csv_115).exists():
        plot_epoch_consistency(args.epoch_csv_115, "Epoch consistency @115W", str(out/"epoch_consistency_115W.png"))

    if args.carbon_csv and Path(args.carbon_csv).exists():
        plot_carbon(args.carbon_csv, str(out/"carbon_runs.png"))

    if args.shap_dir:
        for fn in ["shap_bar.png","shap_beeswarm.png"]:
            p = Path(args.shap_dir)/fn
            if p.exists():
                (out/f"{fn}").write_bytes(p.read_bytes())

if __name__ == "__main__":
    main()
