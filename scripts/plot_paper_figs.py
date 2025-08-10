import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def read_policy_means(csv115, csv65):
    d115 = pd.read_csv(csv115)
    d65 = pd.read_csv(csv65)
    rows=[]
    for tag,df in [("115W",d115),("65W",d65)]:
        for who in ["zeus","greedy","random","table"]:
            rows.append({"tag":tag,"strategy":who,
                         "top1":df[f"top1_{who}"].mean(),
                         "regret":df[f"regret_{who}"].mean(),
                         "vio":df[f"vio_{who}"].mean()})
    out = pd.DataFrame(rows)
    out["top1"] = out["top1"].astype(float)
    out["regret"] = out["regret"].astype(float)
    out["vio"] = out["vio"].astype(float)
    return out

def parse_weight_sweep(path):
    df=pd.read_csv(path)
    y=None
    for c in ["top1","top1_acc","Top1","Top1_acc"]:
        if c in df.columns: y=c; break
    if y is None: raise RuntimeError("no top1 column in weight sweep")
    if "weights" in df.columns:
        df["label"]=df["weights"].astype(str)
    elif set(["T","P","M","A"]).issubset(df.columns):
        df["label"]=df.apply(lambda r:f"{r['T']:.2f},{r['P']:.2f},{r['M']:.2f},{r['A']:.2f}",axis=1)
    else:
        df["label"]=df.index.astype(str)
    out = df[["label",y]].rename(columns={y:"top1"}).copy()
    out["top1"]=out["top1"].astype(float)
    return out

def pick_map_col(df):
    for c in ["map50","avg_map50","mAP50","mAP@0.5"]:
        if c in df.columns: return c
    raise RuntimeError("no mAP column")

def oof_top1(oof_csv, weights):
    df=pd.read_csv(oof_csv)
    map_col=pick_map_col(df)
    wT,wP,wM,wD=weights
    hits=0; total=0
    for _,g in df.groupby(["model","epochs","tag"], sort=False):
        g=g.copy()
        g["delta_map"]=g[map_col].max()-g[map_col]
        g["true_score"]=wT*g["throughput"]-wP*g["avg_power"]-wM*g["avg_mem"]-wD*g["delta_map"]
        if "rank_pred" not in g.columns or g["rank_pred"].isna().all():
            continue
        b_true=int(g.sort_values("true_score",ascending=False)["batch"].iloc[0])
        b_pred=int(g.sort_values("rank_pred",ascending=False)["batch"].iloc[0])
        hits+=int(b_true==b_pred); total+=1
    return 0.0 if total==0 else hits/total

def bar_policy(df, title, out_png):
    order=["zeus","greedy","random","table"]
    tags=["115W","65W"]
    p = df.pivot_table(index="strategy", columns="tag", values="top1", aggfunc="mean").reindex(index=order, columns=tags)
    p = p.fillna(0.0)
    xlabels=[]; y=[]
    for who in order:
        for tag in tags:
            xlabels.append(f"{who}-{tag}")
            y.append(float(p.loc[who, tag]))
    plt.figure()
    plt.bar(range(len(xlabels)), y)
    plt.xticks(range(len(xlabels)), xlabels, rotation=30, ha="right")
    plt.ylabel("Top-1")
    plt.title(title)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def line_weight_sweep(df, out_png):
    plt.figure()
    plt.plot(range(len(df)), df["top1"].values, marker="o")
    plt.xticks(range(len(df)), df["label"].tolist(), rotation=45, ha="right")
    plt.ylabel("Top-1")
    plt.title("Weight sweep (T,P,M,A)")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def bar_oof(overall_csv, leave_tag_csv, weights, out_png):
    v1=oof_top1(overall_csv, weights)
    v2=oof_top1(leave_tag_csv, weights)
    plt.figure()
    plt.bar([0,1],[v1,v2])
    plt.xticks([0,1],["OOF overall","OOF leave-tag"])
    plt.ylabel("Top-1")
    plt.title("OOF Top-1")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--score115", required=True)
    ap.add_argument("--score65", required=True)
    ap.add_argument("--thr115", required=True)
    ap.add_argument("--thr65", required=True)
    ap.add_argument("--sweep_csv", required=True)
    ap.add_argument("--oof_overall", required=True)
    ap.add_argument("--oof_leave_tag", required=True)
    ap.add_argument("--wT", type=float, default=0.60)
    ap.add_argument("--wP", type=float, default=0.20)
    ap.add_argument("--wM", type=float, default=0.10)
    ap.add_argument("--wA", type=float, default=0.10)
    ap.add_argument("--out_dir", default="scripts/outputs/figs")
    args=ap.parse_args()
    df_score=read_policy_means(args.score115, args.score65)
    bar_policy(df_score, "Policies vs. score-Oracle", str(Path(args.out_dir)/"policy_score.png"))
    df_thr=read_policy_means(args.thr115, args.thr65)
    bar_policy(df_thr, "Policies vs. throughput-Oracle", str(Path(args.out_dir)/"policy_throughput.png"))
    df_sw=parse_weight_sweep(args.sweep_csv)
    line_weight_sweep(df_sw, str(Path(args.out_dir)/"weight_sweep.png"))
    bar_oof(args.oof_overall, args.oof_leave_tag, (args.wT,args.wP,args.wM,args.wA), str(Path(args.out_dir)/"oof_bar.png"))

if __name__=="__main__":
    main()
