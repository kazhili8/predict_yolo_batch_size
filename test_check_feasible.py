import pandas as pd
for tag in ["115W","65W"]:
    df = pd.read_csv(f"scripts/outputs/policy_eval_{tag}_constrained.csv")
    print(tag, "N=", len(df))
    for who in ["zeus","greedy","random","table"]:
        t1 = df[f"top1_{who}"].mean()
        rg = df[f"regret_{who}"].mean()
        vr = df[f"vio_{who}"].mean()
        print(who, "Top1=", round(t1,3), "Regret=", round(rg,3), "Violation=", round(vr,3))