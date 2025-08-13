import pandas as pd, pathlib
out=[]
for tag in ["115W","65W"]:
    df=pd.read_csv(f"scripts/outputs/policy_eval_{tag}_constrained.csv")
    out.append(f"## {tag} (N={len(df)})")
    for who in ["zeus","greedy","random","table"]:
        out.append(f"- {who}: Top1={df[f'top1_{who}'].mean():.3f}, Regret={df[f'regret_{who}'].mean():.3f}, Violation={df[f'vio_{who}'].mean():.3f}")
path=pathlib.Path("scripts/outputs/policy_eval_score_summary.md")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("\n".join(out), encoding="utf-8")
print("written →", path)

