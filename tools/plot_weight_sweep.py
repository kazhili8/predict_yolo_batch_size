import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("metrics_weight_sweep.csv")
labels = [f"{t}:{p}:{m}:{d}" for t,p,m,d in df[["T","P","M","Δ"]].values]

plt.figure(figsize=(6,3))
plt.plot(df.index, df["Top1"]*100, marker="o")
plt.xticks(df.index, labels, rotation=40, ha="right", fontsize=8)
plt.ylabel("Top-1 accuracy (%)")
plt.ylim(0, 100); plt.tight_layout()
plt.savefig("weight_sweep.png", dpi=150)
print("picture has saved to weight_sweep.png")