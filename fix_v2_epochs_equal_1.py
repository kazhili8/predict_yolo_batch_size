import pandas as pd, pathlib, numpy as np

CSV = pathlib.Path(r"scripts/outputs/dataframe/features_v4.csv")
df  = pd.read_csv(CSV)

df["epochs"] = df["epochs"].fillna(1).astype(int)

print("unique epochs:", sorted(df["epochs"].unique()))
df.to_csv(CSV, index=False)
print("File have changed")