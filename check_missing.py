import pandas as pd
import itertools

df = pd.read_csv("scripts/outputs/dataframe/features_v6_agg.csv")

models = sorted(df["model"].unique())
epochs = sorted(df["epochs"].unique())
tags = sorted(df["tag"].unique())
batches = sorted(df["batch"].unique())

print("Available models:", models)
print("Available epochs:", epochs)
print("Available tags  :", tags)
print("Available batches:", batches)

existing_combinations = set(zip(df["model"], df["epochs"], df["tag"], df["batch"]))
expected_combinations = set(itertools.product(models, epochs, tags, batches))
missing_combinations = sorted(expected_combinations - existing_combinations)

print(f"Number of missing combinations: {len(missing_combinations)}")
for combo in missing_combinations[:50]:
    print(combo)

batch_counts = (
    df.groupby(["model", "epochs", "tag"])["batch"]
      .nunique()
      .reset_index(name="n_batch")
)
print(batch_counts.sort_values("n_batch"))