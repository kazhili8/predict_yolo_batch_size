import pandas as pd, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--clean_65", required=True)
ap.add_argument("--raw_all", required=True)
ap.add_argument("--out_csv", required=True)
args = ap.parse_args()

clean = pd.read_csv(args.clean_65)
raw   = pd.read_csv(args.raw_all)

raw_b1_65 = raw[(raw["tag"]=="65W") & (raw["batch"]==1)]
keys = ["model","epochs","tag","batch"]
merged = pd.concat([clean, raw_b1_65], axis=0, ignore_index=True)
merged = merged.drop_duplicates(subset=keys, keep="first")
merged.to_csv(args.out_csv, index=False)