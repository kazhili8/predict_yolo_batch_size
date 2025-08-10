import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--group_cols", nargs="+", default=["model","epochs","tag","batch"])
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    num_cols = df.select_dtypes(include=["number"]).columns.difference(args.group_cols)

    agg = df.groupby(args.group_cols, as_index=False).agg({c: "mean" for c in num_cols})
    cnt = df.groupby(args.group_cols, as_index=False).size().rename(columns={"size":"n_logs"})
    out = agg.merge(cnt, on=args.group_cols, how="left")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    if out_path.name.endswith("_agg.csv"):
        mirror = out_path.with_name(out_path.name.replace("_agg.csv", "_agg_clean.csv"))
        out.to_csv(mirror, index=False)

    print(f"[aggregate] saved → {args.out_csv}, rows={len(out)}, from raw rows={len(df)}")

if __name__ == "__main__":
    main()
