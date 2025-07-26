import argparse
import pandas as pd

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

    out.to_csv(args.out_csv, index=False)
    print(f"[aggregate] saved → {args.out_csv}, rows={len(out)}, from raw rows={len(df)}")

if __name__ == "__main__":
    main()
