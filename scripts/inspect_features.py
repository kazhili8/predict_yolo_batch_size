import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Quick feature health check on the CSV.")
    p.add_argument("--csv", required=True, help="Path to features CSV (e.g., features_v5.csv)")
    p.add_argument("--group-cols", nargs="+", default=["model", "epochs", "tag"],
                   help="Columns that define a group/query")
    p.add_argument("--out", default="scripts/outputs/inspect_features.csv",
                   help="Where to save the summary CSV")
    return p.parse_args()

def safe_isfinite(x: pd.Series) -> float:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    return float(np.isfinite(arr).mean())

def summarize_numeric(df: pd.DataFrame, group_cols: list[str], can_group: bool) -> pd.DataFrame:
    num_df = df.select_dtypes(include=[np.number]).copy()
    rows = []
    n = len(num_df)

    grouped = None
    if can_group:
        use_cols = list(dict.fromkeys([*(c for c in group_cols if c in df.columns), *num_df.columns]))
        sub = df[use_cols].copy()
        if not sub.columns.is_unique:
            dup = sub.columns[sub.columns.duplicated()].tolist()
            print(f"[inspect][WARN] duplicated columns in subframe: {dup} (will still group by unique names)")
            sub = sub.loc[:, ~sub.columns.duplicated()]
        grouped = sub.groupby(group_cols)

    for col in num_df.columns:
        s = num_df[col]
        non_na = int(s.notna().sum())
        nunique = int(s.nunique(dropna=True))
        mean = float(s.mean()) if non_na else np.nan
        std = float(s.std(ddof=0)) if non_na else np.nan
        minimum = float(s.min()) if non_na else np.nan
        q25 = float(s.quantile(0.25)) if non_na else np.nan
        median = float(s.median()) if non_na else np.nan
        q75 = float(s.quantile(0.75)) if non_na else np.nan
        maximum = float(s.max()) if non_na else np.nan
        zero_frac = float((s == 0).mean()) if non_na else np.nan
        pos_frac = float((s > 0).mean()) if non_na else np.nan
        neg_frac = float((s < 0).mean()) if non_na else np.nan
        finite_frac = safe_isfinite(s)

        if can_group and non_na and std > 0:
            gm = grouped[col].mean()
            between_var = float(gm.var(ddof=0)) if len(gm) > 1 else 0.0
            total_var = float(s.var(ddof=0))
            between_ratio = float(between_var / (total_var + 1e-12))
            gwstd = grouped[col].std(ddof=0)
            within_std_mean = float(gwstd.mean(skipna=True))
        else:
            between_ratio = np.nan
            within_std_mean = np.nan

        rows.append({
            "feature": col,
            "count": n,
            "non_na": non_na,
            "non_na_ratio": round(non_na / max(n, 1), 6),
            "n_unique": nunique,
            "mean": mean,
            "std": std,
            "min": minimum,
            "p25": q25,
            "median": median,
            "p75": q75,
            "max": maximum,
            "zero_frac": zero_frac,
            "pos_frac": pos_frac,
            "neg_frac": neg_frac,
            "finite_frac": finite_frac,
            "between_group_var_ratio": between_ratio,
            "within_group_std_mean": within_std_mean,
        })
    out = pd.DataFrame(rows).sort_values(["std", "n_unique"], ascending=[False, False])
    return out

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    df.columns = df.columns.map(lambda c: str(c).strip())
    if not df.columns.is_unique:
        dup = df.columns[df.columns.duplicated()].tolist()
        print(f"[inspect][WARN] duplicated columns detected in CSV: {dup}. Keeping first occurrence.")
        df = df.loc[:, ~df.columns.duplicated()]

    group_cols = list(dict.fromkeys(args.group_cols))
    missing = [c for c in group_cols if c not in df.columns]
    can_group = len(missing) == 0 and df[group_cols].notna().all(axis=1).all()
    if not can_group:
        print(f"[inspect][WARN] cannot group by {group_cols}. Missing/invalid: {missing}. "
              f"Will compute global stats only.")

    summary = summarize_numeric(df, group_cols, can_group)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"[inspect] saved → {args.out}  (rows={len(summary)})")

if __name__ == "__main__":
    main()
