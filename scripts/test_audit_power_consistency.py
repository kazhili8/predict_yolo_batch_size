from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser("Audit power consistency between 65W and 115W runs.")
    p.add_argument("--json_dir", required=True, help="Folder that contains raw *.json logs")
    p.add_argument("--out_csv", required=True, help="Where to save the audit csv (base name).")
    p.add_argument("--peak_thresh_65W", type=float, default=85.0,
                   help="Flag 65W rows whose peak_power exceeds this (W).")
    p.add_argument("--avg_margin", type=float, default=5.0,
                   help="Flag rows where avg_power(65W) >= avg_power(115W) - margin.")
    return p.parse_args()

def _iter_json_rows(json_dir: str):
    json_dir = Path(json_dir)
    for p in json_dir.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue
        model = obj.get("model") or obj.get("weights") or ""
        epochs = int(obj.get("epochs", 0))
        batch  = int(obj.get("batch_size", obj.get("batch", 0)))
        tag    = str(obj.get("tag", "")).strip()
        avg_power  = obj.get("avg_power")
        peak_power = obj.get("peak_power")
        if not model or not epochs or not batch:
            name = p.stem
            try:
                parts = name.split("_")
                if not model:  model  = parts[0]
                for part in parts:
                    if part.startswith("b"):
                        batch = int(part[1:])
                    if part.startswith("e"):
                        epochs = int(part[1:])
            except Exception:
                pass
        yield dict(path=str(p), model=model, epochs=epochs, batch=batch, tag=tag,
                   avg_power=avg_power, peak_power=peak_power)

def main():
    args = parse_args()
    rows = list(_iter_json_rows(args.json_dir))
    df = pd.DataFrame(rows)
    df = df[["model","epochs","batch","tag","avg_power","peak_power","path"]].copy()
    df["tag"] = df["tag"].astype(str).str.strip()
    df = df[df["tag"].isin(["65W","115W"])].copy()

    wide_avg = (df.pivot_table(
        index=["model","epochs","batch"], columns="tag", values="avg_power", aggfunc="mean"
    ).reset_index())
    wide_peak = (df.pivot_table(
        index=["model","epochs","batch"], columns="tag", values="peak_power", aggfunc="max"
    ).reset_index())

    for col in ["65W","115W"]:
        if col not in wide_avg.columns:
            wide_avg[col] = pd.NA
        if col not in wide_peak.columns:
            wide_peak[col] = pd.NA

    print("[debug] wide_avg columns:", list(wide_avg.columns))
    print("[debug] wide_peak columns:", list(wide_peak.columns))

    too_high_peak = wide_peak[wide_peak["65W"].astype("float64", errors="ignore") > args.peak_thresh_65W].copy()

    m = wide_avg[wide_avg["65W"].notna() & wide_avg["115W"].notna()].copy()
    suspicious_avg = m[m["65W"] >= (m["115W"] - args.avg_margin)].copy()

    out_base = Path(args.out_csv)
    out_dir = out_base.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    too_high_peak.to_csv(str(out_base).replace(".csv", "_peak_anomalies.csv"), index=False)
    suspicious_avg.to_csv(str(out_base).replace(".csv", "_avg_anomalies.csv"), index=False)

    summary = pd.DataFrame({
        "n_rows_raw":[len(df)],
        "n_groups":[len(wide_avg)],
        "n_peak_anomalies":[len(too_high_peak)],
        "n_avg_anomalies":[len(suspicious_avg)],
        "peak_thresh_65W":[args.peak_thresh_65W],
        "avg_margin":[args.avg_margin],
    })
    summary.to_csv(str(out_base).replace(".csv", "_summary.csv"), index=False)

    print(f"[done] raw rows={len(df)}, groups={len(wide_avg)}, "
          f"peak_anomalies={len(too_high_peak)}, avg_anomalies={len(suspicious_avg)}")
    print(f"[save] {str(out_base).replace('.csv','_peak_anomalies.csv')}")
    print(f"[save] {str(out_base).replace('.csv','_avg_anomalies.csv')}")
    print(f"[save] {str(out_base).replace('.csv','_summary.csv')}")

if __name__ == "__main__":
    main()
