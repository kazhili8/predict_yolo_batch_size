import pandas as pd
from pathlib import Path
import argparse, json

def extract_metrics(fp: Path) -> list[dict]:
    j = json.loads(fp.read_text())

    model  = j.get("model", "unknown")
    epochs = int(j.get("epochs", 1))
    suf    = f"_e{epochs}"

    rows = []
    mpb = j.get("metrics_per_batch", {})
    if not mpb:
        raise KeyError("'metrics_per_batch' missing")

    for batch_key, m in mpb.items():
        row = {
            "model": model,
            "batch": int(batch_key),
            f"avg_step_time{suf}": m.get("avg_step_time"),
            f"avg_power{suf}"    : m.get("avg_power"),
            f"avg_mem{suf}"      : m.get("avg_mem"),
            f"avg_map50{suf}"    : m.get("avg_map50") or m.get("avg_map"),
            f"avg_map95{suf}"    : m.get("avg_map50-95") or m.get("avg_map95"),
        }
        rows.append(row)

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True,
                    help="folder containing *_unified.json")
    ap.add_argument("--out", default=r"D:\Predict_YOLO_batch_size\scripts\outputs\dataframe\features.csv",
                    help="path to save features.csv")
    args = ap.parse_args()

    rows = []
    for fp in Path(args.json_dir).glob("*.json"):
        try:
            rows.extend(extract_metrics(fp))
        except Exception as e:
            print("skip", fp.name, "->", e)

    df = pd.DataFrame(rows).sort_values(["model", "batch"])
    df.to_csv(args.out, index=False)
    print(f"saved -> {args.out},  rows={len(df)}")

if __name__ == "__main__":
    main()
