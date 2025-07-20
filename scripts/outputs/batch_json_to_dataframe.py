import pandas as pd
from pathlib import Path
import argparse, json

def extract_metrics(fp: Path) -> dict:
    data = json.loads(fp.read_text())
    batch_key, metrics = next(iter(data["metrics_per_batch"].items()))
    return {
        "model"  : data["model"],
        "batch"  : int(batch_key),
        "avg_step_time": metrics["avg_step_time"],
        "avg_power"    : metrics["avg_power"],
        "avg_mem"      : metrics["avg_mem"],
        "avg_map"      : metrics.get("avg_map50") or metrics.get("avg_map"),
        "avg_map95"    : metrics.get("avg_map50-95") or metrics.get("avg_map95"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True,
                    help="folder containing *_unified.json")
    ap.add_argument("--out", default=r"D:\Predict_YOLO_batch_size\scripts\outputs\dataframe\features.csv",
                    help="rD:\Predict_YOLO_batch_size\scripts\outputs\dataframe\features.csv")
    args = ap.parse_args()

    rows = []
    for fp in Path(args.json_dir).glob("*.json"):
        try:
            rows.append(extract_metrics(fp))
        except Exception as e:
            print("skip", fp.name, "->", e)

    df = pd.DataFrame(rows).sort_values(["model", "batch"])
    df.to_csv(args.out, index=False)
    print(f"saved -> {args.out},  rows={len(df)}")

if __name__ == "__main__":
    main()