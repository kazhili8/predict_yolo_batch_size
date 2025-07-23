import argparse, json, pathlib, statistics, tqdm, pandas as pd

def collect_features(json_dir: pathlib.Path):
    rows = []
    for p in tqdm.tqdm(list(json_dir.glob("*.json")), desc="scan"):
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        batch = obj.get("batch") or obj.get("batch_size")
        epochs = obj.get("epochs") or obj.get("epoch") or obj.get("num_epochs")
        map50 = obj.get("avg_map50") or obj.get("map50")

        row = {
            "json_file": p.name,
            "model": obj.get("model"),
            "batch": batch,
            "epochs": epochs,
            "tag": obj.get("tag", ""),
            "avg_step_time": obj.get("avg_step_time"),
            "avg_power": obj.get("avg_power"),
            "avg_mem": obj.get("avg_mem"),
            "map50": map50,
        }
        series = obj.get("power_series", [])
        if series:
            row["pwr_mean"] = statistics.mean(series)
            row["pwr_std"]  = statistics.pstdev(series)
            row["pwr_max"]  = max(series)
            row["pwr_min"]  = min(series)
        else:
            row.update({"pwr_mean": None, "pwr_std": None,
                        "pwr_max": None,  "pwr_min": None})

        if row["avg_step_time"]:
            row["throughput"] = row["batch"] / row["avg_step_time"]
        else:
            row["throughput"] = None

        if row["throughput"] and row["avg_power"]:
            row["energy_per_img"] = row["avg_power"] / row["throughput"]
        else:
            row["energy_per_img"] = None

        rows.append(row)

    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = collect_features(pathlib.Path(args.json_dir))
    df.to_csv(args.out, index=False)
    print(f"saved → {args.out},  rows={len(df)}")

if __name__ == "__main__":
    main()
