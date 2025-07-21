import json, argparse, pathlib, joblib, subprocess, shutil
import pynvml
import pandas as pd

CANDIDATES = [2, 4, 8, 16]
DELTA_MAP = 0.01

def _read_map_from_results(res_dir: pathlib.Path) -> float:
    json_path = res_dir / "results.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        last_rec = data["metrics"][-1]
        for k, v in last_rec.items():
            if "mAP50" in k:
                return float(v)

    for p in res_dir.glob("results*.csv"):
        df = pd.read_csv(p)
        for col in df.columns:
            if "mAP50" in col:
                return float(df[col].iloc[-1])

    raise FileNotFoundError("mAP50 not found in results.json or results.csv")


def run_once(model_pt: str, out_json: pathlib.Path):
    """Run YOLO train for one epoch with batch=1, save baseline metrics."""
    cmd = [
        "yolo", "train",
        f"model={model_pt}", "data=coco128.yaml",
        "epochs=1", "batch=1", "device=0", "verbose=False",
        "project=temp_predict", "name=tmp", "exist_ok=True"
    ]
    subprocess.run(cmd, check=True)

    res_dir = pathlib.Path("temp_predict/tmp")
    map50  = _read_map_from_results(res_dir)

    stats_path = res_dir / "results.json"
    power = mem = None
    if stats_path.exists():
        j = json.loads(stats_path.read_text())
        power = j["train/avg_power"]
        mem   = j["train/avg_mem"]
    if power is None:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20
        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000

    payload = {
        "baseline_power":  power,
        "baseline_mem":    mem,
        "baseline_map50":  map50
    }
    out_json.write_text(json.dumps(payload))

    shutil.rmtree("temp_predict", ignore_errors=True)
    return payload

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    tmp_json = pathlib.Path("baseline_tmp.json")
    base = run_once(args.model, tmp_json)

    model = joblib.load("models/model.pkl")

    batch_codes = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
    rows = []

    for b in CANDIDATES:
        dp   = base["baseline_power"]  - model["baseline_power"]
        dm   = base["baseline_mem"]    - model["baseline_mem"]
        dmap = base["baseline_map50"]  - model["baseline_map50"]

        X = [[batch_codes[b], dp, dm, dmap]]
        e_pred   = model["E"].predict(X)[0]
        map_pred = model["map50"].predict(X)[0]
        rows.append((b, e_pred, map_pred))

    limit = base["baseline_map50"] * (1 - DELTA_MAP)
    valid = [r for r in rows if r[2] >= limit]
    best  = min(valid or rows, key=lambda x: x[1])

    print("\nPredictions (W / mAP50):")
    for b, pwr, mp in rows:
        flag = "*" if b == best[0] else " "
        print(f"{flag} b{b:<2}  {pwr:6.2f} W   {mp:.4f}")
    print(f"\nBest batch size (≤1% drop): {best[0]}")

    tmp_json.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
