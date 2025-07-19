import json, argparse, pathlib, joblib, subprocess, time
import pynvml

CANDIDATES = [2, 4, 8, 16]
DELTA_MAP = 0.01    # 1%

def run_once(model_pt: str, out_json: pathlib.Path):
    """Run YOLO train for one epoch with batch=1, save metrics to JSON."""
    cmd = [
        "yolo", "train",
        f"model={model_pt}", "data=coco128.yaml",
        "epochs=1", "batch=1", "device=0", "verbose=False",
        "project=temp_predict", "name=tmp", "exist_ok=True"
    ]
    subprocess.run(cmd, check=True)
    res_json = pathlib.Path("temp_predict/tmp/results.json")
    with res_json.open() as f: res = json.load(f)
    map50  = res["metrics"][-1]["metrics/mAP50"]
    pwr_W  = res["train/avg_power"]        # Ultralytics >=8.3 有
    mem_MB = res["train/avg_mem"]
    payload = {"baseline_power": pwr_W, "baseline_mem": mem_MB, "baseline_map": map50}
    out_json.write_text(json.dumps(payload))
    return payload

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    tmp_json = pathlib.Path("baseline_tmp.json")
    base = run_once(args.model, tmp_json)

    model = joblib.load("models/model.pkl")
    rows = []
    for b in CANDIDATES:
        X = [[b, base["baseline_power"], base["baseline_mem"], base["baseline_map"]]]
        e_pred = model["E"].predict(X)[0]
        a_pred = model["A"].predict(X)[0]
        rows.append((b, e_pred, a_pred))

    # filter by accuracy drop
    valid = [r for r in rows if r[2] >= base["baseline_map"] - DELTA_MAP]
    if not valid:
        best = min(rows, key=lambda x: x[1])
    else:
        best = min(valid, key=lambda x: x[1])

    print(f"Best batch size: {best[0]}")
    print("Predictions:", best)

if __name__ == "__main__":
    main()