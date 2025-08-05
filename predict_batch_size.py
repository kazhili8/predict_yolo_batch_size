import json, argparse, pathlib, joblib, subprocess, shutil
import pynvml
import pandas as pd
from scripts.tag_detect import detect_tag_by_nvml
batch_codes = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
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
        power = j.get("train/avg_power", None)
        mem   = j.get("train/avg_mem",   None)
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

def _try_lookup_from_table(model_name: str, tag: str, rec115_path: str, rec65_path: str):
    try:
        if tag == "115W":
            rec_path = rec115_path
        elif tag == "65W":
            rec_path = rec65_path
        else:
            return None
        rec_file = pathlib.Path(rec_path)
        if not rec_file.exists():
            return None
        df = pd.read_csv(rec_file)
        m = pathlib.Path(model_name).name
        df = df.copy()
        col_model = "model" if "model" in df.columns else None
        col_epochs = "epochs" if "epochs" in df.columns else None
        col_tag = "tag" if "tag" in df.columns else None
        col_batch = "batch" if "batch" in df.columns else ("recommended_batch" if "recommended_batch" in df.columns else None)
        col_rank = "rank_pred" if "rank_pred" in df.columns else None
        col_score = "pred_score" if "pred_score" in df.columns else None
        if not col_model or not col_batch:
            return None
        q = df[df[col_model] == m]
        if col_epochs:
            q = q[q[col_epochs] == 1]
        if col_tag:
            q = q[q[col_tag] == tag]
        if len(q) == 0:
            return None
        if col_rank and (q[col_rank].notna().any()):
            q = q.sort_values(col_rank, ascending=True)
            return int(q.iloc[0][col_batch])
        if col_score and (q[col_score].notna().any()):
            q = q.sort_values(col_score, ascending=False)
            return int(q.iloc[0][col_batch])
        return int(q.iloc[0][col_batch])
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="*.pt weights")
    ap.add_argument("--tag", default="auto", choices=["auto", "65W", "115W"],
                    help="power limit: auto = automatic detection")
    ap.add_argument("--rec115", default="scripts/outputs/recommendations_115W.csv")
    ap.add_argument("--rec65",  default="scripts/outputs/recommendations_65W.csv")
    args = ap.parse_args()
    if args.tag == "auto":
        tag, info = detect_tag_by_nvml()
        if tag is None:
            print(f"[warn] Failed to auto-detect power limit: {info}; fallback to probe.")
            detected_tag = None
        else:
            print(f"[info] Detected power tag: {tag} (power_limit≈{info}W)")
            detected_tag = tag
    else:
        detected_tag = args.tag
        print(f"[info] Using user-specified power tag: {detected_tag}")
    if detected_tag is not None:
        table_batch = _try_lookup_from_table(args.model, detected_tag, args.rec115, args.rec65)
        if table_batch is not None:
            print("\nPredictions (from table):")
            print(f"* b{table_batch:<2}")
            print(f"\nBest batch size (table): {table_batch}")
            return
    tmp_json = pathlib.Path("baseline_tmp.json")
    base = run_once(args.model, tmp_json)
    model = joblib.load("models/model.pkl")
    power_reg = model.get("power_lep") or model.get("E") or model.get("E1")
    acc_reg = model.get("map50") or model.get("A")
    rows = []
    for b in CANDIDATES:
        dp   = base["baseline_power"]  - model["baseline_power"]
        dm   = base["baseline_mem"]    - model["baseline_mem"]
        dmap = base["baseline_map50"]  - model["baseline_map50"]
        X = [[batch_codes[b], dp, dm, dmap]]
        e_pred   = float(power_reg.predict(X)[0])
        map_pred = float(acc_reg.predict(X)[0])
        rows.append((b, e_pred, map_pred))
    limit = base["baseline_map50"] * (1 - DELTA_MAP)
    valid = [r for r in rows if r[2] >= limit]
    best  = min(valid or rows, key=lambda x: x[1])
    print("\nPredictions (W / mAP50):")
    for b, pwr, mp in rows:
        flag = "*" if b == best[0] else " "
        print(f"{flag} b{b:<2}  {pwr:6.2f} W   {mp:.4f}")
    print(f"\nBest batch size (≤1% drop): {best[0]}")
    tmp_json.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
