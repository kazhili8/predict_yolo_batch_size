import json
import pathlib
import sys
import argparse
import datetime as dt
from typing import Tuple, Any, Dict
import numpy as np

def _safe_mean(seq):
    """Return mean of seq or None when empty."""
    return float(np.mean(seq)) if seq else None


def _safe_std(seq):
    return float(np.std(seq)) if seq else None


def _safe_max(seq):
    return float(np.max(seq)) if seq else None


def _safe_min(seq):
    return float(np.min(seq)) if seq else None

def convert_one(fp: pathlib.Path, out_dir: pathlib.Path, algo: str) -> None:
    """Read one legacy JSON log -> write a unified log with extra power stats."""
    raw = json.load(fp.open())
    batch = int(
        raw.get("batch_size")
        or raw.get("batch")
        or next(iter(raw.get("metrics_per_batch", {"1": {}})))
    )
    epochs = int(raw.get("epochs", 1))

    steps = raw.get("steps", [])
    step_times = [s.get("step_time", 0.0) for s in steps]
    mem_usage   = [s.get("mem", 0.0) for s in steps]

    avg_step_time = _safe_mean(step_times)
    throughput    = batch / avg_step_time if avg_step_time else None
    avg_mem       = _safe_mean(mem_usage)

    power_series = raw.get("power_series", [])
    pwr_mean = _safe_mean(power_series)
    pwr_std  = _safe_std(power_series)
    pwr_max  = _safe_max(power_series)
    pwr_min  = _safe_min(power_series)
    map50 = raw.get("map50") or raw.get("avg_map50")
    map95 = raw.get("map50_95") or raw.get("avg_map50-95")

    metrics: Dict[str, Any] = {
        "avg_step_time": round(avg_step_time, 5) if avg_step_time else None,
        "throughput":    round(throughput, 5)    if throughput    else None,

        "pwr_mean": round(pwr_mean, 5) if pwr_mean else None,
        "pwr_std":  round(pwr_std, 5)  if pwr_std  else None,
        "pwr_max":  round(pwr_max, 5)  if pwr_max  else None,
        "pwr_min":  round(pwr_min, 5)  if pwr_min  else None,

        "avg_mem":      round(avg_mem, 2)   if avg_mem   else None,
        "avg_map50":    round(map50 or 0, 4) if map50 is not None else None,
        "avg_map50-95": round(map95 or 0, 4) if map95 is not None else None,
    }

    unified = {
        "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "model": raw.get("model") or "unknown_model.pt",
        "algo": algo,
        "objective": "throughput",
        "tag": raw.get("tag", ""),
        "reco_batch": batch,
        "ground_truth_batch": batch,
        "epochs": epochs,
        "metrics_per_batch": {str(batch): metrics},
    }

    out_path = out_dir / f"{fp.stem}_unified.json"
    json.dump(unified, out_path.open("w"), indent=2)
    print("wrote", out_path)

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    p = argparse.ArgumentParser(
        description="Convert legacy YOLO benchmark logs to the unified schema (+power stats)",
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="legacy JSON files (wildcards supported)",
    )
    p.add_argument(
        "-o",
        "--out",
        default="logs_unified",
        help="output directory for unified logs",
    )
    p.add_argument(
        "--algo",
        default="baseline",
        help="value to place in the `algo` field of the unified log",
    )
    args = p.parse_args(argv)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pattern in args.inputs:
        for fp in pathlib.Path().glob(pattern):
            try:
                convert_one(fp, out_dir, args.algo)
            except Exception as e:
                print("skip", fp.name, "->", e)

    print("\nAll done. Unified logs with power stats saved in:", out_dir.resolve())


if __name__ == "__main__":
    main()
