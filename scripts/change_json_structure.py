import json, pathlib, sys, uuid, argparse, collections
import datetime as dt
from typing import Dict, Any, Tuple
import pandas as pd

def load_old_log(path: pathlib.Path) -> Tuple[str, int, Dict[str, Any]]:
    with path.open() as fp:
        raw = json.load(fp)

    model = raw.get("model") or "unknown_model.pt"
    batch = int(raw.get("batch_size") or 1)

    def _mean(lst):
        return sum(lst) / len(lst) if lst else None

    avg_step_time = raw.get("avg_step_time") or _mean(
        [s.get("step_time", 0) for s in raw.get("steps", [])]
    )

    throughput = batch / avg_step_time if avg_step_time else None

    avg_power = raw.get("avg_power") or _mean(raw.get("power_series", []))

    avg_mem = raw.get("avg_mem") or _mean([s.get("mem", 0) for s in raw.get("steps", [])])

    avg_map = raw.get("avg_map") or None

    metrics = {
        "avg_step_time": round(avg_step_time, 5) if avg_step_time else None,
        "throughput": round(throughput, 5) if throughput else None,
        "avg_power": round(avg_power, 5) if avg_power else None,
        "avg_mem": round(avg_mem, 2) if avg_mem else None,
        "avg_map": round(avg_map or 0, 4)
    }
    return model, batch, metrics

def convert_one(fp: pathlib.Path, out_dir: pathlib.Path, algo: str):
    raw = json.load(fp.open())
    batch = int(raw.get("batch_size")
                or raw.get("batch")
                or next(iter(raw.get("metrics_per_batch", {"1": {}}))))
    epochs = int(raw.get("epochs", 1))
    avg_step = raw.get("avg_step_time")
    throughput = raw.get("throughput") or (1 / avg_step if avg_step else None)
    avg_power = raw.get("avg_power")
    avg_mem   = raw.get("avg_mem")
    map50     = raw.get("map50") or raw.get("avg_map50")
    map95     = raw.get("map50_95") or raw.get("avg_map50-95")

    metrics = {
        "avg_step_time": round(avg_step, 5) if avg_step else None,
        "throughput":    round(throughput, 5) if throughput else None,
        "avg_power":     round(avg_power, 5) if avg_power else None,
        "avg_mem":       round(avg_mem, 2) if avg_mem else None,
        "avg_map50":       round(map50 or 0, 4),
        "avg_map50-95":     round(map95 or 0, 4),
    }

    unified = {
        "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "model": raw.get("model") or "unknown_model.pt",
        "algo": "baseline",
        "algo": algo,
        "objective": "throughput",
        "reco_batch": batch,
        "ground_truth_batch": batch,
        "epochs": epochs,
        "metrics_per_batch": {str(batch): metrics},
    }

    out_path = out_dir / f"{fp.stem}_unified.json"
    json.dump(unified, out_path.open("w"), indent=2)
    print("wrote", out_path)

def group_key(model: str) -> str:
    return model

def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Convert legacy YOLO benchmark logs to the unified schema",
    )
    parser.add_argument("inputs", nargs="+", help="legacy JSON files (wildcards supported)",
    )
    parser.add_argument("-o","--out",default="logs_unified",help="output directory for unified logs",
    )
    parser.add_argument("--algo", default="baseline", help="value for the `algo` field",
    )
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pattern in args.inputs:
        for fp in pathlib.Path().glob(pattern):
            try:
                convert_one(fp, out_dir, args.algo)
            except Exception as e:
                print("skip", fp.name, "->", e)

    print("\nAll done. Unified logs saved in:", out_dir.resolve())


if __name__ == "__main__":
    main()