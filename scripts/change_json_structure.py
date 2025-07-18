import datetime as dt
import json, pathlib, sys, uuid, collections, argparse
from typing import Dict, Any, Tuple


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


#Classification
def group_key(model: str) -> str:
    return model


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Convert legacy YOLO benchmark logs to the unified schema",
    )
    parser.add_argument(
        "inputs", nargs="+", help="legacy JSON files (wildcards supported)",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="logs_unified",
        help="output directory for unified logs",
    )
    parser.add_argument(
        "--algo", default="baseline", help="value for the `algo` field",
    )
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)
    for pattern in args.inputs:
        for path in pathlib.Path().glob(pattern):
            model, batch, metrics = load_old_log(path)
            grouped[group_key(model)][str(batch)] = metrics
            print(f"parsed {path} -> model={model}, batch={batch}")

    for model, metrics_per_batch in grouped.items():
        best_batch = max(
            metrics_per_batch.items(), key=lambda kv: kv[1]["throughput"] or 0
        )[0]

        unified_log = {
            "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds"),
            "model": model,
            "algo": args.algo,
            "objective": "throughput",
            "reco_batch": int(best_batch),
            "ground_truth_batch": int(best_batch),
            "metrics_per_batch": metrics_per_batch,
        }

        out_name = f"{uuid.uuid4().hex[:8]}_{model.replace('.pt', '')}.json"
        out_path = out_dir / out_name
        json.dump(unified_log, out_path.open("w"), indent=2)
        print("wrote", out_path)

    print("\nAll done. Unified logs saved in:", out_dir.resolve())


if __name__ == "__main__":
    main()