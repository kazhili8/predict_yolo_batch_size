import argparse, json, pathlib, datetime
from typing import Dict, Any, Tuple, List


def load_latest_log(logs_dir: pathlib.Path, model_name: str) -> Dict[str, Any]:
    target_stem = model_name.lower().replace(".pt", "")
    matched: List[Tuple[pathlib.Path, Dict[str, Any]]] = []

    for p in logs_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        stem = str(data.get("model", "")).lower().replace(".pt", "")
        if stem == target_stem:
            matched.append((p, data))

    if not matched:
        raise FileNotFoundError(
            f"No unified logs for model '{model_name}' were found in {logs_dir}"
        )

    matched.sort(key=lambda kv: kv[1].get("timestamp", ""), reverse=True)
    return matched[0][1]


def choose_batch(metrics_per_batch: Dict[str, Dict[str, Any]],
                 objective: str) -> Tuple[int, Dict[str, Any]]:

    if objective == "throughput":
        best = max(metrics_per_batch.items(),
                   key=lambda kv: kv[1].get("throughput", float("-inf")))
    elif objective == "avg_power":
        best = min(metrics_per_batch.items(),
                   key=lambda kv: kv[1].get("avg_power", float("inf")))
    elif objective == "avg_mem":
        best = min(metrics_per_batch.items(),
                   key=lambda kv: kv[1].get("avg_mem", float("inf")))
    else:
        raise ValueError(f"Unsupported objective '{objective}'")
    return int(best[0]), best[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend an optimal batch size from unified YOLO logs."
    )
    parser.add_argument("--model", required=True, help="e.g. yolov11n.pt")
    parser.add_argument(
        "--objective",
        default="throughput",
        choices=["throughput", "avg_power", "avg_mem"],
        help="Metric to optimise (default: throughput)",
    )
    parser.add_argument(
        "--logs_dir",
        default="logs_unified",
        help="Directory containing unified log JSON files (default: logs_unified)",
    )
    args = parser.parse_args()

    logs_dir = pathlib.Path(args.logs_dir)
    if not logs_dir.is_dir():
        raise NotADirectoryError(f"Logs folder '{logs_dir}' does not exist")

    log = load_latest_log(logs_dir, args.model)
    batch, metrics = choose_batch(log["metrics_per_batch"], args.objective)

    print(f"Model           : {args.model}")
    print(f"Objective       : {args.objective}")
    print(f"Recommended bsz : {batch}")
    print("Metrics         :", json.dumps(metrics, indent=2))

    runs_dir = pathlib.Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_file = runs_dir / f"{timestamp}_{pathlib.Path(args.model).stem}_{args.objective}.json"

    record = {
        "model": args.model,
        "algo": "argmax",
        "objective": args.objective,
        "reco_batch": batch,
        "ground_truth_batch": None,
        "metrics_per_batch": log["metrics_per_batch"]
    }

    out_file.write_text(json.dumps(record, indent=2))
    print(f"[autotune] Result saved to {out_file}")


if __name__ == "__main__":
    main()
