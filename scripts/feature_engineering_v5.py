import argparse, json, pathlib, statistics
from typing import List, Sequence, Dict, Any, Tuple
import numpy as np
import pandas as pd


def _q(x: Sequence[float], q: float) -> float | None:
    x = [v for v in x if v is not None]
    if not x:
        return None
    return float(np.quantile(x, q))


def _mean(x: Sequence[float]) -> float | None:
    x = [v for v in x if v is not None]
    if not x:
        return None
    return float(np.mean(x))


def _std(x: Sequence[float]) -> float | None:
    x = [v for v in x if v is not None]
    if not x:
        return None
    return float(np.std(x, ddof=0))


def _slope(y: Sequence[float]) -> float | None:
    """Return linear slope of y over index [0..n-1]."""
    y = [v for v in y if v is not None]
    n = len(y)
    if n < 3:
        return None
    x = np.arange(n, dtype=float)
    k, _ = np.polyfit(x, np.asarray(y, dtype=float), 1)
    return float(k)


def _early_late_diff(y: Sequence[float], frac: float = 0.2) -> float | None:
    y = [v for v in y if v is not None]
    n = len(y)
    if n < 5:
        return None
    k = max(1, int(n * frac))
    early = float(np.mean(y[:k]))
    late  = float(np.mean(y[-k:]))
    return late - early


def collect_features(json_dir: pathlib.Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in json_dir.glob("*.json"):
        obj = json.loads(p.read_text(encoding="utf-8"))

        batch = obj.get("batch") or obj.get("batch_size")
        epochs = obj.get("epochs")
        map50 = obj.get("avg_map50") or obj.get("map50")

        steps = obj.get("steps", [])
        step_time = [d.get("step_time") for d in steps]
        step_mem  = [d.get("mem") for d in steps]
        step_thr  = [d.get("thr") if d.get("thr") is not None and d.get("thr") != 0 else
                     (batch / d["step_time"] if d.get("step_time") else None) for d in steps]

        power_series = obj.get("power_series", [])
        gpu_util_series = obj.get("gpu_util_series", [])
        mem_util_series = obj.get("mem_util_series", [])
        temp_series = obj.get("temp_series", [])
        sm_clock_series = obj.get("sm_clock_series", [])
        mem_clock_series = obj.get("mem_clock_series", [])

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

        if power_series:
            row["pwr_mean"] = statistics.mean(power_series)
            row["pwr_std"]  = statistics.pstdev(power_series)
            row["pwr_max"]  = max(power_series)
            row["pwr_min"]  = min(power_series)
            row["pwr_p95"]  = _q(power_series, 0.95)
            row["power_peak_to_mean"] = (row["pwr_max"] / row["pwr_mean"]) if row["pwr_mean"] else None
            row["power_range"] = row["pwr_max"] - row["pwr_min"]
            row["power_slope"] = _slope(power_series)
            row["power_early_late"] = _early_late_diff(power_series)
        else:
            for k in ["pwr_mean","pwr_std","pwr_max","pwr_min","pwr_p95",
                      "power_peak_to_mean","power_range","power_slope","power_early_late"]:
                row[k] = None

        if step_time:
            row["step_time_mean"] = _mean(step_time)
            row["step_time_std"]  = _std(step_time)
            row["step_time_p95"]  = _q(step_time, 0.95)
            row["step_time_cv"]   = (row["step_time_std"] / row["step_time_mean"]) if row["step_time_mean"] else None
            row["step_time_slope"] = _slope(step_time)
        else:
            for k in ["step_time_mean","step_time_std","step_time_p95","step_time_cv","step_time_slope"]:
                row[k] = None

        if step_thr:
            row["thr_mean"] = _mean(step_thr)
            row["thr_std"]  = _std(step_thr)
            row["thr_p95"]  = _q(step_thr, 0.95)
            row["throughput_var_ratio"] = (row["thr_std"] ** 2 / (row["thr_mean"] ** 2)
                                           if row["thr_mean"] else None)
        else:
            for k in ["thr_mean","thr_std","thr_p95","throughput_var_ratio"]:
                row[k] = None

        def add_series(prefix: str, series: Sequence[float]):
            row[f"{prefix}_mean"] = _mean(series)
            row[f"{prefix}_std"]  = _std(series)
            row[f"{prefix}_max"]  = (max(series) if series else None)
            row[f"{prefix}_p95"]  = _q(series, 0.95)
            row[f"{prefix}_slope"] = _slope(series)
            row[f"{prefix}_early_late"] = _early_late_diff(series)

        add_series("gpu_util", gpu_util_series)
        add_series("mem_util", mem_util_series)
        add_series("temp", temp_series)
        add_series("sm_clock", sm_clock_series)
        add_series("mem_clock", mem_clock_series)

        if row["avg_step_time"]:
            row["throughput"] = row["batch"] / row["avg_step_time"]
        else:
            row["throughput"] = None

        if row["throughput"] and row["avg_power"]:
            row["energy_per_img"] = row["avg_power"] / row["throughput"]
        else:
            row["energy_per_img"] = None

        row["power_limit_w"] = obj.get("power_limit_w")
        row["vram_total_mb"] = obj.get("vram_total_mb")
        row["is65W"] = int("65" in str(row["tag"])) if row.get("tag") is not None else 0

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
