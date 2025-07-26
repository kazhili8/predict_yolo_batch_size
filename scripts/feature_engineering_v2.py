import argparse,json,pathlib,statistics
from typing import List, Dict, Any
import numpy as np
import pandas as pd

def _safe_mean(xs: List[float] | None) -> float | None:
    xs = [x for x in (xs or []) if x is not None]
    return float(statistics.fmean(xs)) if xs else None

def _safe_std(xs: List[float] | None) -> float | None:
    xs = [x for x in (xs or []) if x is not None]
    return float(np.std(xs)) if xs else None

def _safe_max(xs: List[float] | None) -> float | None:
    xs = [x for x in (xs or []) if x is not None]
    return float(max(xs)) if xs else None

def _safe_min(xs: List[float] | None) -> float | None:
    xs = [x for x in (xs or []) if x is not None]
    return float(min(xs)) if xs else None

def _safe_p95(xs: List[float] | None) -> float | None:
    xs = [x for x in (xs or []) if x is not None]
    return float(np.percentile(xs, 95)) if xs else None

def _slope(xs: List[float] | None) -> float | None:
    """Return least-squares slope vs index; None if not enough points."""
    xs = [x for x in (xs or []) if x is not None]
    n = len(xs)
    if n < 3:
        return None
    x = np.arange(n, dtype=float)
    y = np.array(xs, dtype=float)
    try:
        k = np.polyfit(x, y, 1)[0]
        return float(k)
    except Exception:
        return None

def _early_late_delta(xs: List[float] | None) -> float | None:
    """Mean(last 25%) - Mean(first 25%)."""
    xs = [x for x in (xs or []) if x is not None]
    n = len(xs)
    if n < 4:
        return None
    k = max(1, n // 4)
    first = float(np.mean(xs[:k]))
    last = float(np.mean(xs[-k:]))
    return last - first

def _collect_series(obj: Dict[str, Any], key: str) -> List[float]:
    """Return a float list for a series key; fallback to empty list."""
    xs = obj.get(key) or []
    return [float(x) for x in xs if x is not None]

def _collect_steps(obj: Dict[str, Any]) -> Dict[str, List[float]]:
    """Extract per-step lists (step_time, mem, thr) if present."""
    steps = obj.get("steps") or []
    st = [float(s.get("step_time", 0.0)) for s in steps if s is not None]
    mem = [float(s.get("mem", 0.0)) for s in steps if s is not None]
    thr = [float(s.get("thr", 0.0)) for s in steps if s is not None]
    return {"step_time": st, "mem": mem, "thr": thr}

def collect_features(json_dir: pathlib.Path) -> pd.DataFrame:
    rows = []

    for p in sorted(json_dir.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        model = obj.get("model") or obj.get("model_name")
        batch = obj.get("batch") or obj.get("batch_size") or obj.get("reco_batch")
        epochs = obj.get("epochs") or obj.get("num_epochs") or obj.get("epoch")
        tag = obj.get("tag", "")

        step_dict = _collect_steps(obj)
        step_time = step_dict["step_time"]
        step_mem = step_dict["mem"]
        step_thr = step_dict["thr"]

        step_time_mean = _safe_mean(step_time)
        step_time_std = _safe_std(step_time)
        step_time_p95 = _safe_p95(step_time)
        step_time_cv = (step_time_std / step_time_mean) if (step_time_std and step_time_mean and step_time_mean > 0) else None
        step_time_slope = _slope(step_time)

        thr_mean = _safe_mean(step_thr)
        thr_std = _safe_std(step_thr)
        thr_p95 = _safe_p95(step_thr)
        thr_var = (thr_std ** 2) if thr_std is not None else None
        thr_vr = (thr_var / (thr_mean ** 2 + 1e-9)) if (thr_var is not None and thr_mean) else None

        avg_step_time = step_time_mean
        throughput = (float(batch) / avg_step_time) if (batch and avg_step_time and avg_step_time > 0) else None
        avg_mem = _safe_mean(step_mem)

        power_series = _collect_series(obj, "power_series")
        pwr_mean = _safe_mean(power_series)
        pwr_std = _safe_std(power_series)
        pwr_max = _safe_max(power_series)
        pwr_min = _safe_min(power_series)
        pwr_p95 = _safe_p95(power_series)
        power_peak_to_mean = (pwr_max / pwr_mean) if (pwr_max and pwr_mean and pwr_mean > 0) else None
        power_range = (pwr_max - pwr_min) if (pwr_max is not None and pwr_min is not None) else None
        power_slope = _slope(power_series)
        power_early_late = _early_late_delta(power_series)

        avg_power = obj.get("avg_power") or pwr_mean

        gpu_util = _collect_series(obj, "gpu_util_series")
        mem_util = _collect_series(obj, "mem_util_series")
        temp_ser = _collect_series(obj, "temp_series")
        sm_clock = _collect_series(obj, "sm_clock_series")
        mem_clock = _collect_series(obj, "mem_clock_series")

        def stats_block(xs: List[float], prefix: str) -> Dict[str, Any]:
            return {
                f"{prefix}_mean": _safe_mean(xs),
                f"{prefix}_std": _safe_std(xs),
                f"{prefix}_max": _safe_max(xs),
                f"{prefix}_p95": _safe_p95(xs),
                f"{prefix}_slope": _slope(xs),
                f"{prefix}_early_late": _early_late_delta(xs),
            }

        gpu_stats = stats_block(gpu_util, "gpu_util")
        memu_stats = stats_block(mem_util, "mem_util")
        temp_stats = stats_block(temp_ser, "temp")
        smc_stats = stats_block(sm_clock, "sm_clock")
        memc_stats = stats_block(mem_clock, "mem_clock")

        map50 = obj.get("map50") or obj.get("avg_map50") or obj.get("mAP50")
        power_limit_w = obj.get("power_limit_w")
        vram_total_mb = obj.get("vram_total_mb")

        energy_per_img = (avg_power / throughput) if (avg_power and throughput and throughput > 0) else None
        is65W = 1 if str(tag).strip().upper().startswith("65W") or "65W" in str(tag).upper() else 0

        row = {
            "json_file": p.name,
            "model": model,
            "batch": int(batch) if batch is not None else None,
            "epochs": int(epochs) if epochs is not None else None,
            "tag": tag,
            "avg_step_time": avg_step_time,
            "avg_power": avg_power,
            "avg_mem": avg_mem,
            "map50": float(map50) if map50 is not None else None,
            "pwr_mean": pwr_mean, "pwr_std": pwr_std, "pwr_max": pwr_max, "pwr_min": pwr_min, "pwr_p95": pwr_p95,
            "power_peak_to_mean": power_peak_to_mean, "power_range": power_range,
            "power_slope": power_slope, "power_early_late": power_early_late,
            "step_time_mean": step_time_mean, "step_time_std": step_time_std, "step_time_p95": step_time_p95,
            "step_time_cv": step_time_cv, "step_time_slope": step_time_slope,
            "thr_mean": thr_mean, "thr_std": thr_std, "thr_p95": thr_p95,
            "throughput_var_ratio": thr_vr,

            **gpu_stats, **memu_stats, **temp_stats, **smc_stats, **memc_stats,

            "throughput": throughput,
            "energy_per_img": energy_per_img,
            "power_limit_w": power_limit_w,
            "vram_total_mb": vram_total_mb,
            "is65W": is65W,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="Directory of raw or unified JSON logs")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    df = collect_features(pathlib.Path(args.json_dir))
    df.to_csv(args.out, index=False)
    print(f"saved → {args.out},  rows={len(df)}, cols={len(df.columns)}")


if __name__ == "__main__":
    main()
