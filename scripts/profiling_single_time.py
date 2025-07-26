import argparse, time, json, pathlib, threading, collections, os, torch
import pynvml as nvml
from ultralytics import YOLO
import pandas as pd

torch.cuda.empty_cache()

DEFAULT_MODEL   = "yolo11npro.pt"
DEFAULT_DATASET = "coco128.yaml"
DEFAULT_IMG_SZ  = 416
DEFAULT_WORKERS = 2


def gpu_stats_mb_w(handle):
    """Return (vram_used_mb, power_w)."""
    mem = nvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20  # MB
    pwr = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0      # W
    return mem, pwr


def nvml_snapshot(handle):
    """Safely read a runtime snapshot from NVML. Missing fields -> None."""
    try:
        u = nvml.nvmlDeviceGetUtilizationRates(handle)  # gpu/memory %
        gpu_u = float(u.gpu)
        mem_u = float(getattr(u, "memory", None))
    except Exception:
        gpu_u, mem_u = None, None

    try:
        temp = float(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))
    except Exception:
        temp = None

    try:
        sm_clock = float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM))
    except Exception:
        sm_clock = None

    try:
        mem_clock = float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM))
    except Exception:
        mem_clock = None

    try:
        pwr = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except Exception:
        pwr = None

    return dict(gpu_util=gpu_u, mem_util=mem_u, temperature=temp,
                sm_clock=sm_clock, mem_clock=mem_clock, power=pwr)


def get_static_info(handle):
    """Return static device/runtime info for logging."""
    try:
        name = nvml.nvmlDeviceGetName(handle)
        gpu_name = name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        gpu_name = "unknown"

    try:
        vram_total_mb = nvml.nvmlDeviceGetMemoryInfo(handle).total / 2**20
    except Exception:
        vram_total_mb = None

    try:
        pwr_limit_w = nvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
    except Exception:
        pwr_limit_w = None

    try:
        _drv = nvml.nvmlSystemGetDriverVersion()
        driver_version = _drv.decode() if isinstance(_drv, bytes) else str(_drv)
    except Exception:
        driver_version = None

    cuda_version = getattr(torch.version, "cuda", None)
    pytorch_version = torch.__version__

    return dict(gpu_name=gpu_name,
                vram_total_mb=vram_total_mb,
                power_limit_w=pwr_limit_w,
                driver_version=driver_version,
                cuda_version=cuda_version,
                pytorch_version=pytorch_version)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      default=DEFAULT_MODEL)
    ap.add_argument("--dataset",    default=DEFAULT_DATASET)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--repeat",     type=int, default=1, help="how many repeated runs")
    ap.add_argument("--epochs",     type=int, default=1, help="train epochs per run")
    ap.add_argument("--imgsz",      type=int, default=DEFAULT_IMG_SZ)
    ap.add_argument("--workers",    type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--tag",        type=str, default="", help="custom tag, e.g. 115W/65W")
    ap.add_argument("--out_dir",    type=str, default="scripts/outputs/json_raw_v5")
    args = ap.parse_args()

    MODEL_NAME  = args.model
    BATCH_SIZE  = args.batch_size
    IMG_SIZE    = args.imgsz
    WORKERS     = args.workers
    DATASET     = args.dataset
    REPEAT      = args.repeat

    print("CUDA visible devices:", os.getenv("CUDA_VISIBLE_DEVICES"))
    print("PyTorch sees:", torch.cuda.device_count(), "GPUs")
    print("Current device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))

    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    static_info = get_static_info(handle)

    def run_one(model_name: str, batch: int, img_sz: int, dataset: str,
                workers: int, handle, power_log, records, mem_peak,
                util_log, mem_util_log, temp_log, sm_clock_log, mem_clock_log):

        power_log.clear(); util_log.clear(); mem_util_log.clear()
        temp_log.clear(); sm_clock_log.clear(); mem_clock_log.clear()
        records.clear(); mem_peak[0] = 0

        model = YOLO(model_name)
        stop_evt = threading.Event()

        def sampler():
            while not stop_evt.is_set():
                snap = nvml_snapshot(handle)
                if snap["power"] is not None:
                    power_log.append(snap["power"])
                if snap["gpu_util"] is not None:
                    util_log.append(snap["gpu_util"])
                if snap["mem_util"] is not None:
                    mem_util_log.append(snap["mem_util"])
                if snap["temperature"] is not None:
                    temp_log.append(snap["temperature"])
                if snap["sm_clock"] is not None:
                    sm_clock_log.append(snap["sm_clock"])
                if snap["mem_clock"] is not None:
                    mem_clock_log.append(snap["mem_clock"])
                time.sleep(0.05)

        thr = threading.Thread(target=sampler, daemon=True)
        thr.start()
        prev_t = [time.perf_counter()]

        def on_batch_start(trainer):
            prev_t[0] = time.perf_counter()

        def on_batch_end(trainer):
            t_now = time.perf_counter()
            step_time = t_now - prev_t[0]
            mem_mb, _ = gpu_stats_mb_w(handle)
            thr_step = batch / step_time if step_time > 0 else 0.0
            records.append({"step_time": step_time, "mem": mem_mb, "thr": thr_step})
            mem_peak[0] = max(mem_peak[0], mem_mb)

        model.add_callback("on_train_batch_start", on_batch_start)
        model.add_callback("on_train_batch_end", on_batch_end)

        model.train(
            data=dataset, epochs=args.epochs, imgsz=img_sz,
            batch=batch, workers=workers, device=0, verbose=False
        )

        stop_evt.set()
        thr.join()
        return model

    power_log = collections.deque(maxlen=100000)
    util_log = collections.deque(maxlen=100000)
    mem_util_log = collections.deque(maxlen=100000)
    temp_log = collections.deque(maxlen=100000)
    sm_clock_log = collections.deque(maxlen=100000)
    mem_clock_log = collections.deque(maxlen=100000)

    records = []
    mem_peak = [0]

    for r in range(REPEAT):
        print(f"\n=== Run {r + 1}/{REPEAT} for batch={BATCH_SIZE} ===")
        model = run_one(MODEL_NAME, BATCH_SIZE, IMG_SIZE, DATASET, WORKERS, handle,
                        power_log, records, mem_peak,
                        util_log, mem_util_log, temp_log, sm_clock_log, mem_clock_log)

        def get_maps(model) -> tuple[float | None, float | None]:
            m = getattr(model.trainer, "metrics", None)
            if isinstance(m, dict):
                box = m.get("box", {})
                if "map50" in box and "map" in box:
                    return float(box["map50"]), float(box["map"])
            try:
                return float(model.trainer.metrics.box.map50), float(model.trainer.metrics.box.map)
            except Exception:
                pass
            csv_path = pathlib.Path(model.trainer.save_dir) / "results.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                last = df.iloc[-1]
                return float(last["metrics/mAP50(B)"]), float(last["metrics/mAP50-95(B)"])
            return None, None

        map50, map5095 = get_maps(model)
        print(f"Final mAP@0.5: {map50:.4f}" if map50 else "mAP@0.5 not found")
        print(f"Final mAP@0.5:0.95: {map5095:.4f}" if map5095 else "mAP@0.5:0.95 not found")

        if not power_log:
            power_log.append(gpu_stats_mb_w(handle)[1])

        power_avg = sum(power_log) / len(power_log)
        power_peak = max(power_log)
        avg_step = sum(d["step_time"] for d in records) / max(len(records), 1)
        avg_mem = sum(d["mem"] for d in records) / max(len(records), 1)
        total_time_s = sum(d["step_time"] for d in records)
        total_energy_wh = power_avg * total_time_s / 3600.0

        print("\n=== Batch =", BATCH_SIZE, "statistics ===")
        print(f"Average latency : {avg_step*1000:.1f} ms")
        print(f"Average memory  : {avg_mem:.0f} MB")
        print(f"Peak memory     : {mem_peak[0]:.0f} MB")
        print(f"Average power   : {power_avg:.1f} W (20 Hz)  |  Peak: {power_peak:.1f} W")
        print(f"Final mAP@0.5: {map50:.4f}" if map50 else "mAP@0.5 not found")
        print(f"Final mAP@0.5:0.95: {map5095:.4f}" if map5095 else "mAP@0.5:0.95 not found")
        print(f"Total energy    : {total_energy_wh:.2f} Wh")

        out_dir = pathlib.Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_json = out_dir / f"{MODEL_NAME.replace('.pt', '')}_b{BATCH_SIZE}_e{args.epochs}_run{r + 1}_{ts}.json"

        payload = {
            "model": MODEL_NAME, "batch_size": BATCH_SIZE, "imgsz": IMG_SIZE,
            "dataset": DATASET, "epochs": args.epochs, "tag": args.tag,
            **static_info,  # gpu_name, vram_total_mb, power_limit_w, versions
            "steps": records,  # [{step_time, mem, thr}, ...]
            "power_series": list(power_log),
            "gpu_util_series": list(util_log),
            "mem_util_series": list(mem_util_log),
            "temp_series": list(temp_log),
            "sm_clock_series": list(sm_clock_log),
            "mem_clock_series": list(mem_clock_log),
            "avg_step_time": avg_step,
            "avg_mem": avg_mem,
            "peak_mem": mem_peak[0],
            "avg_power": power_avg,
            "peak_power": power_peak,
            "total_energy_wh": total_energy_wh,
            "total_time_s": total_time_s,
            "map50": map50,
            "map50_95": map5095,
        }

        out_json.write_text(json.dumps(payload, indent=2))
        print(f"\nDetailed records saved -> {out_json.resolve()}")

    nvml.nvmlShutdown()


if __name__ == "__main__":
    main()
