import time, json, pathlib, threading, collections, os, torch
import pynvml as nvml
from ultralytics import YOLO

def gpu_stats(handle):
    """Return (mem_MB, power_W)."""
    mem = nvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20  # MB
    pwr = nvml.nvmlDeviceGetPowerUsage(handle) / 1000        # W (mW -> W)
    return mem, pwr

def main():
    MODEL_NAME = "yolo11n.pt"
    BATCH_SIZE = 16
    IMG_SIZE = 416
    WORKERS = 2
    DATASET = "coco128.yaml"

    print("CUDA visible devices:", os.getenv("CUDA_VISIBLE_DEVICES"))
    print("PyTorch sees:", torch.cuda.device_count(), "GPUs")
    print("Current device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))

    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)

    power_log = collections.deque(maxlen=2048)  # circular buffer
    stop_evt = threading.Event()

    def power_sampler():
        while not stop_evt.is_set():
            _, pwr = gpu_stats(handle)
            power_log.append(pwr)
            time.sleep(0.05)  # 20 Hz

    thr = threading.Thread(target=power_sampler, daemon=True)
    thr.start()

    # YOLO training (batch=1)
    model = YOLO(MODEL_NAME)
    records = []  # per-step stats
    mem_peak = [0]

    def on_batch_start(trainer):
        on_batch_start.t0 = time.perf_counter()

    def on_batch_end(trainer):
        dt = time.perf_counter() - on_batch_start.t0
        mem, _ = gpu_stats(handle)
        mem_peak[0] = max(mem_peak[0], mem)
        records.append({
            "step_time": dt,
            "mem": mem,
        })

    model.add_callback("on_train_batch_start", on_batch_start)
    model.add_callback("on_train_batch_end", on_batch_end)

    model.train(
        data=DATASET,
        epochs=1,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0,
        workers=WORKERS,
        amp=True,
    )

    stop_evt.set()
    thr.join()

    if not power_log:
        power_log.append(gpu_stats(handle)[1])

    # Calculate stats
    power_avg = sum(power_log) / len(power_log)
    power_peak = max(power_log)
    avg_step = sum(d["step_time"] for d in records) / len(records)
    avg_mem = sum(d["mem"] for d in records) / len(records)
    total_time_s = sum(d["step_time"] for d in records)
    total_energy_wh = power_avg * total_time_s / 3600

    print("\n=== Batch =", BATCH_SIZE, "statistics ===")
    print(f"Average latency : {avg_step*1000:.1f} ms")
    print(f"Average memory  : {avg_mem:.0f} MB")
    print(f"Peak memory     : {mem_peak[0]:.0f} MB")
    print(f"Average power   : {power_avg:.1f} W (20 Hz)  |  Peak: {power_peak:.1f} W")
    print(f"Total energy    : {total_energy_wh:.2f} Wh")

    logdir = pathlib.Path("outputs")
    logdir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_json = logdir / f"{MODEL_NAME.replace('.pt','')}_b{BATCH_SIZE}_{ts}.json"

    payload = {
        "model": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "imgsz": IMG_SIZE,
        "dataset": DATASET,
        "steps": records,
        "power_series": list(power_log),
        "avg_step_time": avg_step,
        "avg_mem": avg_mem,
        "peak_mem": mem_peak[0],
        "avg_power": power_avg,
        "peak_power": power_peak,
        "total_energy_wh": total_energy_wh,
        "total_time_s": total_time_s,
    }

    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nDetailed records saved -> {out_json.resolve()}")


if __name__ == "__main__":
    main()
