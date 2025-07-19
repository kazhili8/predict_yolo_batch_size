import time, json, pathlib
#"time" is used to measure and calculate how long each batch takes
#json is used to save records during the training process (such as power consumption and video memory) to a file

import pynvml, torch, argparse
from ultralytics import YOLO

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

#Define the gpu_stats function to obtain the current video memory and power consumption
def gpu_stats():
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20          # MB
    pwr = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000                # W
    return mem, pwr

model = YOLO("yolo11n.pt")
records = [] #Save the time consumption, video memory, power consumption and other information of each batch
def on_batch_end(trainer):
    now = time.time()
    dt = now - on_batch_end.t0
    on_batch_end.t0 = now
    mem, pwr = gpu_stats()
    records.append({"step_time": dt, "mem": mem, "power": pwr})

on_batch_end.t0 = time.time()
model.add_callback("on_train_batch_end", on_batch_end)

model.train(
    data="coco128.yaml",
    epochs=1,
    imgsz=416,
    batch=1,
    workers=0,
    device=0
)

results_path = pathlib.Path(model.trainer.save_dir) / "results.json"
map50 = None
if results_path.exists():
    with results_path.open() as f:
        r = json.load(f)
    map50 = r["metrics"][-1]["metrics/mAP50"]

else:
    csv_path = results_path.with_suffix(".csv")
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        for col in ("metrics/mAP50", "map50"):
            if col in df.columns:
                map50 = df[col].iloc[-1]
                break
    if map50 is None:
        print("WARNING: neither results.json nor .csv found, mAP50=None")
avg = {
    k: sum(d[k] for d in records) / len(records)
    for k in ("step_time", "mem", "power")
}

print("\nBatch=1 for statistics")
print(f"Average delay per step: {avg['step_time']*1000:.1f} ms")
print(f"Average memory usage: {avg['mem']:.0f} MB")
print(f"Average power consumption {avg['power']:.1f} W")

payload = {
    "model": "yolo11n.pt",          #yolov11n / yolov11x
    "batch_size": 1,
    "dataset": "coco128.yaml",
    "steps": records,
    "power_series": [d["power"] for d in records],
    "avg_step_time": avg["step_time"],
    "avg_mem": avg["mem"],
    "avg_power": avg["power"],
    "avg_map": map50,
}

out_dir = pathlib.Path("scripts/runs")
out = out_dir / (f"yolov11n_b1_{time.strftime('%Y%m%d-%H%M%S')}.json")
out.write_text(json.dumps(payload, indent=2))
print(f"\nUnified log saved -> {out.resolve()}")

total_energy = sum(d["power"] * d["step_time"] / 3600 for d in records)  # Wh
print(f"Total power consumption: {total_energy:.2f} Wh")


