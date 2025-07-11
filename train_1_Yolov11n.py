import time, json, pathlib
#"time" is used to measure and calculate how long each batch takes
#json is used to save records during the training process (such as power consumption and video memory) to a file

import pynvml, torch
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

avg = {
    k: sum(d[k] for d in records) / len(records)
    for k in ("step_time", "mem", "power")
}

print("\n=== Batch=1 for statistics ===")
print(f"Average delay per step: {avg['step_time']*1000:.1f} ms")
print(f"Average memory usage: {avg['mem']:.0f} MB")
print(f"Average power consumption {avg['power']:.1f} W")

out = pathlib.Path("logs_batch1.json")
out.write_text(json.dumps(records, indent=2))
print(f"\nDetailed records have been saved -> {out.resolve()}")

total_energy = sum(d["power"] * d["step_time"] / 3600 for d in records)  # Wh
print(f"Total power consumption: {total_energy:.2f} Wh")


