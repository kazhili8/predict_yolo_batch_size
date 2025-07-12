import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas

CSV_DIR = Path(r"D:\Predict_YOLO_batch_size\scripts\outputs\dataframe")
csv_files = list(CSV_DIR.glob("*.csv"))
print("Found CSV files:", [f.name for f in csv_files])

if not csv_files:
    raise RuntimeError(
        f"No CSV files found in {CSV_DIR}. "
        "Check the path or make sure CSV files exist."
    )

metrics: list[dict] = []

for csv_file in csv_files:
    match = re.search(r"batch_size=(\d+)", csv_file.stem)
    if not match:
        print(f"Skipping invalid file name: {csv_file.name}")
        continue

    batch_size = int(match.group(1))
    model_match = re.search(r"(yolo11[^\_]+)", csv_file.stem)
    model = model_match.group(1) if model_match else "unknown"

    df = pandas.read_csv(csv_file)
    avg_step_time = df["step_time"].mean()
    throughput = batch_size / avg_step_time if avg_step_time > 0 else 0
    avg_memory = df["mem"].max()
    avg_power = df["power"].mean()

    metrics.append(
        {
            "model": model,
            "batch_size": batch_size,
            "throughput": throughput,
            "avg_mem": avg_memory,
            "avg_power": avg_power,
        }
    )

df_all = (
    pandas.DataFrame(metrics)
    .groupby(["model", "batch_size"], as_index=False)
    .mean()
    .sort_values(["model", "batch_size"])
)

color_map = {
    "yolo11x": "tab:blue",
    "yolo11n": "tab:orange",
}

#draw the curve for theplot throughput
plt.figure(figsize=(10, 6))
for mdl, grp in df_all.groupby("model"):
    plt.plot(
        grp["batch_size"],
        grp["throughput"],
        marker="o",
        label=mdl,
        color=color_map.get(mdl),
    )
plt.title("Throughput vs Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Throughput (images/sec)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(CSV_DIR / "plot_throughput.png")
plt.show()

#draw the curve for the plot average power
plt.figure(figsize=(10, 6))
for mdl, grp in df_all.groupby("model"):
    plt.plot(
        grp["batch_size"],
        grp["avg_power"],
        marker="o",
        label=mdl,
        color=color_map.get(mdl),
    )
plt.title("Average Power vs Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Power (W)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(CSV_DIR / "plot_power.png")
plt.show()

#draw the curve for the plot average memory
plt.figure(figsize=(10, 6))
for mdl, grp in df_all.groupby("model"):
    plt.plot(
        grp["batch_size"],
        grp["avg_mem"],
        marker="o",
        label=mdl,
        color=color_map.get(mdl),
    )
plt.title("Average Memory vs Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Memory (MB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(CSV_DIR / "plot_memory.png")
plt.show()

print("Plot images saved to:", CSV_DIR)
