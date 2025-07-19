import json, glob, joblib, pathlib, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

LOG_DIR = pathlib.Path("logs_unified")
OUT_DIR = pathlib.Path("models"); OUT_DIR.mkdir(exist_ok=True)

rows = []
for fp in LOG_DIR.glob("*.json"):
    data = json.loads(fp.read_text())
    m1 = data["metrics_per_batch"]["1"]           # baseline batch=1
    for b, m in data["metrics_per_batch"].items():
        if b == "1": continue
        rows.append({
            "batch": int(b),# features = baseline stats + candidate batch
            "baseline_power": m1["avg_power"],
            "baseline_mem"  : m1["avg_mem"],
            "baseline_map"  : m1["avg_map"],

            "energy": m["avg_power"],# target
            "map"   : m["avg_map"],
        })

df = pd.DataFrame(rows)
X = df[["batch", "baseline_power", "baseline_mem", "baseline_map"]]
y_e = df["energy"];  y_a = df["map"]

re_e = ExtraTreesRegressor(200, random_state=0).fit(X, y_e)
re_a = ExtraTreesRegressor(200, random_state=0).fit(X, y_a)

joblib.dump({"E": re_e, "A": re_a}, OUT_DIR / "model.pkl")
print("Saved -> models/model.pkl")