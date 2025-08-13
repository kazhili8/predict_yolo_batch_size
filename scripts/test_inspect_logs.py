import json, pathlib, statistics as st, pandas as pd

logdir = pathlib.Path("outputs")
files  = sorted(logdir.glob("batch1_*.json"))

rows = []
for fp in files:
    records = json.loads(fp.read_text())
    rows.append({
        "file": fp.name,
        "steps": len(records),
        "lat_ms": st.mean(r["step_time"] for r in records) * 1e3,
        "mem_MB": st.mean(r["mem"] for r in records),
        "pwr_W":  st.mean(r["power"] for r in records),
    })

df = pd.DataFrame(rows)
print(df.round(1))
