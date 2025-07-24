import json, pathlib, tqdm, sys

RAW_DIR = pathlib.Path("outputs")
TAG_VAL = "115W"

cnt = 0
for fp in RAW_DIR.glob("yolo11n_*.json"):
    data = json.load(fp.open())
    if not data.get("tag"):
        data["tag"] = TAG_VAL
        json.dump(data, fp.open("w"), indent=2)
        cnt += 1
print(f"✔ added tag='{TAG_VAL}' to {cnt} files")
