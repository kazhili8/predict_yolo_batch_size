import argparse, pathlib, json, re, pandas as pd
from datetime import datetime, timezone, timedelta

def parse_ts_from_name(name):
    m = re.search(r"(\d{8}-\d{6})", name)
    if not m: return None
    return datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--uk_g_per_kwh", type=float, default=124.0)
    ap.add_argument("--cn_g_per_kwh", type=float, default=492.0)
    ap.add_argument("--cutover_local", default="2025-08-10 21:00:00")
    ap.add_argument("--tz_offset", default="+08:00")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    sign = 1 if args.tz_offset.startswith("+") else -1
    hh,mm = map(int, args.tz_offset[1:].split(":"))
    tz = timezone(sign*timedelta(hours=hh, minutes=mm))
    cutover = datetime.strptime(args.cutover_local, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)

    rows=[]
    for p in pathlib.Path(args.json_dir).glob("*.json"):
        ts = parse_ts_from_name(p.name)
        if ts is None: continue
        ts = ts.replace(tzinfo=tz)
        j = json.loads(p.read_text())
        wh = float(j.get("total_energy_wh", 0.0))
        kwh = wh/1000.0
        country = "UK" if ts < cutover else "CN"
        g = args.uk_g_per_kwh if country=="UK" else args.cn_g_per_kwh
        co2_g = kwh * g
        rows.append({
            "file": p.name,
            "timestamp": ts.isoformat(),
            "country": country,
            "energy_kwh": kwh,
            "co2_g": co2_g,
            "model": j.get("model",""),
            "batch": j.get("batch_size",""),
            "epochs": j.get("epochs",""),
            "tag": j.get("tag","")
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
