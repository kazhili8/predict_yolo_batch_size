import argparse, subprocess, sys, time, pandas as pd
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser("Run profiling for an explicit plan list.")
    p.add_argument("--plan", required=True, help="CSV with columns: model,epochs,tag,batch")
    p.add_argument("--json_dir", default="scripts/outputs/json_raw_v5")
    p.add_argument("--dataset", default="coco128.yaml")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--sleep", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=0, help="optional cap")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()

def run_one(py, row, dataset, imgsz, repeat, out_dir):
    cmd = [
        py, "scripts/profiling_single_time.py",
        "--model", row["model"],
        "--dataset", dataset,
        "--epochs", str(int(row["epochs"])),
        "--batch_size", str(int(row["batch"])),
        "--tag", row["tag"],
        "--repeat", str(repeat),
        "--imgsz", str(imgsz),
        "--out_dir", out_dir,
    ]
    print(">>>", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print("[WARN] failed:", e)
        return False

def main():
    args = parse_args()
    Path(args.json_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.plan)
    for col in ["model","epochs","tag","batch"]:
        if col not in df.columns: raise SystemExit(f"plan missing column: {col}")

    rows = df.to_dict(orient="records")
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"[run] tasks={len(rows)}  dry_run={args.dry_run}")

    if args.dry_run:
        for r in rows[:20]:
            print("   -", r)
        if len(rows) > 20:
            print(f"   ... and {len(rows)-20} more")
        return

    ok = fail = 0
    for i, r in enumerate(rows, 1):
        print(f"\n[{i}/{len(rows)}] (model={r['model']}, epochs={r['epochs']}, tag={r['tag']}, batch={r['batch']})")
        if run_one(args.python, r, args.dataset, args.imgsz, args.repeat, args.json_dir):
            ok += 1
        else:
            fail += 1
        time.sleep(args.sleep)

    print(f"\n[done] succeeded: {ok}, failed: {fail}")
    print("[note] Then run: feature_engineering_v2.py → aggregate_features.py → make_rank_data.py → tune_xgb_small.py → eval_oof_ranker.py")

if __name__ == "__main__":
    main()
