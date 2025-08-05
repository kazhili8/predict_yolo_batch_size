from __future__ import annotations
import argparse
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Tuple, Set
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(
        "Fill missing (model, epochs, tag, batch) by calling profiling_single_time.py only for gaps."
    )
    p.add_argument("--features", default="scripts/outputs/dataframe/features_v7.csv",
                   help="CSV produced by feature_engineering_v2.py")
    p.add_argument("--json_dir", default="scripts/outputs/json_raw_v5",
                   help="Where profiling_single_time.py writes raw JSON")
    p.add_argument("--models", nargs="+", default=None,
                   help="Models to consider. If omitted, read unique models from --features; "
                        "fallback to ['yolo11n.pt','yolo11s.pt','yolo11m.pt','yolo11x.pt'].")
    p.add_argument("--epochs", nargs="+", type=int, default=None,
                   help="Epochs to consider. If omitted, read unique epochs from --features; "
                        "fallback to [1,5,10,20].")
    p.add_argument("--batches", default="1-32",
                   help="Batch spec. Examples: '1-32' or '1,8,16,24,32'.")
    p.add_argument("--tag", required=True, choices=["115W", "65W", "both"],
                   help="Which tag to run. If 'both', it runs 115W first, then 65W.")
    p.add_argument("--dataset", default="coco128.yaml")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--repeat", type=int, default=1,
                   help="Repeat runs per combination for stability.")
    p.add_argument("--sleep", type=float, default=2.0,
                   help="Seconds to sleep between runs.")
    p.add_argument("--max_runs", type=int, default=0,
                   help="Optional cap to stop after N successful runs (0 = no cap).")
    p.add_argument("--min_logs", type=int, default=1,
                   help="Treat a combo as 'present' if it has at least this many logs in features CSV.")
    p.add_argument("--python", default=sys.executable,
                   help="Python executable to call profiling_single_time.py")
    p.add_argument("--dry_run", action="store_true",
                   help="Only print the plan without executing.")
    return p.parse_args()

def parse_batch_spec(spec: str) -> List[int]:
    """Parse '1-32' or '1,8,16,32' into a list of ints."""
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
        if lo > hi:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    parts = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            parts.append(int(tok))
    return sorted(set(parts))

def load_existing(features_csv: str) -> pd.DataFrame:
    """Load the features CSV and select the columns we need."""
    df = pd.read_csv(features_csv)
    required = ["model", "epochs", "tag", "batch"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {features_csv}")
    return df[required].copy()

def combos_from_df(df: pd.DataFrame, tag: str, min_logs: int) -> Set[Tuple[str, int, str, int]]:
    """Return a set of combos that are 'present' for a given tag (>= min_logs)."""
    df_tag = df if tag == "both" else df[df["tag"] == tag]
    g = df_tag.groupby(["model", "epochs", "tag", "batch"], as_index=False).size()
    g = g[g["size"] >= min_logs]
    return set((r["model"], int(r["epochs"]), r["tag"], int(r["batch"])) for _, r in g.iterrows())

def decide_spaces(df: pd.DataFrame, models_arg, epochs_arg, batches_arg) -> Tuple[List[str], List[int], List[int]]:
    if models_arg is not None:
        models = list(models_arg)
    else:
        if "model" in df.columns:
            models = sorted(df["model"].unique().tolist())
            if not models:
                models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11x.pt"]
        else:
            models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11x.pt"]

    if epochs_arg is not None:
        epochs = [int(e) for e in epochs_arg]
    else:
        if "epochs" in df.columns:
            epochs = sorted(int(x) for x in pd.unique(df["epochs"]))
            if not epochs:
                epochs = [1, 5, 10, 20]
        else:
            epochs = [1, 5, 10, 20]

    batches = parse_batch_spec(batches_arg)
    return models, epochs, batches

def build_plan(models: List[str], epochs: List[int], tag: str, batches: List[int]) -> List[Tuple[str,int,str,int]]:
    tags = ["115W", "65W"] if tag == "both" else [tag]
    plan = []
    for t in tags:
        for m in models:
            for e in epochs:
                for b in batches:
                    plan.append((m, e, t, b))
    return plan

def run_one(py: str, out_dir: str, dataset: str, imgsz: int, repeat: int,
            model: str, epochs: int, tag: str, batch: int) -> bool:
    cmd = [
        py, "scripts/profiling_single_time.py",
        "--model", model,
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--batch_size", str(batch),
        "--tag", tag,
        "--repeat", str(repeat),
        "--imgsz", str(imgsz),
        "--out_dir", out_dir,
    ]
    print(">>>", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] failed: {e}")
        return False
    except OSError as e:
        print(f"[ERROR] OS error: {e}")
        return False

def main():
    args = parse_args()
    Path(args.json_dir).mkdir(parents=True, exist_ok=True)

    df = load_existing(args.features)
    models, epochs, batches = decide_spaces(df, args.models, args.epochs, args.batches)
    print(f"[space] models={models}")
    print(f"[space] epochs={epochs}")
    print(f"[space] batches={batches}")
    print(f"[space] tag={args.tag}")

    existing = combos_from_df(df, args.tag, args.min_logs)
    full_plan = build_plan(models, epochs, args.tag, batches)
    todo = [c for c in full_plan if c not in existing]

    if args.tag == "both":
        ex_115 = combos_from_df(df, "115W", args.min_logs)
        ex_65  = combos_from_df(df, "65W",  args.min_logs)
        full_115 = build_plan(models, epochs, "115W", batches)
        full_65  = build_plan(models, epochs, "65W",  batches)
        print(f"[present] 115W: {len(ex_115)}/{len(full_115)}")
        print(f"[present] 65W : {len(ex_65)}/{len(full_65)}")

    print(f"[plan] total target = {len(full_plan)}")
    print(f"[plan] already have  = {len(existing)} (min_logs={args.min_logs})")
    print(f"[plan] to run        = {len(todo)}")
    if args.dry_run:
        for (m,e,t,b) in todo[:30]:
            print(f"   - missing: (model={m}, epochs={e}, tag={t}, batch={b})")
        if len(todo) > 30:
            print(f"   ... and {len(todo)-30} more")
        print("[dry-run] stop here.")
        return

    ok, fail = 0, 0
    for i, (m, e, t, b) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] (model={m}, epochs={e}, tag={t}, batch={b})")
        success = run_one(
            args.python, args.json_dir, args.dataset, args.imgsz, args.repeat,
            m, e, t, b
        )
        if success:
            ok += 1
        else:
            fail += 1
        if args.max_runs and ok >= args.max_runs:
            print(f"[stop] reached max_runs={args.max_runs}")
            break
        time.sleep(args.sleep)

    print(f"\n[done] succeeded: {ok}, failed: {fail}")
    print("[note] After collection, run feature_engineering_v2.py → aggregate_features.py → make_rank_data.py → tune_xgb_small.py → eval_oof_ranker.py")

if __name__ == "__main__":
    main()
