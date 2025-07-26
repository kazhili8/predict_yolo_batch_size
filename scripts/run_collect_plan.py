import argparse, itertools, pathlib, subprocess, sys, time

PY = sys.executable
DEFAULT_OUT_DIR = "scripts/outputs/json_raw_v5"
DEFAULT_MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
DEFAULT_EPOCHS = [1, 10]
DEFAULT_BATCHES = [1, 8, 16, 32]
DEFAULT_REPEAT = 2
DEFAULT_IMGZ = 416
DEFAULT_DATASET = "coco128.yaml"

def _print_power_info():
    try:
        import pynvml as nvml
        nvml.nvmlInit()
        h = nvml.nvmlDeviceGetHandleByIndex(0)
        mn, mx = nvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
        enforced = nvml.nvmlDeviceGetEnforcedPowerLimit(h)
        print(f"[power] allowed range ≈ {mn/1000:.0f}–{mx/1000:.0f} W | enforced ≈ {enforced/1000:.0f} W")
        nvml.nvmlShutdown()
    except Exception:
        pass

def run_one(args, model, epochs, batch):
    cmd = [
        PY, "scripts/profiling_single_time.py",
        "--model", model,
        "--dataset", args.dataset,
        "--epochs", str(epochs),
        "--batch_size", str(batch),
        "--tag", args.tag,
        "--repeat", str(args.repeat),
        "--imgsz", str(args.imgsz),
        "--out_dir", args.out_dir,
    ]
    print("\n>>>", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print("[WARN] failed:", e)
        return False

def main():
    ap = argparse.ArgumentParser("Run data collection for a single power tag")
    ap.add_argument("--tag", required=True, choices=["115W", "65W"],
                    help="Run data collection for the specified power tag; run the other tag separately later")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--epochs", nargs="+", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES)
    ap.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGZ)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="Pause duration in seconds between each run")
    args = ap.parse_args()

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(args.models, args.epochs, args.batches))
    print(f"[plan] tag={args.tag}")
    print(f"[plan] models={args.models}")
    print(f"[plan] epochs={args.epochs}")
    print(f"[plan] batches={args.batches}  repeat={args.repeat}")
    print(f"[plan] total runs = {len(combos) * args.repeat} (each repeated {args.repeat}x)")
    print("[hint] Please set the power limit as administrator, e.g.:")
    if args.tag == "115W":
        print("       nvidia-smi -pm 1 && nvidia-smi -pl 115")
    else:
        print("       nvidia-smi -pm 1 && nvidia-smi -pl 65")
    _print_power_info()

    ok = fail = 0
    for (m, e, b) in combos:
        success = run_one(args, m, e, b)
        ok += int(success)
        fail += int(not success)
        time.sleep(args.sleep)

    print(f"\n[done] succeeded: {ok}, failed: {fail}")
    print("[note] After collection, you can run feature_engineering_v2.py to generate the CSV directly.")

if __name__ == "__main__":
    main()
