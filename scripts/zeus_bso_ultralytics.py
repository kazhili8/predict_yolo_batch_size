import argparse, os, time, pandas as pd
from pathlib import Path
from ultralytics import YOLO
from zeus.monitor import ZeusMonitor
from zeus.optimizer.batch_size import BatchSizeOptimizer, JobSpec

def parse_batches(spec):
    spec = spec.strip()
    if "-" in spec:
        a,b = spec.split("-",1)
        lo,hi = int(a),int(b)
        if lo>hi: lo,hi = hi,lo
        return list(range(lo,hi+1))
    return sorted(set(int(x.strip()) for x in spec.split(",") if x.strip()))

def last_map50(save_dir):
    p = Path(save_dir) / "results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if len(df)==0:
        return None
    row = df.iloc[-1]
    for c in ["metrics/mAP50(B)","mAP50","map50","metrics/mAP50"]:
        if c in df.columns:
            return float(row[c])
    return None

def run_one(model_pt, data, imgsz, epochs, server_url, job_prefix, eta, batches, default_bs):
    monitor = ZeusMonitor()
    bso = BatchSizeOptimizer(
        monitor=monitor,
        server_url=server_url,
        job=JobSpec(
            job_id=os.environ.get("ZEUS_JOB_ID"),
            job_id_prefix=job_prefix,
            default_batch_size=default_bs,
            batch_sizes=batches,
            max_epochs=epochs,
            eta=eta
        ),
    )
    bs = bso.get_batch_size()
    model = YOLO(model_pt)
    bso.on_train_begin()
    save_dir = None
    for e in range(epochs):
        model.train(data=data, epochs=1, imgsz=imgsz, batch=bs, workers=2, device=0, verbose=False, resume=(e>0))
        save_dir = model.trainer.save_dir
        metric = last_map50(save_dir) or 0.0
        bso.on_evaluate(metric)
        if getattr(bso, "training_finished", False):
            break
    return int(bs), str(save_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_url", default="http://127.0.0.1:8000")
    ap.add_argument("--job_prefix", default="yolo")
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--dataset", default="coco128.yaml")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batches", default="1-32")
    ap.add_argument("--default_bs", type=int, default=8)
    ap.add_argument("--tag", choices=["65W","115W"], required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    batches = parse_batches(args.batches)
    rows = []
    for m in args.models:
        bs, save_dir = run_one(m, args.dataset, args.imgsz, args.epochs, args.server_url, args.job_prefix, args.eta, batches, args.default_bs)
        rows.append({"model":m,"epochs":int(args.epochs),"tag":args.tag,"recommended_batch":bs,"save_dir":save_dir,"eta":args.eta})
        time.sleep(1.0)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
