import argparse, json, re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Set
import pandas as pd

FN_RE = re.compile(r"_b(?P<b>\d+)_e(?P<e>\d+)_", re.IGNORECASE)

def parse_batch_spec(spec: str) -> List[int]:
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
        if lo > hi: lo, hi = hi, lo
        return list(range(lo, hi + 1))
    return sorted({int(x) for x in spec.split(",") if x.strip()})

def infer_tag_from_power_limit(power_limit_w: Optional[float]) -> Optional[str]:
    if power_limit_w is None:
        return None
    if 90 <= power_limit_w <= 140:
        return "115W"
    if 50 <= power_limit_w <= 80:
        return "65W"
    return None

def safe_get(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def extract_fields(p: Path) -> Optional[Tuple[str, int, str, int]]:
    """Return (model, epochs, tag, batch) or None if insufficient."""
    try:
        with p.open("r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        return None

    model = safe_get(j, "model", "model_name")
    if not model:
        stem = p.stem
        model = stem.split("_b")[0] + ".pt" if "_b" in stem else None

    epochs = safe_get(j, "epochs", "train_epochs")
    if epochs is None:
        m = FN_RE.search(p.name)
        if m:
            try:
                epochs = int(m.group("e"))
            except Exception:
                pass
    try:
        epochs = int(epochs) if epochs is not None else None
    except Exception:
        epochs = None

    batch = safe_get(j, "batch", "batch_size")
    if batch is None:
        m = FN_RE.search(p.name)
        if m:
            try:
                batch = int(m.group("b"))
            except Exception:
                pass
    try:
        batch = int(batch) if batch is not None else None
    except Exception:
        batch = None

    tag = safe_get(j, "tag", "power_tag")
    if not tag:
        pl = safe_get(j, "power_limit_w", default=None)
        try:
            pl = float(pl) if pl is not None else None
        except Exception:
            pl = None
        tag = infer_tag_from_power_limit(pl) or "UNKNOWN"

    if not model or epochs is None or batch is None or not tag:
        return None
    return str(model), int(epochs), str(tag), int(batch)

def parse_args():
    p = argparse.ArgumentParser("List missing combos by scanning raw JSON folder.")
    p.add_argument("--json_dir", required=True)
    p.add_argument("--models", nargs="+", default=["yolo11n.pt","yolo11s.pt","yolo11m.pt","yolo11x.pt"])
    p.add_argument("--epochs", nargs="+", type=int, required=True)
    p.add_argument("--batches", default="1-32")
    p.add_argument("--tags", nargs="+", default=["115W","65W"])
    p.add_argument("--min_logs", type=int, default=1, help="present if >= min_logs per combo")
    p.add_argument("--out_prefix", required=True, help="output prefix (no extension)")
    return p.parse_args()

def main():
    a = parse_args()
    json_dir = Path(a.json_dir)
    files = list(json_dir.rglob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files under: {json_dir}")

    rows = []
    skipped = 0
    for f in files:
        r = extract_fields(f)
        if r is None:
            skipped += 1
            continue
        rows.append(r)
    df = pd.DataFrame(rows, columns=["model","epochs","tag","batch"])
    if df.empty:
        raise SystemExit("No usable records parsed from JSON.")

    batches = parse_batch_spec(a.batches)
    models = list(a.models)
    epochs = [int(e) for e in a.epochs]
    tags   = list(a.tags)

    target = pd.MultiIndex.from_product(
        [models, epochs, tags, batches],
        names=["model","epochs","tag","batch"]
    ).to_frame(index=False)

    g = (
        df.groupby(["model","epochs","tag","batch"], as_index=False)
          .size()
          .rename(columns={"size":"n_logs"})
    )

    merged = target.merge(g, on=["model","epochs","tag","batch"], how="left")
    merged["n_logs"] = merged["n_logs"].fillna(0).astype(int)

    present = merged[merged["n_logs"] >= a.min_logs].copy()
    missing = merged[merged["n_logs"] < a.min_logs].copy()

    dups = merged[merged["n_logs"] >= 2].copy()

    out_dir = Path(a.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    present.to_csv(f"{a.out_prefix}_present_counts.csv", index=False)
    missing.to_csv(f"{a.out_prefix}_missing_plan.csv", index=False)
    dups.to_csv(f"{a.out_prefix}_dups_ge2.csv", index=False)

    print(f"[scan] json files = {len(files)}, parsed = {len(df)}, skipped = {skipped}")
    print(f"[grid] models={len(models)}, epochs={len(epochs)}, tags={len(tags)}, batches={len(batches)}")
    print(f"[grid] target combos = {len(target)}")
    print(f"[stat] present (n_logs>={a.min_logs}) = {len(present)}")
    print(f"[stat] missing (n_logs<{a.min_logs})  = {len(missing)}")
    print(f"[stat] duplicates (n_logs>=2)         = {len(dups)}")
    for t in tags:
        pt = present[present["tag"] == t]
        mt = missing[missing["tag"] == t]
        print(f"[tag {t}] present={len(pt)}  missing={len(mt)}")

if __name__ == "__main__":
    main()
