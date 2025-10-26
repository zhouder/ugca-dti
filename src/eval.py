from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def gather_and_summarize(root: str, out: str):
    root = Path(root)
    recs = []
    for p in sorted(root.glob('fold*/best_metrics.json')):
        try:
            recs.append(json.loads(p.read_text())['metrics'])
        except Exception:
            recs.append(json.loads(p.read_text()))
    if not recs:
        raise SystemExit(f"No best_metrics.json under {root}")
    keys = ['auroc','auprc','f1','acc','sensitivity','mcc']
    stats = {}
    for k in keys:
        arr = np.array([r[k] for r in recs if k in r], dtype=float)
        stats[k] = {'mean': float(np.nanmean(arr)), 'std': float(np.nanstd(arr))}
    Path(out).write_text(json.dumps({'n': len(recs), 'metrics': stats}, indent=2, ensure_ascii=False))
    print(f"Saved summary to {out}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    gather_and_summarize(args.root, args.out)
