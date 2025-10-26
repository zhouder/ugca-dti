from __future__ import annotations
import argparse, glob, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except Exception:
    AutoTokenizer = AutoModel = torch = None

MODEL = 'facebook/esm2_t6_8M_UR50D'

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:24]

def save_npz(out_dir: Path, key: str, arr: np.ndarray):
    out = out_dir / f"{key}.npz"
    np.savez_compressed(out, protein=arr.astype(np.float16))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_glob', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(args.csv_glob))
    if AutoTokenizer is None:
        print('transformers not available; writing zeros as placeholder')
    tok = mdl = None
    if AutoTokenizer is not None:
        tok = AutoTokenizer.from_pretrained(MODEL)
        mdl = AutoModel.from_pretrained(MODEL)
        mdl.eval()
    seen = set()
    for f in files:
        df = pd.read_csv(f)
        for seq in df['protein'].astype(str).unique():
            k = _hash(seq)
            if k in seen: continue
            seen.add(k)
            if mdl is None:
                arr = np.zeros((max(1,len(seq)), 128), dtype=np.float32)
            else:
                with torch.no_grad():
                    t = tok(seq, return_tensors='pt', truncation=True)
                    h = mdl(**t).last_hidden_state.squeeze(0).cpu().numpy()
                    arr = h
            save_npz(out, k, arr)
