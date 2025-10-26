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

MODEL = 'DeepChem/ChemBERTa-77M-MLM'

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:24]

def save_npz(out_dir: Path, key: str, vec: np.ndarray):
    out = out_dir / f"{key}.npz"
    np.savez_compressed(out, chemberta=vec.astype(np.float16))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_glob', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(args.csv_glob))
    tok = mdl = None
    if AutoTokenizer is None:
        print('transformers not available; writing zeros as placeholder')
    else:
        tok = AutoTokenizer.from_pretrained(MODEL)
        mdl = AutoModel.from_pretrained(MODEL)
        mdl.eval()
    seen = set()
    for f in files:
        df = pd.read_csv(f)
        for smi in df['smiles'].astype(str).unique():
            k = _hash(smi)
            if k in seen: continue
            seen.add(k)
            if mdl is None:
                vec = np.zeros((384,), dtype=np.float32)
            else:
                with torch.no_grad():
                    t = tok(smi, return_tensors='pt', truncation=True)
                    vec = mdl(**t).last_hidden_state[:,0,:].squeeze(0).cpu().numpy()
            save_npz(out, k, vec)
