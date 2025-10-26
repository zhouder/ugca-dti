from __future__ import annotations
import argparse, glob, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:24]

def atom_feats(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros((1, 16), dtype=np.float32)
    feats = []
    for a in mol.GetAtoms():
        feats.append([
            a.GetAtomicNum(), a.GetTotalDegree(), int(a.GetIsAromatic()), a.GetTotalNumHs(), a.GetFormalCharge(),
            int(a.IsInRing()), a.GetMass()*0.01, a.GetImplicitValence(), a.GetHybridization()
        ])
    return np.array(feats, dtype=np.float32)

def save_npz(out_dir: Path, key: str, arr: np.ndarray):
    out = out_dir / f"{key}.npz"
    np.savez_compressed(out, drug_atoms=arr.astype(np.float16))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_glob', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(args.csv_glob))
    seen = set()
    for f in files:
        df = pd.read_csv(f)
        for smi in df['smiles'].astype(str).unique():
            k = _hash(smi)
            if k in seen: continue
            seen.add(k)
            arr = atom_feats(smi)
            save_npz(out, k, arr)
