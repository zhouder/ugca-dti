from __future__ import annotations
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def _hash(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode()).hexdigest()[:24]

@dataclass
class DMConfig:
    train_csv: str
    test_csv: str
    batch_size: int = 32
    num_workers: int = 8
    cache_dirs: Dict[str, str] = None  # {esm2, molclr, chemberta}

def _load_cache(cache_root: str, key: str):
    p = Path(cache_root) / f"{key}.npz"
    if p.exists():
        return dict(np.load(p, allow_pickle=False))
    return None

class DTIDataset(Dataset):
    def __init__(self, csv_path: str, cache_dirs: Dict[str,str]):
        self.df = pd.read_csv(csv_path)
        assert set(['smiles','protein','label']).issubset(self.df.columns)
        self.cache_dirs = cache_dirs

    def __len__(self): return len(self.df)

    def __getitem__(self, i) -> Dict[str, Any]:
        r = self.df.iloc[i]
        smi = str(r['smiles']); seq = str(r['protein']); y = float(r['label'])
        k_smi = _hash(smi); k_seq = _hash(seq)
        drug = _load_cache(self.cache_dirs['molclr'], k_smi) or {}
        prot = _load_cache(self.cache_dirs['esm2'],   k_seq) or {}
        chem = _load_cache(self.cache_dirs['chemberta'], k_smi) or {}
        H_D = drug.get('drug_atoms', np.zeros((1, 64), dtype=np.float32))
        H_P = prot.get('protein',    np.zeros((max(1,len(seq)), 64), dtype=np.float32))
        h_C = chem.get('chemberta',  np.zeros((384,), dtype=np.float32))
        return {
            'smiles': smi, 'protein': seq, 'label': y,
            'H_D': H_D, 'H_P': H_P, 'h_C': h_C,
        }

def _pad2d(arrs: List[np.ndarray]):
    maxn = max(a.shape[0] for a in arrs)
    dim  = arrs[0].shape[1]
    out = torch.zeros((len(arrs), maxn, dim), dtype=torch.float32)
    mask = torch.zeros((len(arrs), maxn), dtype=torch.bool)
    for i,a in enumerate(arrs):
        n = a.shape[0]
        out[i,:n,:] = torch.from_numpy(a.astype(np.float32))
        mask[i,:n] = True
    return out, mask

def collate(batch: List[Dict[str,Any]]) -> Dict[str,Any]:
    HD = [b['H_D'] for b in batch]
    HP = [b['H_P'] for b in batch]
    import numpy as np
    hC = np.stack([b['h_C'] for b in batch]).astype(np.float32)
    y  = torch.tensor([b['label'] for b in batch], dtype=torch.float32)
    HD_t, mD = _pad2d(HD)
    HP_t, mP = _pad2d(HP)
    hC_t = torch.from_numpy(hC)
    return {
        'H_D': HD_t, 'mask_D': mD,
        'H_P': HP_t, 'mask_P': mP,
        'h_C': hC_t,
        'labels': y,
        'info': {'smiles': [b['smiles'] for b in batch], 'protein': [b['protein'] for b in batch]},
    }

class DataModule:
    def __init__(self, cfg: DMConfig):
        self.cfg = cfg
        self.train_set = DTIDataset(cfg.train_csv, cfg.cache_dirs)
        self.test_set  = DTIDataset(cfg.test_csv,  cfg.cache_dirs)
    def loaders(self):
        tr = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, collate_fn=collate)
        te = DataLoader(self.test_set,  batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, collate_fn=collate)
        return tr, te
