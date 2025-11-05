# src/datamodule.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib, time, csv, random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------- configs ----------------
@dataclass
class DMConfig:
    train_csv: str
    test_csv: str
    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = False

@dataclass
class CacheDirs:
    esm2_dir: str
    molclr_dir: str
    chemberta_dir: str

@dataclass
class CacheDims:
    esm2: int
    molclr: int
    chemberta: int

# ---------------- utils ----------------
def _sha1_24(s: str) -> str:
    # 离线脚本使用的命名规则：sha1(s) 的前 24 位
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]

def _index_npz_dir(dir_path: Path) -> Dict[str, Path]:
    tbl: Dict[str, Path] = {}
    if not dir_path.exists():
        return tbl
    for p in dir_path.rglob("*.npz"):
        tbl[p.stem] = p
    return tbl

def _npz_load_first_key(p: Optional[Path]) -> Optional[np.ndarray]:
    if p is None or not p.exists():
        return None
    with np.load(str(p)) as z:
        k = list(z.files)[0]
        return z[k]

def _as_1d(x: Optional[np.ndarray], target_dim: int) -> np.ndarray:
    if x is None:
        out = np.zeros((target_dim,), dtype=np.float32)
    else:
        try:
            x = np.asarray(x)
            if x.ndim == 2:
                x = x.mean(axis=0)
            x = np.asarray(x, dtype=np.float32)
        except Exception:
            return np.zeros((target_dim,), dtype=np.float32)
        x = x.reshape(-1)
        if x.shape[0] < target_dim:
            pad = np.zeros((target_dim - x.shape[0],), dtype=np.float32)
            out = np.concatenate([x, pad], axis=0)
        else:
            out = x[:target_dim]
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out, dtype=np.float32)
    return out

def _to_tensor(x: np.ndarray) -> torch.Tensor:
    try:
        arr = np.array(x, dtype=np.float32, copy=True, order="C")
    except Exception:
        arr = np.zeros((1,), dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float32)

# ---- CSV（只取三列） ----
def _read_smiles_protein_label(path: str) -> Tuple[List[str], List[str], List[float]]:
    smiles, proteins, labels = [], [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        header = next(reader, None)

        si = pi = li = None
        if header:
            heads = [c.strip().lower() for c in header]
            if all(x in heads for x in ("smiles", "protein", "label")):
                si, pi, li = heads.index("smiles"), heads.index("protein"), heads.index("label")
            else:
                si, pi, li = 0, 1, 2
                if len(header) >= 3:
                    try:
                        labels.append(float(header[2])); smiles.append(header[0]); proteins.append(header[1])
                    except Exception:
                        pass
        else:
            si, pi, li = 0, 1, 2

        for row in reader:
            if not row or len(row) <= max(si, pi, li):
                continue
            s, p, l = row[si], row[pi], row[li]
            try:
                labels.append(float(l)); smiles.append(s); proteins.append(p)
            except Exception:
                continue
    return smiles, proteins, labels

# ---- 关键：生成多种 key 变体并查表 ----
_AMINO = set("ACDEFGHIKLMNPQRSTVWY")  # 20 标准氨基酸

def _prot_variants(s: str) -> List[str]:
    raw = s or ""
    v = []
    v.append(raw)
    v.append(raw.strip())
    v.append("".join(raw.split()))                  # 去所有空白
    v.append("".join(raw.split()).upper())         # 去空白 + 大写
    # 非法字符清理（只保留 20 种氨基酸）
    only_aa = "".join(ch for ch in raw if ch.upper() in _AMINO)
    if only_aa:
        v.append(only_aa)
        v.append(only_aa.upper())
    # 去重保持顺序
    out, seen = [], set()
    for x in v:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _smiles_variants(s: str) -> List[str]:
    raw = s or ""
    v = []
    v.append(raw)
    v.append(raw.strip())
    v.append(raw.replace(" ", ""))                 # 去空格
    v.append("".join(raw.split()))                 # 去所有空白
    # RDKit 规范化此处不做（避免依赖），先尽量命中已有 md5 命名
    out, seen = [], set()
    for x in v:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _lookup(tbl: Dict[str, Path], keys: List[str]) -> Optional[Path]:
    for k in keys:
        key = _sha1_24(k)
        p = tbl.get(key)
        if p is not None:
            return p
    return None


# ---------------- dataset ----------------
class DTIDataset(Dataset):
    def __init__(self, csv_path: str,
                 esm2_tbl: Dict[str, Path],
                 molclr_tbl: Dict[str, Path],
                 chem_tbl: Dict[str, Path],
                 dims: CacheDims):
        self.smiles, self.proteins, self.labels = _read_smiles_protein_label(csv_path)

        self.esm2_tbl = esm2_tbl
        self.molclr_tbl = molclr_tbl
        self.chem_tbl = chem_tbl
        self.dims = dims

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        pro = self.proteins[idx]
        y   = float(self.labels[idx])

        # 多变体查找
        p_path = _lookup(self.esm2_tbl, _prot_variants(pro))
        d1_path = _lookup(self.molclr_tbl, _smiles_variants(smi))
        d2_path = _lookup(self.chem_tbl, _smiles_variants(smi))

        v_p  = _as_1d(_npz_load_first_key(p_path),  self.dims.esm2)
        v_d1 = _as_1d(_npz_load_first_key(d1_path), self.dims.molclr)
        v_d2 = _as_1d(_npz_load_first_key(d2_path), self.dims.chemberta)

        return _to_tensor(v_p), _to_tensor(v_d1), _to_tensor(v_d2), torch.tensor(y, dtype=torch.float32)

# ---------------- datamodule ----------------
class DataModule:
    def __init__(self, cfg: DMConfig, cache_dirs: CacheDirs, dims: CacheDims):
        self.cfg = cfg
        self.cache_dirs = cache_dirs
        self.dims = dims

        t0 = time.time()
        self.esm2_tbl = _index_npz_dir(Path(self.cache_dirs.esm2_dir))
        self.molclr_tbl = _index_npz_dir(Path(self.cache_dirs.molclr_dir))
        self.chem_tbl  = _index_npz_dir(Path(self.cache_dirs.chemberta_dir))
        t1 = time.time()
        print(f"[PRELOAD] Index ESM2={len(self.esm2_tbl)} MolCLR={len(self.molclr_tbl)} ChemBERTa={len(self.chem_tbl)}   took {t1 - t0:.1f}s")

        # 自检：抽样统计命中率（各 2000 条以内）
        self._report_hit_rates(cfg.train_csv, tag="train")
        self._report_hit_rates(cfg.test_csv,  tag="test")

    def _report_hit_rates(self, csv_path: str, tag: str):
        smi, pro, lab = _read_smiles_protein_label(csv_path)
        n = len(lab)
        if n == 0:
            print(f"[HIT/{tag}] empty CSV: {csv_path}")
            return
        idxs = list(range(n))
        if n > 2000:
            random.seed(42); idxs = random.sample(idxs, 2000)

        hit_p = hit_d1 = hit_d2 = hit_all = 0
        for i in idxs:
            p = _lookup(self.esm2_tbl, _prot_variants(pro[i]))
            d1 = _lookup(self.molclr_tbl, _smiles_variants(smi[i]))
            d2 = _lookup(self.chem_tbl,  _smiles_variants(smi[i]))
            hp = p is not None
            hd1 = d1 is not None
            hd2 = d2 is not None
            hit_p += hp
            hit_d1 += hd1
            hit_d2 += hd2
            hit_all += (hp and hd1 and hd2)

        total = len(idxs)
        def pct(x): return f"{x/total*100:.1f}%"
        print(f"[HIT/{tag}] sample={total} | ESM2={pct(hit_p)} MolCLR={pct(hit_d1)} ChemBERTa={pct(hit_d2)} | ALL-3={pct(hit_all)}")

    def train_loader(self) -> DataLoader:
        ds = DTIDataset(self.cfg.train_csv, self.esm2_tbl, self.molclr_tbl, self.chem_tbl, self.dims)
        return DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory, persistent_workers=self.cfg.persistent_workers,
            prefetch_factor=self.cfg.prefetch_factor, drop_last=self.cfg.drop_last,
            collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b]),
                                  torch.stack([x[2] for x in b]), torch.stack([x[3] for x in b]))
        )

    def test_loader(self) -> DataLoader:
        ds = DTIDataset(self.cfg.test_csv, self.esm2_tbl, self.molclr_tbl, self.chem_tbl, self.dims)
        return DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory, persistent_workers=self.cfg.persistent_workers,
            prefetch_factor=self.cfg.prefetch_factor, drop_last=False,
            collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b]),
                                  torch.stack([x[2] for x in b]), torch.stack([x[3] for x in b]))
        )
