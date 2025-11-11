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
    # 仅使用“内存数据”模式：直接传 (smiles, proteins, labels) 列表
    train_data: Tuple[List[str], List[str], List[float]]
    val_data:   Tuple[List[str], List[str], List[float]]
    test_data:  Tuple[List[str], List[str], List[float]]
    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = False
    # 是否以“序列级（per-token）”模式工作
    sequence: bool = False

@dataclass
class CacheDirs:
    esm_dir: str        # 传任意一个（esm 或 esm2），内部会自动在二者间回退
    molclr_dir: str
    chemberta_dir: str

@dataclass
class CacheDims:
    esm2: int
    molclr: int
    chemberta: int

# ---------------- utils ----------------
def _sha1_24(s: str) -> str:
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
    # V1 路径：把 2D 平均到 1D
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

def _as_2d(x: Optional[np.ndarray]) -> np.ndarray:
    # V2 路径：尽量保留成 (T, D)；若 1D 则升维到 (1, D)
    if x is None:
        return np.zeros((0, 1), dtype=np.float32)
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    x = np.asarray(x, dtype=np.float32, order="C")
    return x

def _to_tensor(x: np.ndarray) -> torch.Tensor:
    try:
        arr = np.array(x, dtype=np.float32, copy=True, order="C")
    except Exception:
        arr = np.zeros((1,), dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float32)

def _read_smiles_protein_label(path: str) -> Tuple[List[str], List[str], List[float]]:
    """读取 all.csv（列：smiles,protein,label）"""
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

_AMINO = set("ACDEFGHIKLMNPQRSTVWY")
def _prot_variants(s: str) -> List[str]:
    raw = s or ""
    v = [raw, raw.strip(), "".join(raw.split()), "".join(raw.split()).upper()]
    only_aa = "".join(ch for ch in raw if ch.upper() in _AMINO)
    if only_aa:
        v += [only_aa, only_aa.upper()]
    out, seen = [], set()
    for x in v:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _smiles_variants(s: str) -> List[str]:
    raw = s or ""
    v = [raw, raw.strip(), raw.replace(" ", ""), "".join(raw.split())]
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

# ---------------- dataset：仅内存版 ----------------
class DTIDatasetMem(Dataset):
    """直接接收切片后的 (smiles, protein, label) 列表"""
    def __init__(self,
                 data: Tuple[List[str], List[str], List[float]],
                 esm_tbl: Dict[str, Path],
                 molclr_tbl: Dict[str, Path],
                 chem_tbl: Dict[str, Path],
                 dims: CacheDims,
                 sequence: bool = False):
        self.smiles, self.proteins, self.labels = data
        self.esm_tbl = esm_tbl
        self.molclr_tbl = molclr_tbl
        self.chem_tbl = chem_tbl
        self.dims = dims
        self.sequence = sequence
        self._cache_p, self._cache_d1, self._cache_d2 = {}, {}, {}

    def __len__(self):
        return len(self.labels)

    def _vec_cached(self, p: Optional[Path], d: int, cache: dict) -> torch.Tensor:
        if p is None:
            return torch.zeros((d,), dtype=torch.float32)
        k = str(p)
        v = cache.get(k)
        if v is None:
            arr = _as_1d(_npz_load_first_key(Path(k)), d)
            v = _to_tensor(arr)
            cache[k] = v
        return v

    def _seq_cached(self, p: Optional[Path], cache: dict) -> np.ndarray:
        if p is None:
            return np.zeros((0, 1), dtype=np.float32)
        k = str(p)
        v = cache.get(k)
        if v is None:
            arr = _as_2d(_npz_load_first_key(Path(k)))
            cache[k] = arr
            v = arr
        return v

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        pro = self.proteins[idx]
        y   = float(self.labels[idx])

        p_path  = _lookup(self.esm_tbl,   _prot_variants(pro))
        d1_path = _lookup(self.molclr_tbl, _smiles_variants(smi))
        d2_path = _lookup(self.chem_tbl,   _smiles_variants(smi))

        if not self.sequence:
            v_p  = self._vec_cached(p_path,  self.dims.esm2,      self._cache_p)
            v_d1 = self._vec_cached(d1_path, self.dims.molclr,    self._cache_d1)
            v_d2 = self._vec_cached(d2_path, self.dims.chemberta, self._cache_d2)
            return v_p, v_d1, v_d2, torch.tensor(y, dtype=torch.float32)
        else:
            P  = self._seq_cached(p_path,  self._cache_p)
            D1 = self._seq_cached(d1_path, self._cache_d1)
            C  = self._seq_cached(d2_path, self._cache_d2)
            if C.ndim == 2:
                C = C.mean(axis=0)
            return P, D1, C.astype(np.float32, copy=True), np.float32(y)

# ---------------- collate helpers ----------------
def _pad_and_mask(batch_list: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    # 输入一批 (Ti, D)，输出 (B, T_max, D) 和 (B, T_max) bool-mask
    lens = [x.shape[0] for x in batch_list]
    d = max([x.shape[1] for x in batch_list] + [1])
    tmax = max(lens + [1])
    B = len(batch_list)
    out = np.zeros((B, tmax, d), dtype=np.float32)
    mask = np.zeros((B, tmax), dtype=bool)
    for i, x in enumerate(batch_list):
        L = x.shape[0]
        out[i, :L, :x.shape[1]] = x
        mask[i, :L] = True
    return torch.tensor(out, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)

# ---------------- datamodule ----------------
class DataModule:
    def __init__(self, cfg: DMConfig, cache_dirs: CacheDirs, dims: CacheDims):
        self.cfg = cfg
        self.dims = dims

        # 兼容 esm / esm2 两种命名（自动回退）
        cand = [Path(cache_dirs.esm_dir)]
        if "/esm2/" in cache_dirs.esm_dir:
            cand.append(Path(cache_dirs.esm_dir.replace("/esm2/", "/esm/")))
        elif "/esm/" in cache_dirs.esm_dir:
            cand.append(Path(cache_dirs.esm_dir.replace("/esm/", "/esm2/")))
        picked = None
        esm_tbl = {}
        for c in cand:
            tbl = _index_npz_dir(c)
            if len(tbl) > 0:
                picked, esm_tbl = c, tbl
                break
        if picked is None:
            picked = cand[0]
            esm_tbl = _index_npz_dir(picked)

        self.esm_dir_used = str(picked)
        self.esm_tbl = esm_tbl
        self.molclr_tbl = _index_npz_dir(Path(cache_dirs.molclr_dir))
        self.chem_tbl  = _index_npz_dir(Path(cache_dirs.chemberta_dir))
        print(f"[CACHE] esm_dir_used={self.esm_dir_used}")
        print(f"[CACHE] molclr_dir  ={cache_dirs.molclr_dir}")
        print(f"[CACHE] chem_dir    ={cache_dirs.chemberta_dir}")

        t0 = time.time()
        print(f"[PRELOAD] Index ESM={len(self.esm_tbl)} MolCLR={len(self.molclr_tbl)} ChemBERTa={len(self.chem_tbl)}   took {time.time()-t0:.1f}s")

        # 命中率报告
        self._report_hit_rates_data(self.cfg.train_data, tag="train")
        self._report_hit_rates_data(self.cfg.val_data,   tag="val")
        self._report_hit_rates_data(self.cfg.test_data,  tag="test")

    def _report_hit_rates_data(self, data: Tuple[List[str], List[str], List[float]], tag: str):
        smi, pro, _ = data
        n = len(smi)
        if n == 0:
            print(f"[HIT/{tag}] empty data")
            return
        idxs = list(range(n))
        if n > 2000:
            random.seed(42); idxs = random.sample(idxs, 2000)
        hit_p = hit_d1 = hit_d2 = hit_all = 0
        for i in idxs:
            p = _lookup(self.esm_tbl,   _prot_variants(pro[i]))
            d1 = _lookup(self.molclr_tbl, _smiles_variants(smi[i]))
            d2 = _lookup(self.chem_tbl,   _smiles_variants(smi[i]))
            hp = p is not None
            hd1 = d1 is not None
            hd2 = d2 is not None
            hit_p += hp; hit_d1 += hd1; hit_d2 += hd2; hit_all += (hp and hd1 and hd2)
        total = len(idxs)
        def pct(x): return f"{x/total*100:.1f}%"
        print(f"[HIT/{tag}] sample={total} | ESM={pct(hit_p)} MolCLR={pct(hit_d1)} ChemBERTa={pct(hit_d2)} | ALL-3={pct(hit_all)}")

    # ----------- V1：向量模式 -----------
    def _collate_v1(self, b):
        return (torch.stack([x[0] for x in b]),
                torch.stack([x[1] for x in b]),
                torch.stack([x[2] for x in b]),
                torch.stack([torch.tensor(x[3], dtype=torch.float32) for x in b]))

    # ----------- V2：序列模式 -----------
    def _collate_v2(self, b):
        # b[i] = (P:(Mi,dp), D1:(Ni,dd), C:(dc,), y)
        P_list  = [x[0] for x in b]
        D_list  = [x[1] for x in b]
        C_list  = [x[2] for x in b]
        y_list  = [x[3] for x in b]
        P, Pm = _pad_and_mask(P_list)  # (B,Mmax,dp), (B,Mmax)
        D, Dm = _pad_and_mask(D_list)  # (B,Nmax,dd), (B,Nmax)
        C = torch.tensor(np.stack(C_list, axis=0), dtype=torch.float32)  # (B,dc)
        y = torch.tensor(np.asarray(y_list, dtype=np.float32), dtype=torch.float32)
        return P, Pm, D, Dm, C, y

    def _make_loader(self, data, shuffle: bool) -> DataLoader:
        ds = DTIDatasetMem(data, self.esm_tbl, self.molclr_tbl, self.chem_tbl, self.dims, sequence=self.cfg.sequence)
        cf = self._collate_v1 if not self.cfg.sequence else self._collate_v2
        return DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=shuffle, num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory, persistent_workers=self.cfg.persistent_workers,
            prefetch_factor=self.cfg.prefetch_factor, drop_last=self.cfg.drop_last if shuffle else False,
            collate_fn=cf
        )

    def train_loader(self) -> DataLoader:
        return self._make_loader(self.cfg.train_data, shuffle=True)

    def val_loader(self) -> DataLoader:
        return self._make_loader(self.cfg.val_data, shuffle=False)

    def test_loader(self) -> DataLoader:
        return self._make_loader(self.cfg.test_data, shuffle=False)
