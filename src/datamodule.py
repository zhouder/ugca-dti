from __future__ import annotations
# -*- coding: utf-8 -*-

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------- Config ----------------
@dataclass
class DMConfig:
    train_csv: str
    test_csv: str
    num_workers: int = 8
    batch_size: int = 64
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
    esm2: int = 1280
    molclr: int = 300
    chemberta: int = 384

# ---------------- Utils ----------------
def _read_table(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    sep = _sniff_sep(p)
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=sep)
        return [r for r in reader]

def _sniff_sep(path: Path) -> str:
    with path.open("rb") as f:
        head = f.read(4096)
    text = head.decode("utf-8", errors="ignore")
    return "\t" if text.count("\t") > text.count(",") else ","

def _npz_load_first_key(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path, allow_pickle=True, mmap_mode=None)
    k = list(d.keys())[0]
    return d[k]

def _zeros(d: int, dtype=np.float32) -> np.ndarray:
    return np.zeros((d,), dtype=dtype)

def _to_binary(v: Any) -> int:
    s = str(v).strip().lower()
    if s in {"1","1.0","true","t","yes","y","active"}:  return 1
    if s in {"0","0.0","false","f","no","n","inactive"}: return 0
    try:    return int(float(s) >= 0.5)
    except: return 0

def _as_1d_float32(arr: Any, target_dim: int) -> np.ndarray:
    """
    把任意形状/类型的输入（含 object 数组、list、2D）规整为 (target_dim,) 的 np.float32 向量。
    - object/列表：尝试 stack -> 2D
    - 2D：对第0维做均值 -> 1D
    - 写成 C 连续、可写
    - 维度不符自动 pad 或截断
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=object)

    if arr.dtype == object:
        try:
            arr = np.stack([np.asarray(x) for x in arr], axis=0)
        except Exception:
            arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim >= 2:
        arr = arr.reshape(arr.shape[0], -1).mean(0)

    v = np.asarray(arr, dtype=np.float32)
    if not v.flags.c_contiguous:
        v = np.ascontiguousarray(v)

    if v.shape[0] != target_dim:
        if v.shape[0] < target_dim:
            pad = np.zeros((target_dim - v.shape[0],), dtype=np.float32)
            v = np.concatenate([v, pad], axis=0)
        else:
            v = v[:target_dim]

    if not v.flags.writeable:
        v = v.copy()
    return v

def _stack_to_tensor(items: List[Any]) -> torch.Tensor:
    x0 = items[0]
    if isinstance(x0, torch.Tensor):
        return torch.stack(items, dim=0)
    if isinstance(x0, np.ndarray):
        arr = np.stack(items, axis=0)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        if arr.dtype not in (np.float32, np.float16, np.float64):
            arr = arr.astype(np.float32, copy=False)
        return torch.tensor(arr, dtype=torch.float32)  # 安全转换
    return torch.tensor(items, dtype=torch.float32)

def _to_tensor_safe(v: Any) -> torch.Tensor:
    """
    比 from_numpy 更鲁棒的转换：总是先 np.asarray(float32)，
    再 torch.tensor(copy=True)；可兼容 object/只读/跨模块 ndarray。
    """
    arr = np.asarray(v, dtype=np.float32)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return torch.tensor(arr, dtype=torch.float32)

# ---------------- Dataset ----------------
class DTIDataset(Dataset):
    def __init__(self, csv_path: str, cache_dirs: CacheDirs, dims: CacheDims):
        self.path = csv_path
        self.rows = _read_table(csv_path)
        self.dims = dims

        self.esm2_dir = Path(cache_dirs.esm2_dir)
        self.molclr_dir = Path(cache_dirs.molclr_dir)
        self.chemberta_dir = Path(cache_dirs.chemberta_dir)

        self._protein_cache: Dict[str, torch.Tensor] = {}
        self._drug_cache: Dict[str, torch.Tensor] = {}
        self._chem_cache: Dict[str, torch.Tensor] = {}

        self._esm2_tbl = self._build_lookup(self.esm2_dir)
        self._molclr_tbl = self._build_lookup(self.molclr_dir)
        self._chem_tbl = self._build_lookup(self.chemberta_dir)

        import time
        t0 = time.time()
        for i in range(min(2, len(self.rows))):
            _ = self._fetch_one(i)
        print(f"[PRELOAD] ESM2~{len(self._protein_cache)}, MolCLR~{len(self._drug_cache)}, ChemBERTa~{len(self._chem_cache)}  |  took {time.time()-t0:.1f}s")

    @staticmethod
    def _build_lookup(dir_path: Path) -> Dict[str, Path]:
        tbl = {}
        if dir_path.exists():
            for p in dir_path.glob("*.npz"):
                tbl[p.stem] = p
        return tbl

    def __len__(self): return len(self.rows)

    def _find_any_npz(self, tbl: Dict[str, Path]) -> Optional[Path]:
        if not tbl: return None
        return next(iter(tbl.values()))

    def _fetch_one(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        row = self.rows[idx]
        y = _to_binary(row.get("label", row.get("Label", row.get("y", "0"))))

        # --- ESM2 ---
        t_v1 = self._protein_cache.get(str(idx))
        if t_v1 is None:
            p = self._find_any_npz(self._esm2_tbl)
            if p and p.exists():
                arr = _npz_load_first_key(p)
                v = _as_1d_float32(arr, self.dims.esm2)
            else:
                v = _zeros(self.dims.esm2)
            t_v1 = _to_tensor_safe(v) if not isinstance(v, torch.Tensor) else v
            self._protein_cache[str(idx)] = t_v1

        # --- MolCLR ---
        t_v2 = self._drug_cache.get(str(idx))
        if t_v2 is None:
            p = self._find_any_npz(self._molclr_tbl)
            if p and p.exists():
                arr = _npz_load_first_key(p)
                v = _as_1d_float32(arr, self.dims.molclr)
            else:
                v = _zeros(self.dims.molclr)
            t_v2 = _to_tensor_safe(v) if not isinstance(v, torch.Tensor) else v
            self._drug_cache[str(idx)] = t_v2

        # --- ChemBERTa ---
        t_v3 = self._chem_cache.get(str(idx))
        if t_v3 is None:
            p = self._find_any_npz(self._chem_tbl)
            if p and p.exists():
                arr = _npz_load_first_key(p)
                v = _as_1d_float32(arr, self.dims.chemberta)
            else:
                v = _zeros(self.dims.chemberta)
            t_v3 = _to_tensor_safe(v) if not isinstance(v, torch.Tensor) else v
            self._chem_cache[str(idx)] = t_v3

        return t_v1, t_v2, t_v3, float(y)

    def __getitem__(self, idx: int):
        return self._fetch_one(idx)

# ---------------- collate ----------------
def dti_collate(batch: List[Tuple[Any, Any, Any, Any]]):
    v1s, v2s, v3s, ys = list(zip(*batch))
    V1 = _stack_to_tensor(list(v1s))
    V2 = _stack_to_tensor(list(v2s))
    V3 = _stack_to_tensor(list(v3s))
    Y  = torch.as_tensor(ys, dtype=torch.float32)
    return V1, V2, V3, Y

# ---------------- DataModule ----------------
class DataModule:
    def __init__(self, cfg: DMConfig, cache_dirs: CacheDirs, dims: CacheDims):
        self.cfg = cfg
        self.dims = dims
        self.train_set = DTIDataset(cfg.train_csv, cache_dirs, dims)
        self.test_set  = DTIDataset(cfg.test_csv,  cache_dirs, dims)

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
            drop_last=self.cfg.drop_last,
            collate_fn=dti_collate,
        )

    def test_loader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
            collate_fn=dti_collate,
        )
