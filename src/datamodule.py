# src/datamodule.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import hashlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ===================== 配置 =====================

@dataclass
class DMConfig:
    train_csv: str
    test_csv: str
    batch_size: int = 256
    num_workers: int = 6
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = False

@dataclass
class CacheDirs:
    # 传 esm 目录，内部会自动尝试 esm / esm2 的回退
    esm2_dir: str        # e.g. /root/lanyun-tmp/cache/esm2/DAVIS 或 /cache/esm/DAVIS
    molclr_dir: str      # e.g. /root/lanyun-tmp/cache/molclr/DAVIS
    chemberta_dir: str   # e.g. /root/lanyun-tmp/cache/chemberta/DAVIS

@dataclass
class CacheDims:
    esm2: int = 1280
    molclr: int = 300
    chemberta: int = 384


# ===================== 工具函数 =====================

def _scan_npz_dir(dir_path: Path) -> Dict[str, Path]:
    """扫描目录，返回 {样本ID或哈希: 文件路径}（文件名不含后缀）。"""
    table: Dict[str, Path] = {}
    if not dir_path.exists():
        return table
    for p in dir_path.glob("*.npz"):
        table[p.stem] = p
    return table

def _try_esm_fallback(p: Path) -> Path:
    """esm/esm2 目录兜底：给 esm 路径时，若不存在则尝试 esm2；反之同理。"""
    if p.exists():
        return p
    nm = p.name.lower()
    if nm == "esm":
        cand = p.with_name("esm2")
        return cand if cand.exists() else p
    if nm == "esm2":
        cand = p.with_name("esm")
        return cand if cand.exists() else p
    return p

def _infer_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    推断 (drug_col, protein_col, label_col)
    支持：drug 别名含 'smiles','drug','drug_id','ligand','mol'
          protein 别名含 'protein','protein_id','target','seq','sequence'
          label 别名含 'label','y'
    """
    cols_lc = {c.lower(): c for c in df.columns}
    drug_cands = ["smiles", "drug", "drug_id", "ligand", "mol"]
    prot_cands = ["protein", "protein_id", "target", "seq", "sequence"]
    lab_cands  = ["label", "y"]

    drug_col = next((cols_lc[c] for c in drug_cands if c in cols_lc), None)
    prot_col = next((cols_lc[c] for c in prot_cands if c in cols_lc), None)
    lab_col  = next((cols_lc[c] for c in lab_cands  if c in cols_lc), None)

    if drug_col and prot_col and lab_col:
        return drug_col, prot_col, lab_col

    # 回退：按前三列
    names = list(df.columns)
    if len(names) < 3:
        raise ValueError("CSV 至少需要三列（smiles/protein/label 或其别名）。")
    return names[0], names[1], names[2]

def _load_npz_vec(path: Path) -> np.ndarray:
    """从 .npz 里读取向量，容错支持 'x'/'feat'/'arr_0' 等常见 key。"""
    with np.load(str(path)) as npz:
        for k in ("x", "feat", "arr_0"):
            if k in npz:
                arr = npz[k]; break
        else:
            k0 = list(npz.keys())[0]
            arr = npz[k0]
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim > 1:
        arr = arr.reshape(-1).astype(np.float32)
    return arr

def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _resolve_npz(idx: Dict[str, Path], key: str) -> Path | None:
    """
    尝试用原始键、md5(key) 两种方式命中缓存文件。
    说明：很多编码脚本会以 md5(文本) 保存文件名，尤其 SMILES/序列不方便做文件名时。
    """
    if key in idx:
        return idx[key]
    h = _md5(key)
    if h in idx:
        return idx[h]
    return None


# ===================== 数据集 =====================

class UGCDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        col_drug: str,
        col_prot: str,
        col_label: str,
        idx_esm: Dict[str, Path],
        idx_mol: Dict[str, Path],
        idx_chem: Dict[str, Path],
        dims: CacheDims,
        tag: str,
    ):
        self.df = df.reset_index(drop=True)
        self.c_d, self.c_p, self.c_y = col_drug, col_prot, col_label
        self.idx_esm = idx_esm
        self.idx_mol = idx_mol
        self.idx_chem = idx_chem
        self.dims = dims
        self.tag = tag

        # 命中率统计
        hits_e = hits_m = hits_c = 0
        for _, row in self.df.iterrows():
            did = str(row[self.c_d])
            pid = str(row[self.c_p])
            if _resolve_npz(self.idx_esm, pid):  hits_e += 1
            if _resolve_npz(self.idx_mol, did):  hits_m += 1
            if _resolve_npz(self.idx_chem, did): hits_c += 1
        n = len(self.df)
        print(f"[HIT/{self.tag}] sample={n} | ESM2={hits_e/n:>5.1%} MoLCLR={hits_m/n:>5.1%} ChemBERTa={hits_c/n:>5.1%} | ALL-3={(hits_e==n and hits_m==n and hits_c==n)}")

    def __len__(self): return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        did = str(r[self.c_d])
        pid = str(r[self.c_p])
        y   = float(r[self.c_y])

        p_path = _resolve_npz(self.idx_esm, pid)
        m_path = _resolve_npz(self.idx_mol, did)
        c_path = _resolve_npz(self.idx_chem, did)

        pe = _load_npz_vec(p_path) if p_path else np.zeros((self.dims.esm2,), dtype=np.float32)
        dm = _load_npz_vec(m_path) if m_path else np.zeros((self.dims.molclr,), dtype=np.float32)
        dc = _load_npz_vec(c_path) if c_path else np.zeros((self.dims.chemberta,), dtype=np.float32)

        pe = torch.from_numpy(pe).float().view(-1)
        dm = torch.from_numpy(dm).float().view(-1)
        dc = torch.from_numpy(dc).float().view(-1)
        y  = torch.tensor(y, dtype=torch.float32)
        return pe, dm, dc, y


# ===================== DataModule =====================

class DataModule:
    def __init__(self, cfg: DMConfig, cache_dirs: CacheDirs, dims: CacheDims):
        self.cfg = cfg
        self.cache_dirs = cache_dirs
        self.dims = dims

        # —— 解析 CSV
        self.train_df = pd.read_csv(self.cfg.train_csv)
        self.test_df  = pd.read_csv(self.cfg.test_csv)

        # —— 推断列名（兼容 smiles/protein/label）
        self.col_d, self.col_p, self.col_y = _infer_columns(self.train_df)

        # —— 索引编码缓存
        self.idx_esm_train, self.idx_mol_train, self.idx_chem_train = self._build_indices(for_dataset="train")
        self.idx_esm_test,  self.idx_mol_test,  self.idx_chem_test  = self._build_indices(for_dataset="test")

        # —— 构造 Dataset
        self.train_set = UGCDataset(
            self.train_df, self.col_d, self.col_p, self.col_y,
            self.idx_esm_train, self.idx_mol_train, self.idx_chem_train,
            self.dims, tag="train"
        )
        print(f"[INFO] train size = {len(self.train_set):>5d} | batch_size = {self.cfg.batch_size}")

        self.test_set = UGCDataset(
            self.test_df, self.col_d, self.col_p, self.col_y,
            self.idx_esm_test, self.idx_mol_test, self.idx_chem_test,
            self.dims, tag="test"
        )
        print(f"[INFO] test  size = {len(self.test_set):>5d}")

    def _build_indices(self, for_dataset: str):
        """
        cache 目录结构假设：
          cache/
            esm2/      <DatasetName>/*.npz   （或 esm/<DatasetName>/*.npz）
            molclr/    <DatasetName>/*.npz
            chemberta/ <DatasetName>/*.npz
        其中 <DatasetName> 既支持 DAVIS 也支持 davis（内部会统一为 DAVIS）。
        """
        csv_path = Path(self.cfg.train_csv if for_dataset == "train" else self.cfg.test_csv)
        dataset_name = csv_path.parent.name  # e.g. davis
        # 统一大小写
        name_map = {"davis": "DAVIS", "bindingdb": "BindingDB", "biosnap": "BioSNAP"}
        ds_std = name_map.get(dataset_name.lower(), dataset_name)

        # 三路基目录
        esm_root = Path(self.cache_dirs.esm2_dir).parent / ds_std
        esm_root = _try_esm_fallback(esm_root.parent / ("esm2" if "esm" not in esm_root.name.lower() else esm_root.name)) / ds_std
        if not esm_root.exists():
            # 兜底：直接用传入目录（适配你直接传 /cache/esm2/DAVIS 的情况）
            esm_root = _try_esm_fallback(Path(self.cache_dirs.esm2_dir))

        mol_root  = Path(self.cache_dirs.molclr_dir).parent / ds_std
        chem_root = Path(self.cache_dirs.chemberta_dir).parent / ds_std

        print(f"[{for_dataset.upper()}] cache roots:")
        print(f"  ESM2/ESM : {esm_root}")
        print(f"  MolCLR   : {mol_root}")
        print(f"  ChemBERTa: {chem_root}")

        idx_esm  = _scan_npz_dir(esm_root)
        idx_mol  = _scan_npz_dir(mol_root)
        idx_chem = _scan_npz_dir(chem_root)

        print(f"[{for_dataset.upper()}] index sizes: ESM2={len(idx_esm)} MoLCLR={len(idx_mol)} ChemBERTa={len(idx_chem)}")
        return idx_esm, idx_mol, idx_chem

    # ---------- DataLoader ----------
    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=(self.cfg.num_workers > 0 and self.cfg.persistent_workers),
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
            drop_last=self.cfg.drop_last,
        )

    def test_loader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=(self.cfg.num_workers > 0 and self.cfg.persistent_workers),
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
            drop_last=False,
        )
