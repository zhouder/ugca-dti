# src/datamodule.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ===================== 配置结构体 =====================

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
    esm2_dir: str        # e.g. /root/lanyun-tmp/cache/esm/DAVIS
    molclr_dir: str      # e.g. /root/lanyun-tmp/cache/molclr/DAVIS
    chemberta_dir: str   # e.g. /root/lanyun-tmp/cache/chemberta/DAVIS

@dataclass
class CacheDims:
    esm2: int = 1280
    molclr: int = 300
    chemberta: int = 384


# ===================== 工具函数 =====================

def _scan_npz_dir(dir_path: Path) -> Dict[str, Path]:
    """扫描目录，返回 {样本ID: 文件路径}。样本ID取文件名（不含后缀）"""
    table: Dict[str, Path] = {}
    if not dir_path.exists():
        return table
    for p in dir_path.glob("*.npz"):
        key = p.stem
        table[key] = p
    return table

def _try_esm_fallback(p: Path) -> Path:
    """esm/esm2 目录兜底：给 esm 路径时，若不存在则尝试 esm2；反之同理。"""
    if p.exists():
        return p
    name = p.name.lower()
    if name == "esm":
        cand = p.with_name("esm2")
        return cand if cand.exists() else p
    if name == "esm2":
        cand = p.with_name("esm")
        return cand if cand.exists() else p
    return p

def _infer_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    根据常见列名推断 (drug_id_col, protein_id_col, label_col)
    支持的一些别名：drug / drug_id / ligand / mol / smiles_id
                   protein / protein_id / target / seq_id
                   label / y
    """
    candidates = [
        (["drug", "drug_id", "ligand", "mol", "smiles_id"], ["protein", "protein_id", "target", "seq_id"], ["label", "y"]),
    ]
    cols = set(c.lower() for c in df.columns)
    for drug_cands, prot_cands, lab_cands in candidates:
        drug_col = next((c for c in drug_cands if c in cols), None)
        prot_col = next((c for c in prot_cands if c in cols), None)
        lab_col  = next((c for c in lab_cands  if c in cols), None)
        if drug_col and prot_col and lab_col:
            return drug_col, prot_col, lab_col
    # 回退：尽量匹配前三列
    names = list(df.columns)
    if len(names) < 3:
        raise ValueError("CSV 至少需要三列（drug/protein/label 或其他可识别别名）。")
    return names[0], names[1], names[2]

def _load_npz_vec(path: Path) -> np.ndarray:
    """从 .npz 里读取向量，容错支持常见 key：'x' / 'feat' / 'arr_0'。"""
    with np.load(str(path)) as npz:
        for k in ("x", "feat", "arr_0"):
            if k in npz:
                arr = npz[k]
                break
        else:
            # 取第一个数组
            k0 = list(npz.keys())[0]
            arr = npz[k0]
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim > 1:
        arr = arr.reshape(-1).astype(np.float32)
    return arr


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
    ):
        self.df = df.reset_index(drop=True)
        self.c_d, self.c_p, self.c_y = col_drug, col_prot, col_label
        self.idx_esm = idx_esm
        self.idx_mol = idx_mol
        self.idx_chem = idx_chem
        self.dims = dims

        # 预计算命中统计
        hits_e, hits_m, hits_c = 0, 0, 0
        for _, row in self.df.iterrows():
            did = str(row[self.c_d])
            pid = str(row[self.c_p])
            if pid in self.idx_esm:  hits_e += 1
            if did in self.idx_mol:  hits_m += 1
            if did in self.idx_chem: hits_c += 1
        n = len(self.df)
        print(f"[PRELOAD] Index ESM2={len(self.idx_esm)} MoLCLR={len(self.idx_mol)} ChemBERTa={len(self.idx_chem)}")
        print(f"[HIT/train] sample={n} | ESM2={hits_e/n:>5.1%} MoLCLR={hits_m/n:>5.1%} ChemBERTa={hits_c/n:>5.1%} | ALL-3={(hits_e==n and hits_m==n and hits_c==n)}")

    def __len__(self): return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        did = str(r[self.c_d])
        pid = str(r[self.c_p])
        y   = float(r[self.c_y])

        # 读取向量（缺失则用零向量兜底）
        if pid in self.idx_esm:
            pe = _load_npz_vec(self.idx_esm[pid])
        else:
            pe = np.zeros((self.dims.esm2,), dtype=np.float32)

        if did in self.idx_mol:
            dm = _load_npz_vec(self.idx_mol[did])
        else:
            dm = np.zeros((self.dims.molclr,), dtype=np.float32)

        if did in self.idx_chem:
            dc = _load_npz_vec(self.idx_chem[did])
        else:
            dc = np.zeros((self.dims.chemberta,), dtype=np.float32)

        # numpy -> torch
        pe = torch.from_numpy(pe).float()
        dm = torch.from_numpy(dm).float()
        dc = torch.from_numpy(dc).float()
        y  = torch.tensor(y, dtype=torch.float32)

        # 保证形状一致（1D 向量）
        pe = pe.view(-1)
        dm = dm.view(-1)
        dc = dc.view(-1)

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

        # 推断列名
        self.col_d, self.col_p, self.col_y = _infer_columns(self.train_df)
        # test 缺列名时，跟随 train 同名列
        for need_col in (self.col_d, self.col_p, self.col_y):
            if need_col not in set(c.lower() for c in self.test_df.columns):
                # 如果 test 列不一致，尝试通过位置映射
                # 简单容错：按位置拷贝一致的列名
                pass

        # —— 索引编码缓存
        self.idx_esm_train, self.idx_mol_train, self.idx_chem_train = self._build_indices(for_dataset="train")
        self.idx_esm_test,  self.idx_mol_test,  self.idx_chem_test  = self._build_indices(for_dataset="test")

        # —— 构造 Dataset
        self.train_set = UGCDataset(
            self.train_df, self.col_d, self.col_p, self.col_y,
            self.idx_esm_train, self.idx_mol_train, self.idx_chem_train,
            self.dims
        )
        print(f"[INFO] train size = {len(self.train_set):>5d} | batch_size = {self.cfg.batch_size}")

        self.test_set = UGCDataset(
            self.test_df, self.col_d, self.col_p, self.col_y,
            self.idx_esm_test, self.idx_mol_test, self.idx_chem_test,
            self.dims
        )
        print(f"[INFO] test  size = {len(self.test_set):>5d}")

    # ---------- 公共 ----------
    def _build_indices(self, for_dataset: str):
        """
        根据 CSV 所属的数据集（DAVIS / BindingDB / BioSNAP），到 cache 子目录下扫描 npz。
        这里假设目录结构为：
          cache/
            esm/       <DatasetName>/*.npz
            esm2/      <DatasetName>/*.npz     （可二选一）
            molclr/    <DatasetName>/*.npz
            chemberta/ <DatasetName>/*.npz
        """
        # 推断数据集名（DAVIS/BindingDB/BioSNAP），从 csv 路径中提取
        csv_path = Path(self.cfg.train_csv if for_dataset == "train" else self.cfg.test_csv)
        # 默认使用上层目录名，如 .../davis_k5/fold1_train.csv -> davis 或 DAVIS
        dataset_name = csv_path.parent.name.split("_")[0]
        # 大小写规范化映射
        name_map = {"davis": "DAVIS", "bindingdb": "BindingDB", "biosnap": "BioSNAP"}
        dataset_name = name_map.get(dataset_name.lower(), dataset_name)

        # 三路目录
        esm_base = _try_esm_fallback(Path(self.cache_dirs.esm2_dir).parent / dataset_name)
        if not esm_base.exists():
            # 如果传入已经是 .../esm/DAVIS，则 _try_esm_fallback 已经做了回退，这里再兜底一次
            esm_base = _try_esm_fallback(Path(self.cache_dirs.esm2_dir))

        mol_base = Path(self.cache_dirs.molclr_dir).parent / dataset_name
        chem_base = Path(self.cache_dirs.chemberta_dir).parent / dataset_name

        idx_esm  = _scan_npz_dir(esm_base)
        idx_mol  = _scan_npz_dir(mol_base)
        idx_chem = _scan_npz_dir(chem_base)

        # 打印一次
        tag = "train" if for_dataset == "train" else "test"
        print(f"[{tag.upper()}] cache roots:")
        print(f"  ESM : {esm_base}")
        print(f"  MolC: {mol_base}")
        print(f"  Chem: {chem_base}")
        print(f"[{tag.upper()}] index sizes: ESM2={len(idx_esm)} MoLCLR={len(idx_mol)} ChemBERTa={len(idx_chem)}")

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
