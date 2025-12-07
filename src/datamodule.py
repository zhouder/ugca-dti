# datamodule.py
import os
import random
from dataclasses import dataclass
from hashlib import sha1
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha1_24(s: str) -> str:
    return sha1(s.encode("utf-8")).hexdigest()[:24]


def _first_array_from_npz(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    for k in npz.files:
        arr = np.array(npz[k])
        if arr.ndim >= 2:
            return arr
    return np.array(npz[npz.files[0]])


def load_feature(base: str) -> np.ndarray:
    npz_path = base + ".npz"
    npy_path = base + ".npy"
    if os.path.exists(npz_path):
        with np.load(npz_path) as f:
            arr = _first_array_from_npz(f)
    elif os.path.exists(npy_path):
        arr = np.load(npy_path)
    else:
        raise FileNotFoundError(f"Feature not found: {npz_path} / {npy_path}")
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr.astype(np.float32)


def _exists_feature(base: str) -> bool:
    return os.path.exists(base + ".npz") or os.path.exists(base + ".npy")


class DTIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dataset_name: str, cache_root: str):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.dataset = dataset_name
        self.cache_root = cache_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        smiles = str(row["smiles"])
        protein = str(row["protein"])
        label = float(row["label"])

        did = sha1_24(smiles)
        pid = sha1_24(protein)

        mol_base = os.path.join(self.cache_root, "molclr", self.dataset, did)
        prot_base = os.path.join(self.cache_root, "esm2", self.dataset, pid)

        mol = load_feature(mol_base)   # [Ld, d_mol]
        prot = load_feature(prot_base) # [Lp, d_prot]

        return {
            "mol": torch.from_numpy(mol),
            "prot": torch.from_numpy(prot),
            "label": torch.tensor([label], dtype=torch.float32),
        }


def dti_collate(batch: List[Dict]):
    mol_seqs = [b["mol"] for b in batch]
    prot_seqs = [b["prot"] for b in batch]
    labels = torch.cat([b["label"] for b in batch], dim=0)

    mol_padded = pad_sequence(mol_seqs, batch_first=True, padding_value=0.0)
    prot_padded = pad_sequence(prot_seqs, batch_first=True, padding_value=0.0)

    B, Ld_max, _ = mol_padded.shape
    _, Lp_max, _ = prot_padded.shape
    mask_d = torch.zeros((B, Ld_max), dtype=torch.int)
    mask_p = torch.zeros((B, Lp_max), dtype=torch.int)
    for i, (md, pp) in enumerate(zip(mol_seqs, prot_seqs)):
        mask_d[i, : md.shape[0]] = 1
        mask_p[i, : pp.shape[0]] = 1

    return {"mol": mol_padded, "prot": prot_padded, "mask_d": mask_d, "mask_p": mask_p, "label": labels}


@dataclass
class FoldIndex:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class DTIDataModule:
    """
    读取 <data_root>/<dataset>/all.csv：列为 smiles, protein, label
    构建 5 折：
      - warm/random: KFold on samples
      - cold_drug:   KFold on unique smiles
      - cold_protein:KFold on unique protein
      - cold_both:   drug/protein 同时冷（不足则退化为 OR）
    """
    def __init__(
        self,
        data_root: str,
        cache_root: str,
        dataset: str,
        split_mode: str = "warm",
        batch_size: int = 64,
        num_workers: int = 4,
        seed: int = 42,
    ):
        self.data_root = data_root
        self.cache_root = cache_root
        self.dataset = dataset
        self.split_mode = split_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        csv_path = os.path.join(self.data_root, dataset, "all.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        for col in ["smiles", "protein", "label"]:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' missing in all.csv")

        set_global_seed(seed)
        self.folds = self._build_folds()

    def compute_cache_hit_rate(self) -> dict:
        total = len(self.df)
        miss_mol = miss_prot = miss_both = hit_both = 0
        for _, row in self.df.iterrows():
            did = sha1_24(str(row["smiles"]))
            pid = sha1_24(str(row["protein"]))
            mol_base = os.path.join(self.cache_root, "molclr", self.dataset, did)
            prot_base = os.path.join(self.cache_root, "esm2", self.dataset, pid)
            has_mol = _exists_feature(mol_base)
            has_prot = _exists_feature(prot_base)
            if has_mol and has_prot:
                hit_both += 1
            elif (not has_mol) and (not has_prot):
                miss_both += 1
            elif not has_mol:
                miss_mol += 1
            else:
                miss_prot += 1
        pos_ratio = float(self.df["label"].astype(float).mean())
        return {
            "total": int(total),
            "hit_both": int(hit_both),
            "miss_mol": int(miss_mol),
            "miss_prot": int(miss_prot),
            "miss_both": int(miss_both),
            "pos_ratio": pos_ratio,
            "hit_rate": (hit_both / total) if total > 0 else 0.0,
        }

    def _build_folds(self) -> List[FoldIndex]:
        n_splits = 5
        rng = np.random.RandomState(self.seed)

        def split_train_val(indices: np.ndarray, val_ratio: float = 0.125) -> Tuple[np.ndarray, np.ndarray]:
            n = len(indices)
            n_val = max(1, int(round(n * val_ratio)))
            perm = rng.permutation(indices)
            return perm[n_val:], perm[:n_val]

        folds: List[FoldIndex] = []
        if self.split_mode in ["warm", "random"]:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            for train, test in kf.split(self.df):
                tr, va = split_train_val(np.array(train))
                folds.append(FoldIndex(train_idx=tr, val_idx=va, test_idx=np.array(test)))
            return folds

        if self.split_mode == "cold_drug":
            uniques = self.df["smiles"].astype(str).drop_duplicates().values
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            for _, test_u_idx in kf.split(uniques):
                test_smiles = set(uniques[test_u_idx])
                mask_test = self.df["smiles"].astype(str).isin(test_smiles).values
                test_idx = np.where(mask_test)[0]
                warm_idx = np.where(~mask_test)[0]
                tr, va = split_train_val(warm_idx)
                folds.append(FoldIndex(train_idx=tr, val_idx=va, test_idx=test_idx))
            return folds

        if self.split_mode == "cold_protein":
            uniques = self.df["protein"].astype(str).drop_duplicates().values
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            for _, test_u_idx in kf.split(uniques):
                test_prot = set(uniques[test_u_idx])
                mask_test = self.df["protein"].astype(str).isin(test_prot).values
                test_idx = np.where(mask_test)[0]
                warm_idx = np.where(~mask_test)[0]
                tr, va = split_train_val(warm_idx)
                folds.append(FoldIndex(train_idx=tr, val_idx=va, test_idx=test_idx))
            return folds

        if self.split_mode == "cold_both":
            uniq_d = self.df["smiles"].astype(str).drop_duplicates().values
            uniq_p = self.df["protein"].astype(str).drop_duplicates().values
            kf_d = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            kf_p = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed + 1)
            for (_, test_d_idx), (_, test_p_idx) in zip(kf_d.split(uniq_d), kf_p.split(uniq_p)):
                test_drugs = set(uniq_d[test_d_idx])
                test_prots = set(uniq_p[test_p_idx])
                mask_test = self.df["smiles"].astype(str).isin(test_drugs) & self.df["protein"].astype(str).isin(test_prots)
                test_idx = np.where(mask_test.values)[0]
                if len(test_idx) < 1:  # 保障非空
                    mask_test = self.df["smiles"].astype(str).isin(test_drugs) | self.df["protein"].astype(str).isin(test_prots)
                    test_idx = np.where(mask_test.values)[0]
                warm_idx = np.where(~mask_test.values)[0]
                tr, va = split_train_val(warm_idx)
                folds.append(FoldIndex(train_idx=tr, val_idx=va, test_idx=test_idx))
            return folds

        raise ValueError(f"Unknown split_mode: {self.split_mode}")

    def get_input_dims(self) -> Tuple[int, int]:
        row = self.df.iloc[0]
        did = sha1_24(str(row["smiles"]))
        pid = sha1_24(str(row["protein"]))
        mol = load_feature(os.path.join(self.cache_root, "molclr", self.dataset, did))
        prot = load_feature(os.path.join(self.cache_root, "esm2", self.dataset, pid))
        return int(mol.shape[1]), int(prot.shape[1])

    def get_loaders_for_fold(self, fold_idx: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        fi = self.folds[fold_idx]
        ds_train = DTIDataset(self.df.iloc[fi.train_idx], self.dataset, self.cache_root)
        ds_val = DTIDataset(self.df.iloc[fi.val_idx], self.dataset, self.cache_root)
        ds_test = DTIDataset(self.df.iloc[fi.test_idx], self.dataset, self.cache_root)

        train_loader = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True, collate_fn=dti_collate)
        val_loader = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True, collate_fn=dti_collate)
        test_loader = DataLoader(ds_test, batch_size=self.batch_size, shuffle=False,
                                 num_workers=self.num_workers, pin_memory=True, collate_fn=dti_collate)
        return train_loader, val_loader, test_loader
