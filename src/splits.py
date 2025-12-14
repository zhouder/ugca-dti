# src/splits.py
# -*- coding: utf-8 -*-
"""
Splitting utilities for DTI.
This file provides two functions used by train.py:

1) make_outer_splits(split_mode, n_splits, seed, drug_key, prot_key, y)
   -> List[Tuple[np.ndarray(train_idx), np.ndarray(test_idx)]]

2) sample_val_indices(split_mode, train_idx, val_frac, seed, drug_key, prot_key, y)
   -> (train_sub_idx, val_idx)  both are np.ndarray of indices

Supported split_mode:
  - "warm" / "hot"         : standard KFold / StratifiedKFold (binary)
  - "cold-drug"            : drug cold-start (no drug overlap between train/test)
  - "cold-protein"         : protein cold-start
  - "cold-pair"            : cold pair split by (drug, protein) pairs
  - "cold-both"            : both cold-start (prefer intersection of drug-fold & protein-fold; fallback to union if too small)

Notes:
- We use deterministic shuffling based on seed.
- For cold-* splits, we implement our own group-to-fold assignment (no sklearn required).
- For warm/hot we try to use sklearn if installed; otherwise fallback to our own KFold.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Sequence, Set
import numpy as np


# -----------------------------
# Helpers
# -----------------------------
def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(int(seed) & 0x7FFFFFFF)


def _to_np(a) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    return np.asarray(a)


def _is_binary_classification(y: np.ndarray) -> bool:
    """
    判断是否二分类：值域为 {0,1} 或恰好两类且都接近整数 0/1
    """
    if y.size == 0:
        return False
    uniq = np.unique(y)
    if uniq.size != 2:
        return False
    # allow float but only 0/1
    return np.all(np.isin(uniq, [0, 1]))


def _unique_groups_in_idx(keys: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.unique(keys[idx])


def _assign_groups_to_folds(groups: np.ndarray, n_splits: int, seed: int) -> List[np.ndarray]:
    """
    将 unique groups 随机打散后平均分到 n_splits 个 fold。
    返回：fold_groups[k] = groups assigned to fold k (np.ndarray)
    """
    g = np.array(groups, dtype=object)
    r = _rng(seed)
    perm = r.permutation(len(g))
    g = g[perm]

    folds: List[np.ndarray] = []
    # 均匀分块
    sizes = [len(g) // n_splits] * n_splits
    for i in range(len(g) % n_splits):
        sizes[i] += 1

    start = 0
    for sz in sizes:
        folds.append(g[start:start + sz])
        start += sz
    return folds


def _indices_by_group(keys: np.ndarray, group_set: Set[object]) -> np.ndarray:
    """
    返回 keys 属于 group_set 的所有样本索引
    """
    mask = np.isin(keys, list(group_set))
    return np.where(mask)[0]


def _pair_keys(drug_key: np.ndarray, prot_key: np.ndarray) -> np.ndarray:
    # 用字符串拼接构造 pair id（足够稳定）
    return (drug_key.astype(str) + "||" + prot_key.astype(str)).astype(object)


def _kfold_indices(n: int, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    不依赖 sklearn 的 KFold(shuffle=True) 简单实现。
    """
    r = _rng(seed)
    idx = r.permutation(n)
    folds = np.array_split(idx, n_splits)
    out = []
    all_idx = np.arange(n)
    for k in range(n_splits):
        test = folds[k]
        test_set = set(test.tolist())
        train = np.array([i for i in idx if i not in test_set], dtype=np.int64)
        out.append((train, test.astype(np.int64)))
    return out


def _stratified_kfold_indices(y: np.ndarray, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    简单分层 KFold（仅支持二分类 0/1），不依赖 sklearn。
    """
    r = _rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    r.shuffle(idx0)
    r.shuffle(idx1)
    folds0 = np.array_split(idx0, n_splits)
    folds1 = np.array_split(idx1, n_splits)

    out = []
    for k in range(n_splits):
        test = np.concatenate([folds0[k], folds1[k]], axis=0)
        r.shuffle(test)
        test_set = set(test.tolist())
        # train = all - test
        train = np.array([i for i in range(len(y)) if i not in test_set], dtype=np.int64)
        out.append((train, test.astype(np.int64)))
    return out


# -----------------------------
# Public API
# -----------------------------
def make_outer_splits(
    split_mode: str,
    n_splits: int,
    seed: int,
    drug_key: Sequence,
    prot_key: Sequence,
    y: Sequence,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return list of (train_idx, test_idx) for CV outer splits.
    """
    split_mode = str(split_mode).strip().lower()
    drug_key = _to_np(drug_key)
    prot_key = _to_np(prot_key)
    y = _to_np(y).astype(np.float32, copy=False)

    n = len(y)
    assert len(drug_key) == n and len(prot_key) == n, "drug_key/prot_key/y 长度必须一致"

    # warm/hot
    if split_mode in ("warm", "hot"):
        # 优先使用 sklearn
        try:
            from sklearn.model_selection import StratifiedKFold, KFold
            if _is_binary_classification(y):
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                return [(tr.astype(np.int64), te.astype(np.int64)) for tr, te in skf.split(np.zeros(n), y)]
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return [(tr.astype(np.int64), te.astype(np.int64)) for tr, te in kf.split(np.zeros(n))]
        except Exception:
            # fallback
            if _is_binary_classification(y):
                return _stratified_kfold_indices(y, n_splits, seed)
            return _kfold_indices(n, n_splits, seed)

    # cold-drug
    if split_mode == "cold-drug":
        drugs = np.unique(drug_key)
        folds = _assign_groups_to_folds(drugs, n_splits, seed)
        out = []
        all_idx = np.arange(n, dtype=np.int64)
        for k in range(n_splits):
            test_drugs = set(folds[k].tolist())
            test_idx = np.where(np.isin(drug_key, list(test_drugs)))[0].astype(np.int64)
            test_set = set(test_idx.tolist())
            train_idx = np.array([i for i in all_idx if i not in test_set], dtype=np.int64)
            out.append((train_idx, test_idx))
        return out

    # cold-protein
    if split_mode == "cold-protein":
        prots = np.unique(prot_key)
        folds = _assign_groups_to_folds(prots, n_splits, seed)
        out = []
        all_idx = np.arange(n, dtype=np.int64)
        for k in range(n_splits):
            test_prots = set(folds[k].tolist())
            test_idx = np.where(np.isin(prot_key, list(test_prots)))[0].astype(np.int64)
            test_set = set(test_idx.tolist())
            train_idx = np.array([i for i in all_idx if i not in test_set], dtype=np.int64)
            out.append((train_idx, test_idx))
        return out

    # cold-pair
    if split_mode == "cold-pair":
        pair_key = _pair_keys(drug_key, prot_key)
        pairs = np.unique(pair_key)
        folds = _assign_groups_to_folds(pairs, n_splits, seed)
        out = []
        all_idx = np.arange(n, dtype=np.int64)
        for k in range(n_splits):
            test_pairs = set(folds[k].tolist())
            test_idx = np.where(np.isin(pair_key, list(test_pairs)))[0].astype(np.int64)
            test_set = set(test_idx.tolist())
            train_idx = np.array([i for i in all_idx if i not in test_set], dtype=np.int64)
            out.append((train_idx, test_idx))
        return out

    # cold-both
    if split_mode == "cold-both":
        drugs = np.unique(drug_key)
        prots = np.unique(prot_key)
        drug_folds = _assign_groups_to_folds(drugs, n_splits, seed)
        prot_folds = _assign_groups_to_folds(prots, n_splits, seed + 9973)

        out = []
        all_idx = np.arange(n, dtype=np.int64)

        for k in range(n_splits):
            test_drugs = set(drug_folds[k].tolist())
            test_prots = set(prot_folds[k].tolist())

            mask_d = np.isin(drug_key, list(test_drugs))
            mask_p = np.isin(prot_key, list(test_prots))

            inter_idx = np.where(mask_d & mask_p)[0].astype(np.int64)
            union_idx = np.where(mask_d | mask_p)[0].astype(np.int64)

            # 优先用交集（真正 both-cold），如果太小则用并集兜底
            # 这里阈值可以按你需要调；默认：交集至少占 union 的 30% 且非空
            if inter_idx.size > 0 and inter_idx.size >= max(10, int(0.3 * union_idx.size)):
                test_idx = inter_idx
            else:
                test_idx = union_idx

            test_set = set(test_idx.tolist())
            train_idx = np.array([i for i in all_idx if i not in test_set], dtype=np.int64)
            out.append((train_idx, test_idx))

        return out

    raise ValueError(f"Unknown split_mode: {split_mode}")


def sample_val_indices(
    split_mode: str,
    train_idx: np.ndarray,
    val_frac: float,
    seed: int,
    drug_key: Sequence,
    prot_key: Sequence,
    y: Sequence,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given outer-train indices, split into (train_sub, val) according to split_mode.

    Returns:
      train_sub_idx, val_idx  (np.ndarray int64)
    """
    split_mode = str(split_mode).strip().lower()
    train_idx = _to_np(train_idx).astype(np.int64)
    drug_key = _to_np(drug_key)
    prot_key = _to_np(prot_key)
    y = _to_np(y).astype(np.float32, copy=False)

    n_tr = train_idx.size
    if n_tr == 0:
        return train_idx, train_idx

    # val size by samples (approx)
    val_frac = float(val_frac)
    val_target = max(1, int(round(val_frac * n_tr)))

    r = _rng(seed)

    # warm/hot: stratified if binary else random
    if split_mode in ("warm", "hot"):
        # try sklearn StratifiedShuffleSplit
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            if _is_binary_classification(y):
                sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
                tr_mask = np.zeros(len(y), dtype=bool)
                tr_mask[train_idx] = True
                idx_in = np.where(tr_mask)[0]
                y_in = y[idx_in]
                tr_sub_in, va_in = next(sss.split(np.zeros_like(y_in), y_in))
                train_sub = idx_in[tr_sub_in].astype(np.int64)
                val_idx = idx_in[va_in].astype(np.int64)
                return train_sub, val_idx
        except Exception:
            pass

        # fallback
        idx = train_idx.copy()
        r.shuffle(idx)
        val_idx = idx[:val_target]
        train_sub = idx[val_target:]
        return train_sub.astype(np.int64), val_idx.astype(np.int64)

    # cold-drug
    if split_mode == "cold-drug":
        drugs = _unique_groups_in_idx(drug_key, train_idx)
        r.shuffle(drugs)
        # 选一部分 drugs 给 val
        n_val_drugs = max(1, int(round(val_frac * len(drugs))))
        val_drugs = set(drugs[:n_val_drugs].tolist())
        val_idx = train_idx[np.isin(drug_key[train_idx], list(val_drugs))]
        train_sub = train_idx[~np.isin(drug_key[train_idx], list(val_drugs))]
        # 保证 val 非空
        if val_idx.size == 0:
            idx = train_idx.copy()
            r.shuffle(idx)
            val_idx = idx[:val_target]
            train_sub = idx[val_target:]
        return train_sub.astype(np.int64), val_idx.astype(np.int64)

    # cold-protein
    if split_mode == "cold-protein":
        prots = _unique_groups_in_idx(prot_key, train_idx)
        r.shuffle(prots)
        n_val_prots = max(1, int(round(val_frac * len(prots))))
        val_prots = set(prots[:n_val_prots].tolist())
        val_idx = train_idx[np.isin(prot_key[train_idx], list(val_prots))]
        train_sub = train_idx[~np.isin(prot_key[train_idx], list(val_prots))]
        if val_idx.size == 0:
            idx = train_idx.copy()
            r.shuffle(idx)
            val_idx = idx[:val_target]
            train_sub = idx[val_target:]
        return train_sub.astype(np.int64), val_idx.astype(np.int64)

    # cold-pair
    if split_mode == "cold-pair":
        pair_key = _pair_keys(drug_key, prot_key)
        pairs = _unique_groups_in_idx(pair_key, train_idx)
        r.shuffle(pairs)
        n_val_pairs = max(1, int(round(val_frac * len(pairs))))
        val_pairs = set(pairs[:n_val_pairs].tolist())
        val_idx = train_idx[np.isin(pair_key[train_idx], list(val_pairs))]
        train_sub = train_idx[~np.isin(pair_key[train_idx], list(val_pairs))]
        if val_idx.size == 0:
            idx = train_idx.copy()
            r.shuffle(idx)
            val_idx = idx[:val_target]
            train_sub = idx[val_target:]
        return train_sub.astype(np.int64), val_idx.astype(np.int64)

    # cold-both
    if split_mode == "cold-both":
        drugs = _unique_groups_in_idx(drug_key, train_idx)
        prots = _unique_groups_in_idx(prot_key, train_idx)
        r.shuffle(drugs)
        r2 = _rng(seed + 9973)
        r2.shuffle(prots)

        n_val_drugs = max(1, int(round(val_frac * len(drugs))))
        n_val_prots = max(1, int(round(val_frac * len(prots))))
        val_drugs = set(drugs[:n_val_drugs].tolist())
        val_prots = set(prots[:n_val_prots].tolist())

        mask_d = np.isin(drug_key[train_idx], list(val_drugs))
        mask_p = np.isin(prot_key[train_idx], list(val_prots))

        inter = train_idx[mask_d & mask_p]
        union = train_idx[mask_d | mask_p]

        # 优先交集（both-cold），太小就用 union
        if inter.size > 0 and inter.size >= max(10, int(0.3 * union.size)):
            val_idx = inter
        else:
            val_idx = union

        train_sub = np.setdiff1d(train_idx, val_idx, assume_unique=False)
        if val_idx.size == 0:
            idx = train_idx.copy()
            r.shuffle(idx)
            val_idx = idx[:val_target]
            train_sub = idx[val_target:]
        return train_sub.astype(np.int64), val_idx.astype(np.int64)

    raise ValueError(f"Unknown split_mode: {split_mode}")
