# -*- coding: utf-8 -*-
"""
Split utilities for UGCA-DTI
支持以下划分模式：
- warm/hot：随机/分层K折
- cold-protein：按蛋白分组K折（组不交叠）
- cold-drug：按药物分组K折（组不交叠）
- cold-pair：按(药物,蛋白)对分组K折（对不交叠）
- cold-both：双冷；test 为(测试药物∩测试蛋白)的交集；train/val 去除任何含测试实体的样本；val 也按实体抽

返回：
- make_outer_splits(...) -> List[np.ndarray]  每折测试集索引
- sample_val_indices(...) -> (tr_idx, va_idx)  从候选池按模式抽取验证集
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, StratifiedShuffleSplit


def _chunk_groups(groups: np.ndarray, n_splits: int, seed: int) -> List[np.ndarray]:
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    return [uniq[k::n_splits] for k in range(n_splits)]


def make_outer_splits(mode: str, cv_folds: int, seed: int,
                      drug_key: np.ndarray, prot_key: np.ndarray, labels: np.ndarray) -> List[np.ndarray]:
    N = len(labels)
    all_idx = np.arange(N)

    if mode == "cold-protein":
        gkf = GroupKFold(n_splits=cv_folds)
        return [te for _, te in gkf.split(all_idx, groups=prot_key)]

    if mode == "cold-drug":
        gkf = GroupKFold(n_splits=cv_folds)
        return [te for _, te in gkf.split(all_idx, groups=drug_key)]

    if mode == "cold-pair":
        pair_key = np.array([f"{d}||{p}" for d, p in zip(drug_key, prot_key)], dtype=object)
        gkf = GroupKFold(n_splits=cv_folds)
        return [te for _, te in gkf.split(all_idx, groups=pair_key)]

    if mode in ("warm", "hot"):
        if len(np.unique(labels)) == 2:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            return [te for _, te in skf.split(all_idx, labels)]
        else:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            return [te for _, te in kf.split(all_idx)]

    if mode == "cold-both":
        drug_folds = _chunk_groups(drug_key, cv_folds, seed)
        prot_folds = _chunk_groups(prot_key,  cv_folds, seed+7)
        splits: List[np.ndarray] = []
        for k in range(cv_folds):
            td = set(drug_folds[k]); tp = set(prot_folds[k])
            test_mask = np.array([(d in td) and (p in tp) for d, p in zip(drug_key, prot_key)], dtype=bool)
            te_idx = np.where(test_mask)[0]
            if len(te_idx) < max(1, int(0.05 * N / cv_folds)):
                test_mask = np.array([(d in td) or (p in tp) for d, p in zip(drug_key, prot_key)], dtype=bool)
                te_idx = np.where(test_mask)[0]
            splits.append(te_idx)
        return splits

    raise ValueError(f"Unknown split mode: {mode}")


def sample_val_indices(mode: str, pool_idx: np.ndarray, val_frac_in_pool: float, seed: int,
                       drug_key: np.ndarray, prot_key: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pool_idx = np.asarray(pool_idx)
    if mode in ("warm", "hot"):
        if len(np.unique(labels)) == 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_in_pool, random_state=seed)
            tr_sub, va_sub = next(sss.split(pool_idx, labels[pool_idx]))
            return pool_idx[tr_sub], pool_idx[va_sub]
        else:
            n_va = max(1, int(round(val_frac_in_pool * len(pool_idx))))
            rng = np.random.default_rng(seed); rng.shuffle(pool_idx)
            return pool_idx[n_va:], pool_idx[:n_va]

    if mode == "cold-protein":
        g = prot_key[pool_idx]
        uniq = np.unique(g); rng = np.random.default_rng(seed); rng.shuffle(uniq)
        n_val = max(1, int(round(val_frac_in_pool * len(uniq))))
        va_groups = set(uniq[:n_val])
        mask = np.array([x in va_groups for x in g], dtype=bool)
        return pool_idx[~mask], pool_idx[mask]

    if mode == "cold-drug":
        g = drug_key[pool_idx]
        uniq = np.unique(g); rng = np.random.default_rng(seed); rng.shuffle(uniq)
        n_val = max(1, int(round(val_frac_in_pool * len(uniq))))
        va_groups = set(uniq[:n_val])
        mask = np.array([x in va_groups for x in g], dtype=bool)
        return pool_idx[~mask], pool_idx[mask]

    if mode == "cold-pair":
        pair = np.array([f"{d}||{p}" for d,p in zip(drug_key, prot_key)], dtype=object)[pool_idx]
        uniq = np.unique(pair); rng = np.random.default_rng(seed); rng.shuffle(uniq)
        n_val = max(1, int(round(val_frac_in_pool * len(uniq))))
        va_groups = set(uniq[:n_val])
        mask = np.array([x in va_groups for x in pair], dtype=bool)
        return pool_idx[~mask], pool_idx[mask]

    if mode == "cold-both":
        rng = np.random.default_rng(seed)
        d_pool = np.unique(drug_key[pool_idx]); rng.shuffle(d_pool)
        p_pool = np.unique(prot_key[pool_idx]); rng.shuffle(p_pool)
        nd = max(1, int(round(val_frac_in_pool * len(d_pool))))
        np_ = max(1, int(round(val_frac_in_pool * len(p_pool))))
        val_drugs = set(d_pool[:nd]); val_prots = set(p_pool[:np_])
        va_mask = np.array([(d in val_drugs) and (p in val_prots) for d,p in zip(drug_key[pool_idx], prot_key[pool_idx])], dtype=bool)
        va_idx = pool_idx[va_mask]
        tr_mask = np.array([(d not in val_drugs) and (p not in val_prots) for d,p in zip(drug_key[pool_idx], prot_key[pool_idx])], dtype=bool)
        tr_idx = pool_idx[tr_mask]
        if len(va_idx) == 0 or len(tr_idx) == 0:
            return sample_val_indices("cold-pair", pool_idx, val_frac_in_pool, seed, drug_key, prot_key, labels)
        return tr_idx, va_idx

    raise ValueError(f"unknown mode {mode}")
