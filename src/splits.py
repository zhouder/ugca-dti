import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

def generate_ids(df):

    import hashlib

    if 'smiles' not in df.columns or 'seq' not in df.columns:
        raise ValueError("CSV must contain 'smiles' and 'seq'")

    def get_hash(text):
        if pd.isna(text) or text == '': return 'unknown'
        return hashlib.sha1(str(text).encode('utf-8')).hexdigest()[:24]

    df = df.copy()
    df['did'] = df['smiles'].apply(get_hash)
    df['pid'] = df['seq'].apply(get_hash)
    return df

def _assert_no_overlap(train_idx, test_idx, groups, mode_name, fold_num):

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    intersection = train_groups.intersection(test_groups)

    if len(intersection) > 0:
        raise AssertionError(
            f"[FATAL] Split Leakage detected in {mode_name} (Fold {fold_num})! "
            f"Found {len(intersection)} overlapping groups (e.g., {list(intersection)[:3]}). "
            "This violates the cold-start constraint."
        )

def get_kfold_indices(df, mode='warm', n_splits=5, seed=42):

    indices = np.arange(len(df))

    drug_key = df['smiles'].astype(str).values
    prot_key = df['seq'].astype(str).values
    y = df['label'].values

    outer_folds = []

    if mode == 'warm':

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        outer_folds = list(kf.split(indices, y))

    elif mode == 'cold-drug':

        gkf = GroupKFold(n_splits=n_splits)
        outer_folds = list(gkf.split(indices, groups=drug_key))

    elif mode == 'cold-protein':

        gkf = GroupKFold(n_splits=n_splits)
        outer_folds = list(gkf.split(indices, groups=prot_key))

    else:
        raise ValueError(f"Unknown mode: {mode}. Supported: warm, cold-drug, cold-protein")

    final_splits = []

    for fold_i, (temp_train_indices, test_indices) in enumerate(outer_folds):
        baseline_fold_num = fold_i + 1
        temp_df = df.iloc[temp_train_indices]

        if mode == 'cold-drug':
            _assert_no_overlap(temp_train_indices, test_indices, drug_key, "Outer Cold-Drug", baseline_fold_num)
        elif mode == 'cold-protein':
            _assert_no_overlap(temp_train_indices, test_indices, prot_key, "Outer Cold-Protein", baseline_fold_num)

        inner_groups = None
        if mode == 'cold-drug': inner_groups = temp_df['smiles'].values
        elif mode == 'cold-protein': inner_groups = temp_df['seq'].values

        if inner_groups is not None:
            unique_inner = np.unique(inner_groups)

            offset = 2000 if mode == 'cold-protein' else 3000
            magic_seed = seed + offset + baseline_fold_num
            rng = np.random.RandomState(magic_seed)

            rng.shuffle(unique_inner)

            val_frac = 0.1 / (1.0 - 1.0/n_splits)
            n_val = max(1, int(round(val_frac * len(unique_inner))))
            val_groups = set(unique_inner[:n_val])

            is_val = np.array([g in val_groups for g in inner_groups])
            train_idx = temp_train_indices[~is_val]
            val_idx = temp_train_indices[is_val]

            check_key = drug_key if mode == 'cold-drug' else prot_key
            _assert_no_overlap(train_idx, val_idx, check_key, f"Inner {mode}", baseline_fold_num)

        else:

            magic_seed = seed + 1000 + baseline_fold_num

            kf_inner = KFold(n_splits=8, shuffle=True, random_state=magic_seed)
            train_sub, val_sub = next(kf_inner.split(temp_train_indices))
            train_idx = temp_train_indices[train_sub]
            val_idx = temp_train_indices[val_sub]

        final_splits.append((train_idx, val_idx, test_indices))

    return final_splits
