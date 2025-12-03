import hashlib
import os
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from torch.utils.data import Dataset, DataLoader


def sha1_24(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()[:24]


class DTIDataset(Dataset):
    """
    单折数据集：返回离线特征 + label。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        cache_root: str,
        dataset_name: str,
        mol_dir: str = "molclr",
        esm_dir: str = "esm2",
        chem_dir: str = "chemberta",
        graph_dir: str = "gvp",
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.indices = np.array(indices)
        self.cache_root = cache_root
        self.dataset_name = dataset_name
        self.mol_dir = mol_dir
        self.esm_dir = esm_dir
        self.chem_dir = chem_dir
        self.graph_dir = graph_dir

    def __len__(self) -> int:
        return len(self.indices)

    def _load_array(self, subdir: str, key: str) -> np.ndarray:
        """
        尝试按以下顺序加载：
        1) {key}.npz: 若是 NpzFile，优先取 'arr_0'，否则取第一个数组；
        2) {key}.npy。
        """
        base = os.path.join(self.cache_root, subdir, self.dataset_name, key)
        path_npz = base + ".npz"
        path_npy = base + ".npy"

        if os.path.exists(path_npz):
            data = np.load(path_npz, allow_pickle=False)
            # .npz: np.lib.npyio.NpzFile
            if isinstance(data, np.lib.npyio.NpzFile):
                if "arr_0" in data.files:
                    arr = data["arr_0"]
                else:
                    # 取第一个数组
                    arr = data[data.files[0]]
            else:
                # 意外情况：np.save 保存但扩展名是 .npz
                arr = data
            return arr

        if os.path.exists(path_npy):
            arr = np.load(path_npy, allow_pickle=False)
            return arr

        raise FileNotFoundError(f"Feature file not found for key={key} in {subdir}: "
                                f"tried {path_npz} and {path_npy}")

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[self.indices[idx]]
        smiles = str(row["smiles"])
        protein = str(row["protein"])
        label = float(row["label"])

        did = sha1_24(smiles)
        pid = sha1_24(protein)

        mol = self._load_array(self.mol_dir, did)  # (Ld, d_mol)
        esm = self._load_array(self.esm_dir, pid)  # (Lp, d_prot)
        chem = self._load_array(self.chem_dir, did)  # (d_chem,)
        graph = self._load_array(self.graph_dir, pid)  # (d_graph,)

        sample = {
            "drug_seq": mol.astype(np.float32),
            "prot_seq": esm.astype(np.float32),
            "chem": chem.astype(np.float32),
            "graph": graph.astype(np.float32),
            "label": label,  # ✅ 这里改成纯 float 就行了
        }
        return sample


def dti_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    将变长序列 pad 成批。
    """
    bs = len(batch)
    drug_lens = [b["drug_seq"].shape[0] for b in batch]
    prot_lens = [b["prot_seq"].shape[0] for b in batch]
    max_ld = max(drug_lens)
    max_lp = max(prot_lens)

    d_mol = batch[0]["drug_seq"].shape[1]
    d_prot = batch[0]["prot_seq"].shape[1]
    d_chem = batch[0]["chem"].shape[0]
    d_graph = batch[0]["graph"].shape[0]

    drug_seq = torch.zeros(bs, max_ld, d_mol, dtype=torch.float32)
    prot_seq = torch.zeros(bs, max_lp, d_prot, dtype=torch.float32)
    drug_mask = torch.zeros(bs, max_ld, dtype=torch.long)
    prot_mask = torch.zeros(bs, max_lp, dtype=torch.long)
    chem = torch.zeros(bs, d_chem, dtype=torch.float32)
    graph = torch.zeros(bs, d_graph, dtype=torch.float32)
    labels = torch.zeros(bs, dtype=torch.float32)

    for i, b in enumerate(batch):
        ld = b["drug_seq"].shape[0]
        lp = b["prot_seq"].shape[0]
        drug_seq[i, :ld] = torch.from_numpy(b["drug_seq"])
        prot_seq[i, :lp] = torch.from_numpy(b["prot_seq"])
        drug_mask[i, :ld] = 1
        prot_mask[i, :lp] = 1
        chem[i] = torch.from_numpy(b["chem"])
        graph[i] = torch.from_numpy(b["graph"])
        labels[i] = b["label"]

    return {
        "drug_seq": drug_seq,
        "prot_seq": prot_seq,
        "drug_mask": drug_mask,
        "prot_mask": prot_mask,
        "chem": chem,
        "graph": graph,
        "label": labels,
    }


class DTIDataModule:
    """
    负责：
      - 读取 all.csv
      - 生成 5 折划分 (train/val/test)
      - 返回各折 dataloader

    split_mode:
      - 'cold-protein': 按 pid 做 5 折 group KFold
      - 'cold-drug'   : 按 did 做 5 折 group KFold
      - 'cold-both'   : 简化 double-cold（样本级 group，用于近似）
      - 'warm' / 'random': 样本级 stratified KFold（药物和蛋白在各折之间是共享的）
    """

    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        cache_root: str,
        split_mode: str = "cold-protein",
        n_splits: int = 5,
        val_ratio: float = 0.15,
        batch_size: int = 64,
        num_workers: int = 4,
        seed: int = 42,
        pin_memory: bool = True,
    ):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.cache_root = cache_root
        self.split_mode = split_mode
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory

        self.df: Optional[pd.DataFrame] = None
        self.folds: List[Dict[str, np.ndarray]] = []

    @property
    def csv_path(self) -> str:
        return os.path.join(self.data_root, self.dataset_name, "all.csv")

    def load_dataframe(self):
        if self.df is not None:
            return
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset csv not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        if not {"smiles", "protein", "label"}.issubset(df.columns):
            raise ValueError("all.csv must contain columns: smiles, protein, label")

        df["smiles"] = df["smiles"].astype(str)
        df["protein"] = df["protein"].astype(str)
        df["label"] = df["label"].astype(float)

        df["did"] = df["smiles"].map(sha1_24)
        df["pid"] = df["protein"].map(sha1_24)
        self.df = df

    def setup(self):
        """
        生成 n_splits 折划分：每折包含 train_idx, val_idx, test_idx。
        """
        self.load_dataframe()
        df = self.df
        labels = df["label"].values
        pids = df["pid"].values
        dids = df["did"].values
        n = len(df)

        split_mode = self.split_mode.lower()
        folds: List[Dict[str, np.ndarray]] = []

        if split_mode in ["cold-protein", "cold-drug", "cold-both"]:
            if split_mode == "cold-protein":
                groups = pids
            elif split_mode == "cold-drug":
                groups = dids
            else:
                # 简化 double-cold：每个样本为一组，等价随机 KFold
                groups = np.arange(n)

            gkf = GroupKFold(n_splits=self.n_splits)
            outer = gkf.split(np.zeros(n), labels, groups)
            for fold_idx, (trainval_idx, test_idx) in enumerate(outer):
                trainval_idx = np.array(trainval_idx)
                test_idx = np.array(test_idx)
                groups_trainval = groups[trainval_idx]

                gss = GroupShuffleSplit(
                    n_splits=1,
                    test_size=self.val_ratio,
                    random_state=self.seed + fold_idx,
                )
                inner_train_idx_rel, val_idx_rel = next(
                    gss.split(trainval_idx, labels[trainval_idx], groups_trainval)
                )
                train_idx = trainval_idx[inner_train_idx_rel]
                val_idx = trainval_idx[val_idx_rel]

                folds.append(
                    {
                        "train_idx": train_idx,
                        "val_idx": val_idx,
                        "test_idx": test_idx,
                    }
                )

        elif split_mode in ["warm", "random"]:
            skf = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.seed
            )
            outer = skf.split(np.zeros(n), labels)
            for fold_idx, (trainval_idx, test_idx) in enumerate(outer):
                trainval_idx = np.array(trainval_idx)
                test_idx = np.array(test_idx)

                sss = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=self.val_ratio,
                    random_state=self.seed + fold_idx,
                )
                inner_train_idx_rel, val_idx_rel = next(
                    sss.split(trainval_idx, labels[trainval_idx])
                )
                train_idx = trainval_idx[inner_train_idx_rel]
                val_idx = trainval_idx[val_idx_rel]

                folds.append(
                    {
                        "train_idx": train_idx,
                        "val_idx": val_idx,
                        "test_idx": test_idx,
                    }
                )
        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode}")

        self.folds = folds

    def get_fold_indices(self, fold: int) -> Dict[str, np.ndarray]:
        if not self.folds:
            self.setup()
        if fold < 0 or fold >= len(self.folds):
            raise IndexError(f"Fold index {fold} out of range")
        return self.folds[fold]

    def make_dataloader(
        self,
        indices: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        dataset = DTIDataset(
            df=self.df,
            indices=indices,
            cache_root=self.cache_root,
            dataset_name=self.dataset_name,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=dti_collate_fn,
        )
        return loader

    def get_dataloaders(self, fold: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        indices = self.get_fold_indices(fold)
        train_loader = self.make_dataloader(indices["train_idx"], shuffle=True)
        val_loader = self.make_dataloader(indices["val_idx"], shuffle=False)
        test_loader = self.make_dataloader(indices["test_idx"], shuffle=False)
        return train_loader, val_loader, test_loader


def check_datasets_hit(dataset_name: str, data_root: str, cache_root: str):
    """
    在开始训练时只检查当前数据集：
      - all.csv 是否存在
      - 四种特征目录是否存在，以及其中的特征文件数量（.npy/.npz）
    """
    feat_dirs = ["esm2", "molclr", "chemberta", "gvp"]

    csv_path = os.path.join(data_root, dataset_name, "all.csv")
    csv_ok = os.path.exists(csv_path)

    print("==== Dataset & Cache Check ====")
    if csv_ok:
        print(f"[{dataset_name}] csv: HIT -> {csv_path}")
    else:
        print(f"[{dataset_name}] csv: MISS -> {csv_path}")

    for fd in feat_dirs:
        d = os.path.join(cache_root, fd, dataset_name)
        if os.path.isdir(d):
            n_files = len(
                [f for f in os.listdir(d) if f.endswith(".npy") or f.endswith(".npz")]
            )
            print(f"  {fd}: HIT  | files={n_files} | dir={d}")
        else:
            print(f"  {fd}: MISS | dir={d}")
    print("================================")
