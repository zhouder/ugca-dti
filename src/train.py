# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, argparse, math, csv, importlib, inspect
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ====== 数据模块（你仓库里的 datamodule）======
from src.datamodule import DataModule, DMConfig  # 若路径不同，按你的实际改

# ====== 自适应加载模型类：src/model.py 是一个文件，不是包 ======
def load_model_class():
    """
    优先取 src.model.UGCA；若没有，则从 src/model.py 中挑第一个 nn.Module 子类。
    """
    mod = importlib.import_module("src.model")  # 对应 src/model.py
    UGCA = getattr(mod, "UGCA", None)
    if UGCA is not None and inspect.isclass(UGCA) and issubclass(UGCA, nn.Module):
        return UGCA, "UGCA"

    # 自动发现
    candidates = []
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ == mod.__name__ and issubclass(obj, nn.Module):
            candidates.append((name, obj))
    if not candidates:
        raise RuntimeError("在 src/model.py 内没有发现任何 nn.Module 子类，请检查模型定义。")
    print(f"[INFO] 未找到类 UGCA，改用 src/model.py 中的模型类：{candidates[0][0]}")
    return candidates[0][1], candidates[0][0]

ModelClass, _MODEL_NAME = load_model_class()

# ====== 度量 ======
try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        accuracy_score, recall_score, matthews_corrcoef,
    )
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False
    print("[WARN] 未安装 scikit-learn，将使用简化指标（无 AUROC/AUPRC/MCC 计算）。")

def binary_metrics(prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    res: Dict[str, float] = {}
    pred = (prob >= threshold).astype(np.int64)
    # ACC
    res["acc"] = float(accuracy_score(y_true, pred)) if SKLEARN_OK else float((pred == y_true).mean())
    # SEN
    if SKLEARN_OK:
        res["sen"] = float(recall_score(y_true, pred))
    else:
        res["sen"] = float((pred[y_true == 1] == 1).mean()) if (y_true == 1).any() else float("nan")
    # F1
    res["f1"] = float(f1_score(y_true, pred)) if SKLEARN_OK else float("nan")
    # MCC
    res["mcc"] = float(matthews_corrcoef(y_true, pred)) if SKLEARN_OK else float("nan")
    # AUROC / AUPRC
    try:
        res["auroc"] = float(roc_auc_score(y_true, prob)) if SKLEARN_OK else float("nan")
    except Exception:
        res["auroc"] = float("nan")
    try:
        res["auprc"] = float(average_precision_score(y_true, prob)) if SKLEARN_OK else float("nan")
    except Exception:
        res["auprc"] = float("nan")
    return res

def fmt(m: Dict[str, float]) -> str:
    return (f"AUROC {m['auroc']:.4f} AUPRC {m['auprc']:.4f} F1 {m['f1']:.4f} "
            f"ACC {m['acc']:.4f} SEN {m['sen']:.4f} MCC {m['mcc']:.4f}")

# ====== 一次 epoch ======
def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, train: bool,
              optimizer: optim.Optimizer | None) -> Tuple[float, np.ndarray, np.ndarray, float]:
    t0 = time.time()
    model.train(train)
    total_loss = 0.0
    all_prob, all_lbl = [], []

    for step, batch in enumerate(loader):
        # (protein_feat, drug_feat1, drug_feat2), label
        (v_p, v_d1, v_d2), y = batch
        v_p  = v_p.to(device, non_blocking=True)
        v_d1 = v_d1.to(device, non_blocking=True)
        v_d2 = v_d2.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        if step == 0:
            try:
                model_dev = next(model.parameters()).device
            except StopIteration:
                model_dev = torch.device("cpu")
            print(f"[DEBUG] devices | model={model_dev} | v_p={v_p.device} | v_d1={v_d1.device} | v_d2={v_d2.device} | y={y.device}")

        logits = model(v_p, v_d1, v_d2)  # [B]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float())

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        all_prob.append(prob)
        all_lbl.append(y.detach().cpu().numpy())

    probs = np.concatenate(all_prob, axis=0)
    labels = np.concatenate(all_lbl, axis=0)
    return total_loss / max(1, len(loader)), probs, labels, time.time() - t0

# ====== 保存/加载 ======
def save_ckpt(path: Path, model: nn.Module, epoch: int, metrics: Dict[str, float], optimizer_state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "state_dict": model.state_dict(),
         "metrics": metrics, "optimizer": optimizer_state},
        str(path),
    )

# ====== 单折 ======
def run_one_fold(args, dataset: str, fold: int, device: torch.device,
                 out_root: Path, data_root: Path, dataset_dirname: str) -> Dict[str, float]:
    # 数据
    dm = DataModule(DMConfig(
        dataset=dataset, data_root=str(data_root),
        dataset_dirname=dataset_dirname, fold=fold,
    ))
    train_loader = dm.train_loader(batch_size=args.batch_size, num_workers=args.workers,
                                   pin_memory=True, persistent_workers=(args.workers > 0))
    test_loader  = dm.test_loader (batch_size=args.batch_size, num_workers=args.workers,
                                   pin_memory=True, persistent_workers=(args.workers > 0))

    N = len(train_loader.dataset)
    print(f"[INFO] train size = {N} | batch_size = {args.batch_size}")

    # 模型 / 优化器
    model = ModelClass(dm.dims).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 输出
    fold_dir = out_root / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    csv_path = fold_dir / "metrics.csv"

    # 断点
    start_epoch = 1
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[RESUME] loaded {args.resume} -> start from epoch {start_epoch}")

    # CSV 头
    if not csv_path.exists() or start_epoch == 1:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_loss","test_loss","AUROC","AUPRC","F1","ACC","SEN","MCC","time_train_s","time_test_s"])

    # 以 AUROC 作为选优指标（若 AUROC 为 NaN，则回退到 ACC）
    def score_for_best(m: Dict[str, float]) -> float:
        return m["auroc"] if not math.isnan(m["auroc"]) else m["acc"]

    best_score = -1.0
    best_row: Dict[str, float] = {}

    pbar = tqdm(range(start_epoch, args.epochs + 1), ncols=120, desc=f"[{dataset}] fold{fold} epochs")
    for epoch in pbar:
        train_loss, _, _, tr_t = run_epoch(model, train_loader, device, train=True,  optimizer=optimizer)
        test_loss,  prob, y, te_t = run_epoch(model, test_loader,  device, train=False, optimizer=None)
        m = binary_metrics(prob, y)

        # 写一行
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_loss:.6f}", f"{test_loss:.6f}",
                        f"{m['auroc']:.6f}", f"{m['auprc']:.6f}", f"{m['f1']:.6f}",
                        f"{m['acc']:.6f}", f"{m['sen']:.6f}", f"{m['mcc']:.6f}",
                        f"{tr_t:.1f}", f"{te_t:.1f}"])

        pbar.set_postfix_str(f"train_loss {train_loss:.4f} | test_loss {test_loss:.4f} | {fmt(m)} | time train {tr_t:.1f}s test {te_t:.1f}s")

        # last
        save_ckpt(fold_dir / "last.pth", model, epoch, m, optimizer.state_dict())

        # best（AUROC 优先）
        sc = score_for_best(m)
        if sc > best_score:
            best_score = sc
            best_row = dict(m)
            save_ckpt(fold_dir / "best.pth", model, epoch, m, optimizer.state_dict())

        # 如需每 epoch 都留权重，保留这行；否则可注释以节省磁盘
        torch.save(model.state_dict(), str(fold_dir / f"epoch{epoch:03d}.pth"))

    print(f"[{dataset}] fold{fold} best: {fmt(best_row)}")
    return best_row

# ====== 汇总 5 折 ======
def summarize(all_best: List[Dict[str, float]]):
    keys = ["auroc","auprc","f1","acc","sen","mcc"]
    means = {k: float(np.nanmean([d.get(k, np.nan) for d in all_best])) for k in keys}
    stds  = {k: float(np.nanstd ([d.get(k, np.nan) for d in all_best])) for k in keys}
    s = " | ".join([f"{k.upper()} {means[k]:.4f}±{stds[k]:.4f}" for k in keys])
    print(f"[SUMMARY over 5 folds] {s}")
    return means, stds

# ====== CLI ======
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ugca_dti.yaml")   # 占位
    ap.add_argument("--dataset", required=True, help="bindingdb / davis / biosnap（不区分大小写）")
    ap.add_argument("--dataset-dirname", required=True, help="如 bindingdb_k5 / davis_k5 / biosnap_k5")
    ap.add_argument("--data-root", required=True, help="如 /root/lanyun-tmp")
    ap.add_argument("--out", required=True, help="如 /root/lanyun-tmp/ugca-runs/bindingdb")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resume", default="", help="可选：该 fold 的 last.pth 路径")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 设备选择（若曾 export 了 CPU-only，可临时在命令前加 CUDA_VISIBLE_DEVICES=0）
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device: {'cuda' if use_cuda else 'cpu'} | cuda_available: {use_cuda}")
    if use_cuda:
        try:
            print("gpu:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    # 输出根目录
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # 固定随机种子（如你有自己的 seed util，可替换）
    torch.manual_seed(42)
    np.random.seed(42)

    dataset = args.dataset.lower()
    data_root = Path(args.data_root)
    dataset_dirname = args.dataset_dirname

    all_best: List[Dict[str, float]] = []
    for fold in range(1, 6):
        print(f"=== dataset: {dataset} fold: {fold} ===")
        best_row = run_one_fold(args, dataset, fold, device, out_root, data_root, dataset_dirname)
        all_best.append(best_row)

    summarize(all_best)
