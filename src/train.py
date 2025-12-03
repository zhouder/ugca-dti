import argparse
import os
import random
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datamodule import DTIDataModule, check_datasets_hit
from src.model import UGCADTIModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args) -> UGCADTIModel:
    model = UGCADTIModel(
        d_mol=args.d_mol,
        d_prot=args.d_prot,
        d_chem=args.d_chem,
        d_graph=args.d_graph,
        d_model=args.d_model,
        d_fuse=args.d_fuse,
        nlayers=args.nlayers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pool_type=args.pooling,
        fusion_head=args.fusion_head,
        gate_mode=args.gate_mode,
        gate_lambda=args.gate_lambda,
        gate_min=args.g_min,
        attn_temp=args.attn_temp,
        pool_heads=args.pool_heads,
        mutan_rank=args.mutan_rank,
    )
    return model


def build_optimizer(model: torch.nn.Module, args):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    return optimizer


def build_scheduler(optimizer, args):
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cosine_t0, T_mult=args.cosine_tmult
        )
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
    elif args.lr_scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")
    return scheduler


def build_criterion(args):
    if args.loss == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif args.loss == "wbce":
        # 使用 pos_weight，简单按正负比例计算
        pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args.loss == "focal":
        # focal 在下面手动实现
        return None
    else:
        raise ValueError(f"Unknown loss: {args.loss}")


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    logits: (B,)
    targets: (B,)
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}
    if len(np.unique(y_true)) == 2:
        try:
            metrics["AUROC"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["AUROC"] = np.nan
        try:
            metrics["AUPRC"] = average_precision_score(y_true, y_prob)
        except Exception:
            metrics["AUPRC"] = np.nan
    else:
        metrics["AUROC"] = np.nan
        metrics["AUPRC"] = np.nan

    metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["Sensitivity"] = recall_score(y_true, y_pred, zero_division=0)  # recall
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-8)
    metrics["Specificity"] = specificity
    try:
        metrics["MCC"] = matthews_corrcoef(y_true, y_pred)
    except Exception:
        metrics["MCC"] = 0.0

    return metrics

def find_best_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_thresholds: int = 101,
):
    """
    在 [0,1] 上均匀扫阈值，找到 F1 最高的阈值。
    返回 (best_thr, best_f1)。
    """
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)

    best_thr = 0.5
    best_f1 = -1.0
    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t

    return float(best_thr), float(best_f1)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion,
    args,
    scaler: torch.cuda.amp.GradScaler,
) -> (float, np.ndarray, np.ndarray):
    model.train()
    all_losses: List[float] = []
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        drug_seq = batch["drug_seq"].to(device)
        prot_seq = batch["prot_seq"].to(device)
        drug_mask = batch["drug_mask"].to(device)
        prot_mask = batch["prot_mask"].to(device)
        chem = batch["chem"].to(device)
        graph = batch["graph"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            logits = model(drug_seq, prot_seq, chem, graph, drug_mask, prot_mask)
            labels_float = labels
            if args.loss == "focal":
                loss = focal_loss(logits, labels_float, alpha=args.focal_alpha, gamma=args.focal_gamma)
            else:
                if args.loss == "wbce":
                    # pos_weight 已在 CPU 上，移动到设备
                    criterion.pos_weight = criterion.pos_weight.to(device)
                loss = criterion(logits, labels_float)

        if scaler is not None and args.amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        all_losses.append(loss.item())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())

        pbar.set_postfix(loss=loss.item())

    avg_loss = float(np.mean(all_losses))
    all_probs_arr = np.concatenate(all_probs, axis=0)
    all_labels_arr = np.concatenate(all_labels, axis=0)
    return avg_loss, all_probs_arr, all_labels_arr


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion,
    args,
) -> (float, np.ndarray, np.ndarray):
    model.eval()
    all_losses: List[float] = []
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in loader:
        drug_seq = batch["drug_seq"].to(device)
        prot_seq = batch["prot_seq"].to(device)
        drug_mask = batch["drug_mask"].to(device)
        prot_mask = batch["prot_mask"].to(device)
        chem = batch["chem"].to(device)
        graph = batch["graph"].to(device)
        labels = batch["label"].to(device)

        logits = model(drug_seq, prot_seq, chem, graph, drug_mask, prot_mask)
        labels_float = labels
        if args.loss == "focal":
            loss = focal_loss(logits, labels_float, alpha=args.focal_alpha, gamma=args.focal_gamma)
        else:
            if args.loss == "wbce":
                criterion.pos_weight = criterion.pos_weight.to(device)
            loss = criterion(logits, labels_float)

        all_losses.append(loss.item())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())

    avg_loss = float(np.mean(all_losses) if all_losses else 0.0)
    all_probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
    all_labels_arr = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    return avg_loss, all_probs_arr, all_labels_arr


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_val_auprc: float,
):
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val_auprc": best_val_auprc,
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    epoch = ckpt.get("epoch", 0)
    best_val_auprc = ckpt.get("best_val_auprc", -1e9)
    return epoch, best_val_auprc


def log_epoch_to_csv(
    log_path: str,
    epoch: int,
    phase: str,
    metrics: Dict[str, float],
    loss: float,
    lr: float,
):
    row = {"epoch": epoch, "phase": phase, "loss": loss, "lr": lr}
    row.update(metrics)

    file_exists = os.path.exists(log_path)
    df = pd.DataFrame([row])
    if file_exists:
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, mode="w", header=True, index=False)


def save_result_csv(result_path: str, metrics: Dict[str, float]):
    df = pd.DataFrame([metrics])
    df.to_csv(result_path, index=False)


def save_summary_csv(summary_path: str, fold_metrics: List[Dict[str, float]]):
    if not fold_metrics:
        return
    df = pd.DataFrame(fold_metrics)
    metrics_cols = [c for c in df.columns if c != "fold"]

    mean_row = {"fold": "mean"}
    std_row = {"fold": "std"}
    for m in metrics_cols:
        mean_row[m] = df[m].mean()
        std_row[m] = df[m].std(ddof=0)

    summary_df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="UGCA-DTI v1.0 Training Script")

    # 数据 & 路径
    parser.add_argument("--dataset", type=str, default="DAVIS", help="DAVIS / BindingDB / BioSNAP")
    parser.add_argument("--data-root", type=str, default="/root/lanyun-tmp")
    parser.add_argument("--cache-root", type=str, default="/root/lanyun-tmp/cache")
    parser.add_argument("--output-dir", type=str, default="/root/lanyun-tmp/ugca-runs")
    parser.add_argument("--split-mode", type=str, default="cold-protein")

    # 特征维度
    parser.add_argument("--d-mol", type=int, default=300)
    parser.add_argument("--d-prot", type=int, default=1280)
    parser.add_argument("--d-chem", type=int, default=384)
    parser.add_argument("--d-graph", type=int, default=256)

    # 模型结构
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-fuse", type=int, default=512)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="meanmax")
    parser.add_argument("--pool-heads", type=int, default=4)
    parser.add_argument("--fusion-head", type=str, default="match-mlp")
    parser.add_argument("--gate-mode", type=str, default="mu_times_evi")
    parser.add_argument("--gate-lambda", type=float, default=1.0)
    parser.add_argument("--g-min", type=float, default=1e-3)
    parser.add_argument("--attn-temp", type=float, default=1.0)
    parser.add_argument("--mutan-rank", type=int, default=8)

    # 训练设置
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=15)

    # 优化器 & 学习率
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    parser.add_argument("--cosine-t0", type=int, default=10)
    parser.add_argument("--cosine-tmult", type=int, default=2)

    # 损失函数
    parser.add_argument("--loss", type=str, default="focal", choices=["bce", "wbce", "focal"])
    parser.add_argument("--pos-weight", type=float, default=1.0)  # 给 WBCE 用
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    check_datasets_hit(args.dataset, args.data_root, args.cache_root)

    # 数据模块（生成 5 折）
    datamodule = DTIDataModule(
        dataset_name=args.dataset,
        data_root=args.data_root,
        cache_root=args.cache_root,
        split_mode=args.split_mode,
        n_splits=5,
        val_ratio=0.15,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=(device.type == "cuda"),
    )
    datamodule.setup()

    run_dir = os.path.join(args.output_dir, f"{args.dataset}_{args.split_mode}")
    os.makedirs(run_dir, exist_ok=True)

    fold_results: List[Dict[str, float]] = []

    for fold in range(5):
        fold_id = fold + 1
        print(f"\n========== Fold {fold_id} / 5 ==========")
        fold_dir = os.path.join(run_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        best_ckpt_path = os.path.join(fold_dir, "best.pt")
        last_ckpt_path = os.path.join(fold_dir, "last.pt")
        log_csv_path = os.path.join(fold_dir, "log.csv")
        result_csv_path = os.path.join(fold_dir, "result.csv")

        train_loader, val_loader, test_loader = datamodule.get_dataloaders(fold)

        model = build_model(args).to(device)
        optimizer = build_optimizer(model, args)
        scheduler = build_scheduler(optimizer, args)
        criterion = build_criterion(args)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

        start_epoch = 1
        best_val_auprc = -1e9

        if args.resume and os.path.exists(last_ckpt_path):
            print(f"[Fold {fold_id}] Resuming from last checkpoint: {last_ckpt_path}")
            last_epoch, best_val_auprc = load_checkpoint(
                last_ckpt_path, model, optimizer, scheduler, device
            )
            start_epoch = last_epoch + 1
        else:
            print(f"[Fold {fold_id}] Start training from scratch.")
        no_improve_epochs = 0

        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start_time = time.time()

            # ---------- Train ----------
            train_loss, train_probs, train_labels = train_one_epoch(
                model, train_loader, device, optimizer, criterion, args, scaler
            )
            train_metrics = compute_metrics(train_labels, train_probs, threshold=0.5)

            # ---------- Val ----------
            val_loss, val_probs, val_labels = evaluate(
                model, val_loader, device, criterion, args
            )
            val_metrics = compute_metrics(val_labels, val_probs, threshold=0.5)

            # scheduler step
            if scheduler is not None:
                if args.lr_scheduler == "plateau":
                    # 用验证集 AUPRC 调整
                    val_score = val_metrics.get("AUPRC", 0.0)
                    scheduler.step(val_score)
                else:
                    scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start_time

            # 控制台输出：一行 train，一行 val
            train_str = (
                f"[Fold {fold_id}] Epoch {epoch:03d} | TRAIN "
                f"loss={train_loss:.4f} lr={lr:.6f} "
                f"AUROC={train_metrics['AUROC']:.4f} "
                f"AUPRC={train_metrics['AUPRC']:.4f} "
                f"F1={train_metrics['F1']:.4f} "
                f"ACC={train_metrics['Accuracy']:.4f} "
                f"Sens={train_metrics['Sensitivity']:.4f} "
                f"Spec={train_metrics['Specificity']:.4f} "
                f"Prec={train_metrics['Precision']:.4f} "
                f"MCC={train_metrics['MCC']:.4f}"
            )
            val_str = (
                f"[Fold {fold_id}] Epoch {epoch:03d} | VALID "
                f"loss={val_loss:.4f} lr={lr:.6f} "
                f"AUROC={val_metrics['AUROC']:.4f} "
                f"AUPRC={val_metrics['AUPRC']:.4f} "
                f"F1={val_metrics['F1']:.4f} "
                f"ACC={val_metrics['Accuracy']:.4f} "
                f"Sens={val_metrics['Sensitivity']:.4f} "
                f"Spec={val_metrics['Specificity']:.4f} "
                f"Prec={val_metrics['Precision']:.4f} "
                f"MCC={val_metrics['MCC']:.4f} "
                f"time={epoch_time:.1f}s"
            )
            print(train_str)
            print(val_str)

            # 记录到 log.csv
            log_epoch_to_csv(log_csv_path, epoch, "train", train_metrics, train_loss, lr)
            log_epoch_to_csv(log_csv_path, epoch, "val", val_metrics, val_loss, lr)

            # 保存 last
            save_checkpoint(
                last_ckpt_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_auprc,
            )

            # 根据 val AUPRC 保存 best
            val_auprc = val_metrics.get("AUPRC", -1e9)
            prev_best = best_val_auprc
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                save_checkpoint(
                    best_ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_auprc,
                )
                no_improve_epochs = 0  # 🆕 有提升，归零
                if prev_best < 0:
                    prev_str = "N/A"
                else:
                    prev_str = f"{prev_best:.4f}"
                print(
                    f"[Fold {fold_id}] Epoch {epoch:03d} -> NEW BEST VAL AUPRC "
                    f"{best_val_auprc:.4f} (prev={prev_str}), saved best.pt"
                )
            else:
                no_improve_epochs += 1  # 🆕 没提升，+1

            # 🆕 Early Stopping 判断
            if no_improve_epochs >= args.patience:
                print(
                    f"[Fold {fold_id}] Early stopping at epoch {epoch:03d} "
                    f"(no val AUPRC improvement in {args.patience} epochs)."
                )
                break

        # ---------- 使用 best.pt 做验证阈值搜索 + 测试 ----------
        if os.path.exists(best_ckpt_path):
            print(f"[Fold {fold_id}] Loading best checkpoint for testing: {best_ckpt_path}")
            load_checkpoint(best_ckpt_path, model, None, None, device)
        else:
            print(f"[Fold {fold_id}] best.pt not found, using last.pt for testing.")
            if os.path.exists(last_ckpt_path):
                load_checkpoint(last_ckpt_path, model, None, None, device)

        # 先在 VAL 上找 F1 最优阈值
        val_loss_best, val_probs_best, val_labels_best = evaluate(
            model, val_loader, device, criterion, args
        )
        best_thr, best_val_f1 = find_best_f1_threshold(val_labels_best, val_probs_best)
        val_metrics_best = compute_metrics(val_labels_best, val_probs_best, threshold=best_thr)
        print(
            f"[Fold {fold_id}] BEST VAL (for threshold) "
            f"loss={val_loss_best:.4f} thr={best_thr:.3f} "
            f"AUROC={val_metrics_best['AUROC']:.4f} "
            f"AUPRC={val_metrics_best['AUPRC']:.4f} "
            f"F1={val_metrics_best['F1']:.4f}"
        )

        # 再在 TEST 上用同一个阈值
        test_loss, test_probs, test_labels = evaluate(
            model, test_loader, device, criterion, args
        )
        test_metrics = compute_metrics(test_labels, test_probs, threshold=best_thr)
        test_metrics["threshold"] = best_thr  # 顺便记到 csv 里

        print(
            f"[Fold {fold_id}] TEST "
            f"loss={test_loss:.4f} thr={best_thr:.3f} "
            f"AUROC={test_metrics['AUROC']:.4f} "
            f"AUPRC={test_metrics['AUPRC']:.4f} "
            f"F1={test_metrics['F1']:.4f} "
            f"ACC={test_metrics['Accuracy']:.4f} "
            f"Sens={test_metrics['Sensitivity']:.4f} "
            f"Spec={test_metrics['Specificity']:.4f} "
            f"Prec={test_metrics['Precision']:.4f} "
            f"MCC={test_metrics['MCC']:.4f}"
        )

        # 保存 result.csv（fold 用 1-based）
        result_row = {"fold": fold_id}
        result_row.update(test_metrics)
        save_result_csv(result_csv_path, result_row)

        fold_results.append(result_row)

    # ---------- 所有折结束，生成 summary.csv ----------
    summary_path = os.path.join(run_dir, "summary.csv")
    save_summary_csv(summary_path, fold_results)
    print(f"\nAll folds done. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
