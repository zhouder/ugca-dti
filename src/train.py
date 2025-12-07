# train.py
import argparse
import csv
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from torch import optim
from tqdm import tqdm

from src.datamodule import DTIDataModule, set_global_seed
from src.model import UGCADTI


def focal_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0):
    if targets.ndim == 2 and targets.size(-1) == 1:
        targets = targets.squeeze(-1)
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = targets * probs + (1 - targets) * (1 - probs)
    loss = alpha * (1 - pt).pow(gamma) * ce
    return loss.mean()


def bce_with_logits(logits: torch.Tensor, targets: torch.Tensor):
    if targets.ndim == 2 and targets.size(-1) == 1:
        targets = targets.squeeze(-1)
    return F.binary_cross_entropy_with_logits(logits, targets)


@torch.no_grad()
def eval_epoch(
    model: torch.nn.Module,
    loader,
    device,
    fixed_threshold: Optional[float] = None,  # 若提供，则用该阈值，不再在该 split 上搜索
) -> Dict[str, float]:
    model.eval()
    all_logits, all_labels, losses = [], [], []
    for batch in loader:
        mol = batch["mol"].to(device)
        prot = batch["prot"].to(device)
        mask_d = batch["mask_d"].to(device)
        mask_p = batch["mask_p"].to(device)
        label = batch["label"].to(device).squeeze(-1)

        logits = model(mol, prot, mask_d, mask_p)
        loss = F.binary_cross_entropy_with_logits(logits, label)
        losses.append(loss.item())

        all_logits.append(logits.detach().cpu())
        all_labels.append(label.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = 1 / (1 + np.exp(-logits))

    try:
        auroc = roc_auc_score(labels, probs)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(labels, probs)
    except Exception:
        auprc = float("nan")

    # 阈值
    if fixed_threshold is None:
        thresholds = np.arange(0.10, 0.91, 0.05)
        best_f1, best_t = -1.0, 0.5
        for t in thresholds:
            preds = (probs >= t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        use_t = best_t
    else:
        use_t = float(fixed_threshold)

    preds = (probs >= use_t).astype(int)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    sens = recall_score(labels, preds, zero_division=0)
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(labels, preds) if len(np.unique(labels)) > 1 else 0.0
    f1u = f1_score(labels, preds, zero_division=0)

    return {
        "Loss": float(np.mean(losses)),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "F1": float(f1u),
        "Acc": float(acc),
        "Sens": float(sens),
        "Spec": float(spec),
        "Prec": float(prec),
        "MCC": float(mcc),
        "Thresh": float(use_t),
    }


def log_csv(path: str, header: List[str], row: List):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)


def save_checkpoint(path: str, model: torch.nn.Module, optimizer, scheduler, epoch: int):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(state, path)


def train_one_fold(
    args, fold_id: int, dm: DTIDataModule, d_mol_in: int, d_prot_in: int, out_dir: str, device: torch.device
):
    train_loader, val_loader, test_loader = dm.get_loaders_for_fold(fold_id)

    model = UGCADTI(
        d_mol_in=d_mol_in,
        d_prot_in=d_prot_in,
        d_model=args.d_model,
        nlayers=args.nlayers,
        nhead=args.nhead,
        d_fuse=args.d_fuse,
        pooling=args.pooling,
        fusion_head=args.fusion_head,
        gate_mode=args.gate_mode,
        lamb=args.lamb,
        temp=args.temp,
        g_min=args.g_min,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20) if args.lr_scheduler == "cosine" else None

    # 路径
    last_ckpt = os.path.join(out_dir, "last.pt")
    best_path = os.path.join(out_dir, "best.pt")
    log_path = os.path.join(out_dir, "log.csv")
    result_path = os.path.join(out_dir, "result.csv")

    # 断点
    start_epoch = 0
    if args.resume and os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"[Fold {fold_id+1}] Resumed from epoch {start_epoch}")

    # 监控
    best_metric = -1.0
    best_epoch = -1
    best_val_thresh = None
    patience_counter = 0

    header = [
        "Epoch","lr",
        "Train_Loss","Train_AUC","Train_AUPRC","Train_F1","Train_Acc",
        "Val_Loss","Val_AUC","Val_AUPRC","Val_F1","Val_Acc","Val_Sens","Val_Spec","Val_Prec","Val_MCC","Val_Thresh",
    ]

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"[Fold {fold_id+1}] Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            mol = batch["mol"].to(device)
            prot = batch["prot"].to(device)
            mask_d = batch["mask_d"].to(device)
            mask_p = batch["mask_p"].to(device)
            label = batch["label"].to(device).squeeze(-1)

            logits = model(mol, prot, mask_d, mask_p)
            loss = focal_loss_with_logits(logits, label) if args.loss == "focal" else bce_with_logits(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(losses))

        if scheduler is not None:
            scheduler.step()

        # 评估（训练/验证）
        train_metrics = eval_epoch(model, train_loader, device)
        val_metrics = eval_epoch(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]

        row = [
            epoch + 1, lr_now,
            train_metrics["Loss"], train_metrics["AUROC"], train_metrics["AUPRC"], train_metrics["F1"], train_metrics["Acc"],
            val_metrics["Loss"], val_metrics["AUROC"], val_metrics["AUPRC"], val_metrics["F1"], val_metrics["Acc"],
            val_metrics["Sens"], val_metrics["Spec"], val_metrics["Prec"], val_metrics["MCC"], val_metrics["Thresh"],
        ]
        log_csv(log_path, header, row)

        # 控制台详细一行
        print(
            f"[Fold {fold_id+1}] Epoch {epoch+1}/{args.epochs} | LR={lr_now:.6f} | "
            f"Train: loss={train_metrics['Loss']:.4f} auc={train_metrics['AUROC']:.4f} "
            f"auprc={train_metrics['AUPRC']:.4f} f1={train_metrics['F1']:.4f} acc={train_metrics['Acc']:.4f} | "
            f"Val: loss={val_metrics['Loss']:.4f} auc={val_metrics['AUROC']:.4f} "
            f"auprc={val_metrics['AUPRC']:.4f} f1={val_metrics['F1']:.4f} acc={val_metrics['Acc']:.4f} "
            f"sens={val_metrics['Sens']:.4f} spec={val_metrics['Spec']:.4f} mcc={val_metrics['MCC']:.4f} "
            f"thr={val_metrics['Thresh']:.2f}"
        )

        monitor_value = val_metrics["AUPRC"] if args.monitor.lower() == "auprc" else val_metrics["AUROC"]
        improved = monitor_value > best_metric
        if improved:
            best_metric = monitor_value
            best_epoch = epoch + 1
            best_val_thresh = float(val_metrics["Thresh"])
            torch.save({"model": model.state_dict(), "epoch": best_epoch, "monitor": best_metric, "val_best_thresh": best_val_thresh}, best_path)
            patience_counter = 0
            print(f">>> [Fold {fold_id+1}] New BEST on {args.monitor.upper()}: {best_metric:.4f} @ epoch {best_epoch} | val_thr={best_val_thresh:.2f}")
        else:
            patience_counter += 1

        save_checkpoint(last_ckpt, model, optimizer, scheduler, epoch + 1)

        if patience_counter >= args.patience:
            print(f"[Fold {fold_id+1}] Early stopping at epoch {epoch+1}.")
            break

    # ===== Test：加载 best 模型，并按开关决定阈值策略 =====
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])
        saved_val_thr = state.get("val_best_thresh", None)
    else:
        saved_val_thr = None

    if args.test_threshold == "val" and saved_val_thr is not None:
        test_metrics = eval_epoch(model, test_loader, device, fixed_threshold=saved_val_thr)
        thr_used = saved_val_thr
    else:
        test_metrics = eval_epoch(model, test_loader, device, fixed_threshold=None)
        thr_used = test_metrics["Thresh"]

    header_res = ["Loss", "AUROC", "AUPRC", "F1", "Acc", "Sens", "Spec", "Prec", "MCC", "ThreshUsed", "ValBestThresh"]
    row_res = [
        test_metrics["Loss"], test_metrics["AUROC"], test_metrics["AUPRC"], test_metrics["F1"],
        test_metrics["Acc"], test_metrics["Sens"], test_metrics["Spec"], test_metrics["Prec"],
        test_metrics["MCC"], float(thr_used), float(saved_val_thr) if saved_val_thr is not None else np.nan,
    ]
    log_csv(result_path, header_res, row_res)
    return test_metrics


def summarize_all_folds(root_dir: str, n_folds: int = 5):
    sum_full = os.path.join(root_dir, "summary_full.csv")
    rows = []
    for f in range(1, n_folds + 1):
        res_path = os.path.join(root_dir, f"fold{f}", "result.csv")
        if not os.path.exists(res_path):
            continue
        df = pd.read_csv(res_path)
        if len(df):
            r = df.iloc[-1].to_dict()
            r["Fold"] = f
            rows.append(r)
    if not rows:
        return
    df_all = pd.DataFrame(rows)
    num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    mean_row = df_all[num_cols].mean().to_dict(); mean_row["Fold"] = "Mean"
    std_row = df_all[num_cols].std(ddof=0).to_dict(); std_row["Fold"] = "Std"
    pd.concat([df_all, pd.DataFrame([mean_row]), pd.DataFrame([std_row])], ignore_index=True).to_csv(sum_full, index=False)


def parse_args():
    p = argparse.ArgumentParser(description="UGCA-DTI (MolCLR+ESM2) Training")
    # data
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split-mode", type=str, default="warm",
                   choices=["warm", "random", "cold_drug", "cold_protein", "cold_both"])
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--cache-root", type=str, required=True)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    # model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--d-fuse", type=int, default=256)
    p.add_argument("--pooling", type=str, default="attn", choices=["meanmax", "attn", "mh-attn"])
    p.add_argument("--fusion-head", type=str, default="match-mlp", choices=["match-mlp", "concat-mlp"])
    p.add_argument("--gate-mode", type=str, default="mu_times_evi", choices=["mu_times_evi", "mu_only", "evi_only"])
    # train
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--wd", type=float, default=2e-2)
    p.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--loss", type=str, default="bce", choices=["bce", "focal"])
    p.add_argument("--monitor", type=str, default="auprc", choices=["auprc", "auroc"])
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    # gating / attention
    p.add_argument("--lamb", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--g-min", type=float, default=1e-3)
    # misc
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true")
    # threshold policy for test
    p.add_argument("--test-threshold", type=str, default="val", choices=["val", "search"],
                   help="val: 使用最佳模型所在 epoch 的 Val 最优阈值；search: 在 Test 上重新扫描阈值（仅用于分析，不建议做正式汇报）")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device(args.device)

    split_mode = args.split_mode
    exp_root = os.path.join(args.data_root, "ugca-dti", f"{args.dataset}_{split_mode}")
    os.makedirs(exp_root, exist_ok=True)

    # ===== 数据与缓存命中率 =====
    dm = DTIDataModule(
        data_root=args.data_root,
        cache_root=args.cache_root,
        dataset=args.dataset,
        split_mode=split_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    stats = dm.compute_cache_hit_rate()
    print(
        f"[Data] Dataset={args.dataset} | total={stats['total']} | pos_ratio={stats['pos_ratio']:.4f} | "
        f"feature_hit_rate={stats['hit_rate']*100:.2f}% (hit_both={stats['hit_both']}, miss_mol={stats['miss_mol']}, "
        f"miss_prot={stats['miss_prot']}, miss_both={stats['miss_both']})"
    )
    d_mol_in, d_prot_in = dm.get_input_dims()
    print(f"[Data] Input dims -> MolCLR: {d_mol_in}, ESM2: {d_prot_in}")

    # ===== 5 折训练 =====
    for f in range(5):
        fold_dir = os.path.join(exp_root, f"fold{f+1}")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"========== Fold {f+1} ==========")
        train_one_fold(args, f, dm, d_mol_in, d_prot_in, fold_dir, device)

    summarize_all_folds(exp_root, n_folds=5)
    print("Done.")


if __name__ == "__main__":
    main()
