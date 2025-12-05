import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, precision_score, recall_score,
                             matthews_corrcoef, confusion_matrix)
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import shutil
import warnings
import time

# 忽略不必要的 sklearn 警告
warnings.filterwarnings("ignore")

from src.model import UGCA_DTI
from src.datamodule import DTIDataModule


# ==========================================
# 工具函数：计算详细指标 (全指标版)
# ==========================================
def compute_metrics(y_true, y_logits):
    # [Fix Warning] 确保输入是平坦的 numpy array
    y_true = np.array(y_true).flatten()
    y_logits = np.array(y_logits).flatten()

    # Sigmoid 转换概率
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))

    # 1. 基础 AUC 指标
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except:
        auroc = 0.5
    try:
        auprc = average_precision_score(y_true, y_prob)
    except:
        auprc = 0.0

    # 2. 搜索最佳 F1 阈值
    best_f1 = 0
    best_thresh = 0.5
    # 粗搜 + 细搜 (为了速度，这里用0.05步长)
    thresholds = np.arange(0.1, 0.95, 0.05)

    for thresh in thresholds:
        y_pred_tmp = (y_prob > thresh).astype(int)
        f1_tmp = f1_score(y_true, y_pred_tmp, zero_division=0)
        if f1_tmp > best_f1:
            best_f1 = f1_tmp
            best_thresh = thresh

    # 3. 使用最佳阈值计算所有分类指标
    y_pred = (y_prob > best_thresh).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)  # Recall = Sensitivity
    mcc = matthews_corrcoef(y_true, y_pred)

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'AUROC': auroc,
        'AUPRC': auprc,
        'F1': best_f1,
        'Acc': acc,
        'Sens': sensitivity,
        'Spec': specificity,
        'Prec': precision,
        'MCC': mcc,
        'Thresh': best_thresh
    }


def print_metrics(metrics, prefix="Val"):
    """格式化打印指标"""
    print(f"{prefix:<5} | Loss: {metrics['Loss']:.4f} | "
          f"AUC: {metrics['AUROC']:.4f} | AUPRC: {metrics['AUPRC']:.4f} | "
          f"F1: {metrics['F1']:.4f} | Acc: {metrics['Acc']:.4f} | "
          f"Sens: {metrics['Sens']:.4f} | Spec: {metrics['Spec']:.4f} | "
          f"MCC: {metrics['MCC']:.4f}")


# ==========================================
# Loss & Arguments
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def get_args():
    parser = argparse.ArgumentParser(description="UGCA-DTI V1.0 Training")
    # Data
    parser.add_argument('--dataset', type=str, default='DAVIS')
    parser.add_argument('--split-mode', type=str, default='warm',
                        choices=['warm', 'random', 'cold_drug', 'cold_protein'])
    parser.add_argument('--data-root', type=str, default='/root/lanyun-tmp')
    parser.add_argument('--cache-root', type=str, default='/root/lanyun-tmp/cache')

    # Fold Setting
    parser.add_argument('--fold', type=int, default=0, help='Specific fold index (0-4) to run')
    parser.add_argument('--run-all', action='store_true',
                        help='If set, run all 5 folds sequentially and average results')

    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    # Model
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--d-fuse', type=int, default=512)
    parser.add_argument('--gate-mode', type=str, default='mu_times_evi')
    parser.add_argument('--pooling', type=str, default='meanmax')
    parser.add_argument('--fusion-head', type=str, default='match-mlp')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--g-min', type=float, default=1e-3)
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr-scheduler', type=str, default='cosine', choices=['cosine', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--loss', type=str, default='focal')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--resume', action='store_true', help='Resume from last.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()


# ==========================================
# Main Loops
# ==========================================
def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_logits_list = []
    all_targets_list = []

    pbar = tqdm(loader, desc="Train" if is_train else "Val/Test", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        if is_train:
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch['label'])
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(batch)
                loss = criterion(logits, batch['label'])

        total_loss += loss.item()

        # 收集结果
        all_logits_list.append(logits.detach().cpu().numpy())
        all_targets_list.append(batch['label'].detach().cpu().numpy())

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)

    # 统一拼接
    all_logits = np.concatenate(all_logits_list, axis=0)
    all_targets = np.concatenate(all_targets_list, axis=0)

    metrics = compute_metrics(all_targets, all_logits)
    metrics['Loss'] = avg_loss
    return metrics


def train_single_fold(args, fold_idx):
    """单独训练一个Fold的逻辑封装"""
    # 1. 目录设置
    exp_name = f"{args.dataset}_{args.split_mode}"
    base_dir = os.path.join(args.data_root, 'ugca-dti', exp_name)
    fold_dir = os.path.join(base_dir, f"fold{fold_idx + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"Starting Fold {fold_idx + 1} / 5")
    print(f"Output Dir: {fold_dir}")
    print(f"{'=' * 50}\n")

    # 2. 数据
    dm = DTIDataModule(
        args.data_root, args.cache_root, args.dataset,
        split_mode=args.split_mode, batch_size=args.batch_size,
        num_workers=args.num_workers, fold=fold_idx  # 注意这里传入 fold_idx
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # 3. 模型
    model = UGCA_DTI(args).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # [Scheduler Switch]
    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    criterion = FocalLoss() if args.loss == 'focal' else nn.BCEWithLogitsLoss()

    # 4. Resume
    start_epoch = 0
    best_auprc = 0.0
    patience = 0
    last_ckpt_path = os.path.join(fold_dir, 'last.pt')
    best_ckpt_path = os.path.join(fold_dir, 'best.pt')

    if args.resume and os.path.exists(last_ckpt_path):
        print(f"Resuming from {last_ckpt_path}...")
        checkpoint = torch.load(last_ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_auprc = checkpoint['best_auprc']
        patience = checkpoint.get('patience', 0)
        print(f"Resumed at Epoch {start_epoch}, Best AUPRC: {best_auprc:.4f}")

    # 5. Log
    log_csv_path = os.path.join(fold_dir, 'log.csv')
    headers = "Epoch,lr,Train_Loss,Train_AUC,Val_Loss,Val_AUC,Val_AUPRC,Val_F1,Val_Acc,Val_Sens,Val_Spec,Val_Prec,Val_MCC\n"
    if not os.path.exists(log_csv_path) or start_epoch == 0:
        with open(log_csv_path, 'w') as f: f.write(headers)

    # 6. Loop
    for epoch in range(start_epoch, args.epochs):
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"\nFold {fold_idx + 1} | Epoch {epoch + 1}/{args.epochs} | LR: {curr_lr:.2e}")

        # Train
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, args.device, is_train=True)
        # Val
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, args.device, is_train=False)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        # Output
        print_metrics(train_metrics, prefix="Train")
        print_metrics(val_metrics, prefix="Val  ")

        # CSV Logging
        log_str = (f"{epoch + 1},{curr_lr:.6f},{train_metrics['Loss']:.4f},{train_metrics['AUROC']:.4f},"
                   f"{val_metrics['Loss']:.4f},{val_metrics['AUROC']:.4f},{val_metrics['AUPRC']:.4f},"
                   f"{val_metrics['F1']:.4f},{val_metrics['Acc']:.4f},{val_metrics['Sens']:.4f},"
                   f"{val_metrics['Spec']:.4f},{val_metrics['Prec']:.4f},{val_metrics['MCC']:.4f}\n")

        with open(log_csv_path, 'a') as f:
            f.write(log_str)

        # Save Last
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_auprc': best_auprc,
            'patience': patience
        }, last_ckpt_path)

        # Best Save
        if val_metrics['AUPRC'] > best_auprc:
            best_auprc = val_metrics['AUPRC']
            patience = 0
            shutil.copy(last_ckpt_path, best_ckpt_path)
            # print(">>> New Best Model Saved!")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 7. Final Test
    print(f"\nFold {fold_idx + 1} Finished. Loading Best Model for Testing...")
    checkpoint = torch.load(best_ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = run_epoch(model, test_loader, criterion, optimizer, args.device, is_train=False)
    print("\nTest Results:")
    print_metrics(test_metrics, prefix="Test ")

    # Save Results
    res_df = pd.DataFrame([test_metrics])
    res_df.to_csv(os.path.join(fold_dir, 'result.csv'), index=False)

    # Summary Update
    summary_path = os.path.join(base_dir, 'summary.csv')
    fold_res = test_metrics.copy()
    fold_res['Fold'] = fold_idx + 1
    df_new = pd.DataFrame([fold_res])

    if not os.path.exists(summary_path):
        df_new.to_csv(summary_path, index=False)
    else:
        # Avoid duplicate header
        df_new.to_csv(summary_path, mode='a', header=False, index=False)

    return test_metrics


def main():
    args = get_args()

    # 如果指定了 --run-all，则跑完 5 折
    if args.run_all:
        all_fold_metrics = []
        for i in range(5):
            metrics = train_single_fold(args, fold_idx=i)
            all_fold_metrics.append(metrics)

        print(f"\n{'=' * 60}")
        print("ALL 5 FOLDS FINISHED. AVERAGING RESULTS...")
        print(f"{'=' * 60}")

        # 计算平均值
        df_all = pd.DataFrame(all_fold_metrics)
        mean_metrics = df_all.mean(numeric_only=True)
        std_metrics = df_all.std(numeric_only=True)

        print("\nFinal Average Results (Mean +/- Std):")
        for col in mean_metrics.index:
            if col not in ['Fold', 'Thresh', 'Loss']:
                print(f"{col}: {mean_metrics[col]:.4f} +/- {std_metrics[col]:.4f}")

        # 保存到 summary_avg.csv
        exp_name = f"{args.dataset}_{args.split_mode}"
        base_dir = os.path.join(args.data_root, 'ugca-dti', exp_name)
        df_all.loc['Mean'] = mean_metrics
        df_all.loc['Std'] = std_metrics
        df_all.to_csv(os.path.join(base_dir, 'summary_full.csv'))
        print(f"Full summary saved to {os.path.join(base_dir, 'summary_full.csv')}")

    else:
        # 否则只跑指定的 args.fold
        train_single_fold(args, fold_idx=args.fold)


if __name__ == '__main__':
    main()