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

warnings.filterwarnings("ignore")

from src.model import UGCA_DTI
from src.datamodule import DTIDataModule


# ==========================================
# 核心指标计算
# ==========================================
def compute_metrics(y_true, y_logits):
    y_true = np.array(y_true).flatten()
    y_logits = np.array(y_logits).flatten()
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))

    # 基础指标
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except:
        auroc = 0.5
    try:
        auprc = average_precision_score(y_true, y_prob)
    except:
        auprc = 0.0

    # 搜索最佳 F1
    best_f1, best_thresh = 0, 0.5
    thresholds = np.arange(0.1, 0.95, 0.05)
    for thresh in thresholds:
        y_pred_tmp = (y_prob > thresh).astype(int)
        f1_tmp = f1_score(y_true, y_pred_tmp, zero_division=0)
        if f1_tmp > best_f1:
            best_f1, best_thresh = f1_tmp, thresh

    y_pred = (y_prob > best_thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'AUROC': auroc, 'AUPRC': auprc, 'F1': best_f1, 'Acc': acc,
        'Sens': sensitivity, 'Spec': specificity, 'Prec': precision, 'MCC': mcc,
        'Thresh': best_thresh
    }


def print_metrics(metrics, prefix="Val"):
    print(f"{prefix:<5} | Loss: {metrics['Loss']:.4f} | AUC: {metrics['AUROC']:.4f} | "
          f"AUPRC: {metrics['AUPRC']:.4f} | F1: {metrics['F1']:.4f} | MCC: {metrics['MCC']:.4f}")


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
    parser = argparse.ArgumentParser(description="UGCA-DTI Auto 5-Fold")
    # Data
    parser.add_argument('--dataset', type=str, default='DAVIS')
    parser.add_argument('--split-mode', type=str, default='warm')
    parser.add_argument('--data-root', type=str, default='/root/lanyun-tmp')
    parser.add_argument('--cache-root', type=str, default='/root/lanyun-tmp/cache')

    # Model Structure
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--d-fuse', type=int, default=256)
    parser.add_argument('--gate-mode', type=str, default='mu_times_evi')
    parser.add_argument('--pooling', type=str, default='attn', choices=['meanmax', 'attn', 'mh-attn'])
    parser.add_argument('--fusion-head', type=str, default='match-mlp', choices=['match-mlp', 'concat-mlp'])

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--wd', type=float, default=1e-2, help='Weight Decay')

    # Switches
    parser.add_argument('--lr-scheduler', type=str, default='cosine', choices=['cosine', 'none'])
    parser.add_argument('--loss', type=str, default='bce', choices=['focal', 'bce'])
    parser.add_argument('--monitor', type=str, default='auprc', choices=['auprc', 'auroc'],
                        help='Metric to select best model')
    parser.add_argument('--patience', type=int, default=20)

    # Misc
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lamb', type=float, default=0.6)
    parser.add_argument('--temp', type=float, default=0.6)
    parser.add_argument('--g-min', type=float, default=1e-3)
    parser.add_argument('--resume', action='store_true')
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

        all_logits_list.append(logits.detach().cpu().numpy())
        all_targets_list.append(batch['label'].detach().cpu().numpy())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    all_logits = np.concatenate(all_logits_list, axis=0)
    all_targets = np.concatenate(all_targets_list, axis=0)

    metrics = compute_metrics(all_targets, all_logits)
    metrics['Loss'] = avg_loss
    return metrics


def train_single_fold(args, fold_idx):
    exp_name = f"{args.dataset}_{args.split_mode}"
    base_dir = os.path.join(args.data_root, 'ugca-dti', exp_name)
    fold_dir = os.path.join(base_dir, f"fold{fold_idx + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    print(f"\n{'=' * 50}\nFold {fold_idx + 1}/5 | Monitor: {args.monitor.upper()} | Output: {fold_dir}\n{'=' * 50}")

    dm = DTIDataModule(
        args.data_root, args.cache_root, args.dataset,
        split_mode=args.split_mode, batch_size=args.batch_size,
        num_workers=args.num_workers, fold=fold_idx
    )
    dm.setup()

    model = UGCA_DTI(args).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    criterion = FocalLoss() if args.loss == 'focal' else nn.BCEWithLogitsLoss()

    start_epoch, best_metric, patience = 0, 0.0, 0
    last_ckpt = os.path.join(fold_dir, 'last.pt')
    best_ckpt = os.path.join(fold_dir, 'best.pt')
    monitor_key = 'AUPRC' if args.monitor == 'auprc' else 'AUROC'

    if args.resume and os.path.exists(last_ckpt):
        print(f"Resuming fold {fold_idx + 1}...")
        ckpt = torch.load(last_ckpt)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_metric = ckpt['best_metric']

    # 包含 Train 和 Val 全指标的日志 Header
    log_path = os.path.join(fold_dir, 'log.csv')
    cols = ["Epoch", "lr",
            "Train_Loss", "Train_AUC", "Train_AUPRC", "Train_F1", "Train_Acc",
            "Val_Loss", "Val_AUC", "Val_AUPRC", "Val_F1", "Val_Acc", "Val_Sens", "Val_Spec", "Val_Prec", "Val_MCC"]

    if not os.path.exists(log_path) or start_epoch == 0:
        with open(log_path, 'w') as f:
            f.write(",".join(cols) + "\n")

    for epoch in range(start_epoch, args.epochs):
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"\nFold {fold_idx + 1} | Ep {epoch + 1}/{args.epochs} | LR: {curr_lr:.2e}")

        train_res = run_epoch(model, dm.train_dataloader(), criterion, optimizer, args.device, is_train=True)
        val_res = run_epoch(model, dm.val_dataloader(), criterion, optimizer, args.device, is_train=False)

        if scheduler:
            scheduler.step()

        print_metrics(train_res, "Train")
        print_metrics(val_res, "Val  ")

        # 记录到 CSV
        row = [epoch + 1, curr_lr,
               train_res['Loss'], train_res['AUROC'], train_res['AUPRC'], train_res['F1'], train_res['Acc'],
               val_res['Loss'], val_res['AUROC'], val_res['AUPRC'], val_res['F1'], val_res['Acc'],
               val_res['Sens'], val_res['Spec'], val_res['Prec'], val_res['MCC']]
        with open(log_path, 'a') as f:
            f.write(",".join(map(str, row)) + "\n")

        # Save
        curr_val = val_res[monitor_key]
        torch.save(
            {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'best_metric': best_metric},
            last_ckpt
        )

        if curr_val > best_metric:
            best_metric = curr_val
            patience = 0
            shutil.copy(last_ckpt, best_ckpt)
            print(f">>> New Best! ({monitor_key}: {best_metric:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Final Test
    print(f"\nFold {fold_idx + 1} Finished. Testing Best Model...")
    ckpt = torch.load(best_ckpt)
    model.load_state_dict(ckpt['model_state_dict'])
    test_res = run_epoch(model, dm.test_dataloader(), criterion, optimizer, args.device, is_train=False)

    print("\nTest Results:")
    print_metrics(test_res, "Test ")

    res_df = pd.DataFrame([test_res])
    res_df.to_csv(os.path.join(fold_dir, 'result.csv'), index=False)

    # 实时写入 Summary，防止中断后数据丢失
    summary_path = os.path.join(base_dir, 'summary.csv')
    fold_res = test_res.copy()
    fold_res['Fold'] = fold_idx + 1
    df_new = pd.DataFrame([fold_res])

    if not os.path.exists(summary_path):
        df_new.to_csv(summary_path, index=False)
    else:
        df_new.to_csv(summary_path, mode='a', header=False, index=False)

    return test_res


def main():
    args = get_args()
    all_metrics = []

    print(f"Starting Auto 5-Fold Cross Validation on {args.dataset} ({args.split_mode})")

    for i in range(5):
        metrics = train_single_fold(args, fold_idx=i)
        all_metrics.append(metrics)

    print("\n" + "=" * 50)
    print("ALL FOLDS COMPLETED. FINAL AVERAGED RESULTS:")
    print("=" * 50)

    df = pd.DataFrame(all_metrics)
    mean = df.mean(numeric_only=True)
    std = df.std(numeric_only=True)

    for k in mean.index:
        if k not in ['Fold', 'Thresh']:
            print(f"{k}: {mean[k]:.4f} ± {std[k]:.4f}")

    # Save Full Summary
    exp_name = f"{args.dataset}_{args.split_mode}"
    path = os.path.join(args.data_root, 'ugca-dti', exp_name, 'summary_full.csv')
    df.loc['Mean'] = mean
    df.loc['Std'] = std
    df.to_csv(path)
    print(f"\nFinal Summary saved to: {path}")


if __name__ == '__main__':
    main()
