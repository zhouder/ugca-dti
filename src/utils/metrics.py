# src/utils/metrics.py
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, matthews_corrcoef

def pick_threshold_by_max_f1(y_true: np.ndarray, y_prob: np.ndarray):
    # 在所有唯一得分上搜索（含 0.5 兜底）
    cand = np.unique(np.concatenate([y_prob, [0.5]]))
    best_t, best_f1 = 0.5, -1.0
    for t in cand:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else 0.0
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true))>1 else 0.0
    thr, _ = pick_threshold_by_max_f1(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    TP = ((y_true==1) & (y_pred==1)).sum()
    FN = ((y_true==1) & (y_pred==0)).sum()
    sensitivity = TP / max(TP+FN, 1)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred))>1 else 0.0
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1),
        "acc": float(acc),
        "sensitivity": float(sensitivity),
        "mcc": float(mcc),
        "threshold": float(thr),
    }
