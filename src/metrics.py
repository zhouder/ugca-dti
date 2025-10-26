from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, recall_score, matthews_corrcoef

def _find_threshold(y_true, y_score, mode: str) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if mode == 'max_f1':
        ts = np.linspace(0,1,201)
        f1s = [f1_score(y_true, (y_score>=t).astype(int)) for t in ts]
        return float(ts[int(np.nanargmax(f1s))])
    elif mode == 'youden':
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y_true, y_score)
        j = tpr - fpr
        return float(thr[int(np.nanargmax(j))])
    else:
        try:
            return float(mode)
        except:
            return 0.5

def compute_all_metrics(y_true, y_score, threshold: float|str=0.5) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thr = _find_threshold(y_true, y_score, threshold) if isinstance(threshold, str) else float(threshold)
    y_pred = (y_score >= thr).astype(int)
    out = { 'threshold': thr }
    try:
        out['auroc'] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out['auroc'] = float('nan')
    try:
        out['auprc'] = float(average_precision_score(y_true, y_score))
    except Exception:
        out['auprc'] = float('nan')
    out['f1'] = float(f1_score(y_true, y_pred))
    out['acc'] = float(accuracy_score(y_true, y_pred))
    out['sensitivity'] = float(recall_score(y_true, y_pred))
    out['mcc'] = float(matthews_corrcoef(y_true, y_pred))
    return out

def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    bins = np.linspace(0., 1., n_bins+1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        m = (y_score >= lo) & (y_score < hi)
        if m.sum() == 0: continue
        conf = y_score[m].mean()
        acc = (y_true[m] == (y_score[m] >= 0.5)).mean()
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)
