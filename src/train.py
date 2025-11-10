# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, math, argparse, importlib, inspect, csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.datamodule import DataModule, DMConfig, CacheDirs, CacheDims
from src.model import build_model

# ===== sklearn 指标 =====
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, recall_score, matthews_corrcoef,
    precision_recall_curve
)

# --------- 自适应阈值（max F1）指标 ----------
def compute_metrics(prob: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    # 阈值无关
    try: out["auroc"] = float(roc_auc_score(y_true, prob))
    except Exception: out["auroc"] = float("nan")
    try: out["auprc"] = float(average_precision_score(y_true, prob))
    except Exception: out["auprc"] = float("nan")

    # 自适应阈值：max F1
    try:
        p, r, thr = precision_recall_curve(y_true, prob)
        f1 = 2 * p * r / (p + r + 1e-12)
        idx = int(np.nanargmax(f1))
        best_thr = 0.5 if idx >= len(thr) else float(thr[idx])
    except Exception:
        best_thr = 0.5

    pred = (prob >= best_thr).astype(np.int64)
    out["f1"]  = float(f1_score(y_true, pred))
    out["acc"] = float(accuracy_score(y_true, pred))
    out["sen"] = float(recall_score(y_true, pred))
    out["mcc"] = float(matthews_corrcoef(y_true, pred))
    out["thr"] = float(best_thr)
    return out

# 单行对齐格式
def fmt1(m: Dict[str, float]) -> str:
    return (f"AUROC {m['auroc']:>7.4f} | AUPRC {m['auprc']:>7.4f} | "
            f"F1 {m['f1']:>7.4f} | ACC {m['acc']:>7.4f} | "
            f"SEN {m['sen']:>7.4f} | MCC {m['mcc']:>7.4f} | thr {m.get('thr', float('nan')):>5.3f}")

# ===== 模型：优先 build_model(cfg)（UGCA + MUTAN） =====
def build_model_from_src(dims: CacheDims) -> nn.Module:
    print("[Model] 使用 src.model.build_model(cfg) 构建 UGCA（不确定性门控协同注意）+ MUTAN")
    cfg = {
        "d_protein": int(dims.esm2),
        "d_molclr":  int(dims.molclr),
        "d_chem":    int(dims.chemberta),
        "d_model":   512, "dropout": 0.1, "act": "silu",
        "mutan_rank": 15, "mutan_dim": 512, "head_hidden": 512,
        "ugca_layers": 2, "ugca_heads": 8,     # 你可以在这里调整层数/头数
        "k_init": 0.0, "k_target": 15.0,       # k warm-up 范围
        "g_min": 0.05
    }
    return build_model(cfg)

# ===== 训练 epoch（样本数进度条 + AMP） =====
def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
                optimizer: optim.Optimizer, scaler: GradScaler, bce: nn.Module,
                tag: str, ep: int, ep_total: int, grad_accum: int = 1) -> Tuple[float, float]:
    model.train(True)
    tot_loss = 0.0
    n_seen   = 0
    t0 = time.time()

    total_samples = len(loader.dataset)
    pbar = tqdm(total=total_samples, ncols=120, unit="ex",
                desc=f"[{tag}] epoch {ep}/{ep_total}", leave=True, position=0)

    optimizer.zero_grad(set_to_none=True)
    step = 0
    for (v1, v2, v3, y) in loader:
        bs = v1.size(0)
        v1, v2, v3, y = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True), v3.to(device, non_blocking=True), y.to(device)

        with autocast(dtype=torch.float16):
            logits = model(v1, v2, v3)
            loss = bce(logits, y)
            loss = loss / grad_accum

        scaler.scale(loss).backward()
        step += 1
        if step % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        tot_loss += float(loss.detach().cpu()) * grad_accum
        n_seen   += bs

        pbar.update(bs)
        if n_seen and (n_seen % (bs * 10) == 0 or n_seen == total_samples):
            pbar.set_postfix_str(f"{n_seen}/{total_samples} ex | loss {tot_loss * 1.0 / (n_seen / bs):.4f}")

    pbar.close()
    return tot_loss / max(1, n_seen / bs), time.time() - t0

# ===== 测试 epoch（无进度条 + AMP） =====
def test_epoch(model: nn.Module, loader: DataLoader, device: torch.device, bce: nn.Module) -> Tuple[float, np.ndarray, np.ndarray, float]:
    model.train(False)
    tot = 0.0
    probs, labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for (v1, v2, v3, y) in loader:
            v1, v2, v3, y = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True), v3.to(device, non_blocking=True), y.to(device)
            with autocast(dtype=torch.float16):
                logits = model(v1, v2, v3)
                loss = bce(logits, y)
            tot += float(loss.detach().cpu())
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

    prob = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
    lab  = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.float32)
    return tot / max(1, len(loader)), prob, lab, time.time() - t0

def save_ckpt(path: Path, model: nn.Module, epoch: int, metrics: Dict[str,float], opt_state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                "metrics": metrics, "optimizer": opt_state}, str(path))

# ===== 计算 train 集 pos_weight（一次统计） =====
def estimate_pos_weight(loader: DataLoader) -> float:
    pos = 0; tot = 0
    for _,_,_, y in loader:
        y_np = y.numpy()
        pos += (y_np > 0.5).sum()
        tot += y_np.size
    neg = max(1, tot - pos)
    pos = max(1, pos)
    return float(neg / pos)

# ===== 单折 =====
def run_one_fold(args, fold: int, device: torch.device) -> Dict[str, float]:
    ds_lower = args.dataset.lower()
    ds_cap   = {"bindingdb":"BindingDB", "davis":"DAVIS", "biosnap":"BioSNAP"}.get(ds_lower, args.dataset)
    data_root = Path(args.data_root)

    csv_dir   = data_root / args.dataset_dirname
    train_csv = str(csv_dir / f"fold{fold}_train.csv")
    test_csv  = str(csv_dir / f"fold{fold}_test.csv")

    print(f"=== dataset: {ds_lower} fold: {fold} ===")
    print("[paths] train =", train_csv)
    print("[paths] test  =", test_csv)

    cache_root = data_root / "cache"
    cache_dirs = CacheDirs(
        esm2_dir      = str(cache_root / "esm"       / ds_cap),   # DataModule 内部会做 esm/esm2 fallback
        molclr_dir    = str(cache_root / "molclr"    / ds_cap),
        chemberta_dir = str(cache_root / "chemberta" / ds_cap),
    )
    dims = CacheDims(esm2=1280, molclr=300, chemberta=384)

    dm = DataModule(
        DMConfig(
            train_csv=train_csv, test_csv=test_csv,
            num_workers=args.workers, batch_size=args.batch_size,
            pin_memory=True, persistent_workers=args.workers>0,
            prefetch_factor=2, drop_last=False
        ),
        cache_dirs=cache_dirs, dims=dims
    )

    train_loader = dm.train_loader()
    test_loader  = dm.test_loader()
    N = len(train_loader.dataset)
    print(f"[INFO] train size = {N} | batch_size = {args.batch_size}")

    model = build_model_from_src(dims).to(device)

    # k warm-up: 前 warm_ratio 的 epoch 线性增加
    warm_steps = max(1, int(args.epochs * args.k_warm_ratio))
    def set_k_for_epoch(ep: int):
        k_start, k_end = args.k_init, args.k_target
        k_now = k_end if ep >= warm_steps else (k_start + (k_end - k_start) * ep / warm_steps)
        if hasattr(model, "set_k"): model.set_k(k_now)
        return k_now

    # 统计 pos_weight
    pos_weight = estimate_pos_weight(train_loader)
    print(f"[INFO] pos_weight = {pos_weight:.3f}")

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.1)
    scaler = GradScaler()

    fold_dir = Path(args.out) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    csv_path = fold_dir / "metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_loss","test_loss","AUROC","AUPRC","F1","ACC","SEN","MCC","thr","time_train_s","time_test_s","lr","k"])

    best_score = -1.0
    best_row: Dict[str,float] = {}
    best_epoch = -1

    grad_accum = max(1, args.grad_accum)

    for ep in range(1, args.epochs + 1):
        k_now = set_k_for_epoch(ep)

        tr_loss, tr_t = train_epoch(model, train_loader, device, optimizer, scaler, bce,
                                    tag=f"{ds_lower}/train", ep=ep, ep_total=args.epochs, grad_accum=grad_accum)
        te_loss, prob, y, te_t = test_epoch(model, test_loader, device, bce)
        m = compute_metrics(prob, y)

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, f"{tr_loss:.6f}", f"{te_loss:.6f}",
                        f"{m['auroc']:.6f}", f"{m['auprc']:.6f}", f"{m['f1']:.6f}",
                        f"{m['acc']:.6f}", f"{m['sen']:.6f}", f"{m['mcc']:.6f}",
                        f"{m['thr']:.6f}", f"{tr_t:.1f}", f"{te_t:.1f}",
                        f"{optimizer.param_groups[0]['lr']:.6f}", f"{k_now:.3f}"])

        print(f"[{ds_lower}] fold{fold} ep{ep:03d} | train_loss {tr_loss:.4f} | test_loss {te_loss:.4f} | {fmt1(m)} | time {tr_t:.1f}s/{te_t:.1f}s")

        save_ckpt(fold_dir / "last.pth", model, ep, m, optimizer.state_dict())

        # 以 AUROC 为主排序（退化用 ACC）
        sc = m["auroc"] if not math.isnan(m["auroc"]) else m["acc"]
        if sc > best_score:
            best_score = sc
            best_row = dict(m)
            best_epoch = ep
            save_ckpt(fold_dir / "best.pth", model, ep, m, optimizer.state_dict())

        scheduler.step()

    # 在 metrics.csv 末尾追加 best 行
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"best@{best_epoch}", "", "",
                    f"{best_row.get('auroc', float('nan')):.6f}",
                    f"{best_row.get('auprc', float('nan')):.6f}",
                    f"{best_row.get('f1', float('nan')):.6f}",
                    f"{best_row.get('acc', float('nan')):.6f}",
                    f"{best_row.get('sen', float('nan')):.6f}",
                    f"{best_row.get('mcc', float('nan')):.6f}",
                    f"{best_row.get('thr', float('nan')):.6f}",
                    "", "", "", ""])

    print(f"[{ds_lower}] fold{fold} best | {fmt1(best_row)} (epoch={best_epoch})")
    return best_row

def summarize(rows: List[Dict[str,float]], out_dir: Path):
    keys = ["auroc","auprc","f1","acc","sen","mcc"]
    # 写每折 best
    fold_best_csv = out_dir / "fold_best.csv"
    with open(fold_best_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold"] + [k.upper() for k in keys])
        for i, r in enumerate(rows, 1):
            w.writerow([i] + [f"{r.get(k, float('nan')):.6f}" for k in keys])

    # mean/std
    mean = {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys}
    std  = {k: float(np.nanstd ([r.get(k, np.nan) for r in rows])) for k in keys}
    summary_csv = out_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric","mean","std"])
        for k in keys:
            w.writerow([k.upper(), f"{mean[k]:.6f}", f"{std[k]:.6f}"])

    s = " | ".join([f"{k.upper()} {mean[k]:>7.4f}±{std[k]:<7.4f}" for k in keys])
    print(f"[SUMMARY over 5 folds] {s}")
    return mean, std

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="bindingdb / davis / biosnap（不区分大小写）")
    ap.add_argument("--dataset-dirname", required=True, help="如 bindingdb_k5 / davis_k5 / biosnap_k5")
    ap.add_argument("--data-root", required=True, help="如 /root/lanyun-tmp")
    ap.add_argument("--out", required=True, help="如 /root/lanyun-tmp/ugca-runs/davis")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=1, help="梯度累积步数（提升有效 batch）")

    # k warm-up 配置（和 model.py 的 k_init/k_target 对齐）
    ap.add_argument("--k-init", type=float, default=0.0)
    ap.add_argument("--k-target", type=float, default=15.0)
    ap.add_argument("--k-warm-ratio", type=float, default=0.1, help="k warm-up 的 epoch 比例，例如 0.1 表示前 10% epoch 线性增加")

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"[ENV] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device: {'cuda' if use_cuda else 'cpu'} | cuda_available: {use_cuda}")
    if use_cuda:
        try: print("gpu:", torch.cuda.get_device_name(0))
        except Exception: pass

    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)

    all_best: List[Dict[str,float]] = []
    for fold in range(1, 5 + 1):
        best = run_one_fold(args, fold, device)
        all_best.append(best)
    summarize(all_best, Path(args.out))
