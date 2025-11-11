# src/train.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, math, argparse, importlib, inspect, csv, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.datamodule import DataModule, DMConfig, CacheDirs, CacheDims

# =============== metrics ===============
try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        accuracy_score, recall_score, matthews_corrcoef
    )
    SKLEARN = True
except Exception:
    SKLEARN = False
    print("[WARN] scikit-learn 不可用，将退化到少量指标。")

def compute_metrics(prob: np.ndarray, y_true: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    out: Dict[str, float] = {}
    pred = (prob >= thr).astype(np.int64)
    out["acc"] = float(accuracy_score(y_true, pred)) if SKLEARN else float((pred == y_true).mean())
    out["sen"] = float(recall_score(y_true, pred))    if SKLEARN else float((pred[y_true == 1] == 1).mean() if (y_true == 1).any() else 0.0)
    out["f1"]  = float(f1_score(y_true, pred))        if SKLEARN else float("nan")
    out["mcc"] = float(matthews_corrcoef(y_true, pred)) if SKLEARN else float("nan")
    try:    out["auroc"] = float(roc_auc_score(y_true, prob)) if SKLEARN else float("nan")
    except Exception: out["auroc"] = float("nan")
    try:    out["auprc"] = float(average_precision_score(y_true, prob)) if SKLEARN else float("nan")
    except Exception: out["auprc"] = float("nan")
    return out

def find_best_threshold(prob: np.ndarray, y_true: np.ndarray, step: float = 0.01,
                        thr_min: float = 0.0, thr_max: float = 1.0) -> Tuple[float, Dict[str,float]]:
    """Grid search best threshold for F1 on (prob, y_true)."""
    best_thr = 0.5
    best_val = -1.0
    best_metrics = None
    thr = float(thr_min)
    # ensure bounds sane
    thr_min = max(0.0, thr_min); thr_max = min(1.0, thr_max)
    while thr <= thr_max + 1e-12:
        m = compute_metrics(prob, y_true, thr=thr)
        val = m.get("f1", float("nan"))
        if not np.isnan(val) and val > best_val:
            best_val = val
            best_thr = float(thr)
            best_metrics = m
        thr += step
    if best_metrics is None:
        best_metrics = compute_metrics(prob, y_true, thr=0.5)
        best_thr = 0.5
    return best_thr, best_metrics

def fmt1(m: Dict[str, float]) -> str:
    return (f"AUROC {m['auroc']:>7.4f} | AUPRC {m['auprc']:>7.4f} | "
            f"F1 {m['f1']:>7.4f} | ACC {m['acc']:>7.4f} | "
            f"SEN {m['sen']:>7.4f} | MCC {m['mcc']:>7.4f}")

# =============== model loader ===============
def build_model_from_src(dims: CacheDims, args) -> nn.Module:
    model_mod = importlib.import_module("src.model")
    if hasattr(model_mod, "build_model"):
        print("[Model] 使用 src.model.build_model(cfg)")
        cfg = {
            "d_protein": int(dims.esm2),
            "d_molclr":  int(dims.molclr),
            "d_chem":    int(dims.chemberta),
            "d_model":   args.d_model, "dropout": args.dropout, "act": "silu",
            "mutan_rank": args.mutan_rank, "mutan_dim": args.mutan_dim, "head_hidden": args.head_hidden,
            # V2
            "sequence": bool(args.sequence),
            "nhead": args.nhead, "nlayers": args.nlayers,
            # Gate
            "gate_type": args.gate_type,
            "gate_mode": args.gate_mode,
            "gate_lambda": args.gate_lambda,
            "g_min": args.g_min,
            "smooth_g": args.smooth_g,
            "topk_ratio": 0.0,  # 训练时动态 warm-up
            # Cross-attn
            "attn_temp": args.attn_temp,
            # Pooling
            "pool_type": args.pooling,
            # Reg
            "entropy_reg": args.entropy_reg,
        }
        return getattr(model_mod, "build_model")(cfg)
    raise RuntimeError("在 src/model.py 中没找到 build_model(cfg)。")

# =============== losses ===============
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)
        w = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = - w * (1 - pt).pow(self.gamma) * torch.log(pt + 1e-8)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def make_criterion(args, device: torch.device, train_csv: str):
    if args.loss_type == "focal":
        return FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    elif args.loss_type == "bce_weighted":
        # 读取 CSV 估计 pos_weight = neg/pos
        pos, neg = 0, 0
        with open(train_csv, "r", encoding="utf-8", newline="") as f:
            rd = csv.reader(f)
            header = next(rd, None)
            si = 2 if (header and len(header)>=3 and header[2].strip().lower()=="label") else None
            if header and si is None:
                try:
                    val = float(header[2]); pos += (val >= 0.5); neg += (val < 0.5)
                except Exception:
                    pass
            for row in rd:
                if not row: continue
                try:
                    val = float(row[2])
                    if val >= 0.5: pos += 1
                    else: neg += 1
                except Exception:
                    continue
        pw = (neg / max(1, pos)) if pos>0 else 1.0
        print(f"[LOSS] BCEWithLogitsLoss(pos_weight={pw:.2f})  (pos={pos},neg={neg})")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], dtype=torch.float32, device=device))
    else:
        return nn.BCEWithLogitsLoss()

# =============== schedulers ===============
def build_warmup_cosine(optimizer, total_epochs: int, warmup_ratio: float = 0.05):
    warmup_epochs = max(1, int(total_epochs * warmup_ratio))
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        progress = (current_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

# =============== train/test epoch ===============
def _batch_to_device(batch, device):
    if len(batch) == 4:
        v1, v2, v3, y = batch
        return (v1.to(device, non_blocking=True),
                v2.to(device, non_blocking=True),
                v3.to(device, non_blocking=True),
                y.to(device))
    elif len(batch) == 6:
        P, Pm, D, Dm, C, y = batch
        return (P.to(device, non_blocking=True),
                Pm.to(device, non_blocking=True),
                D.to(device, non_blocking=True),
                Dm.to(device, non_blocking=True),
                C.to(device, non_blocking=True),
                y.to(device))
    else:
        raise RuntimeError(f"Unknown batch format len={len(batch)}")

def _forward_any(model: nn.Module, batch_tensors, topk_ratio: Optional[float] = None):
    if len(batch_tensors) == 4:
        v1, v2, v3, y = batch_tensors
        logits = model(v1, v2, v3)
        return logits, y
    else:
        P, Pm, D, Dm, C, y = batch_tensors
        logits = model(P, Pm, D, Dm, C, topk_ratio=topk_ratio)
        return logits, y

def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
                optimizer: optim.Optimizer, criterion: nn.Module,
                tag: str, ep: int, ep_total: int,
                gate_budget: float = 0.0, gate_rho: float = 0.6,
                evi_reg_lambda: float = 0.0, entropy_reg_lambda: float = 0.0,
                amp: bool = True, scaler: Optional["torch.cuda.amp.GradScaler"] = None,
                topk_ratio: float = 0.0) -> Tuple[float, float]:
    model.train(True)
    tot = 0.0
    n_seen = 0
    t0 = time.time()

    total_samples = len(loader.dataset)
    pbar = tqdm(total=total_samples, ncols=120, unit="ex",
                desc=f"[{tag}] epoch {ep}/{ep_total}", leave=True, position=0)

    for batch in loader:
        batch = _batch_to_device(batch, device)
        ctx = torch.cuda.amp.autocast(enabled=amp and device.type=="cuda")
        with ctx:
            logits, y = _forward_any(model, batch, topk_ratio=topk_ratio)
            loss = criterion(logits, y)

            # 门控预算正则（仅 V2 生效）
            if gate_budget > 0.0 and hasattr(model, "last_gates"):
                gd, gp = model.last_gates()
                if gd is not None and gp is not None:
                    def _mean1d(g): return g.mean(dim=1).mean()
                    Lb = ((_mean1d(gd) - gate_rho) ** 2 + (_mean1d(gp) - gate_rho) ** 2) * 0.5
                    loss = loss + gate_budget * Lb

            # Evidential 正则（简化 proxy）
            if evi_reg_lambda > 0.0 and hasattr(model, "_last_gd") and model._last_gd is not None:
                proxy = 1.0 - (model._last_gd.mean() + model._last_gp.mean()) * 0.5
                loss = loss + evi_reg_lambda * proxy

            # 注意力熵正则
            if entropy_reg_lambda > 0.0 and hasattr(model, "_last_entropy"):
                loss = loss + entropy_reg_lambda * model._last_entropy

        optimizer.zero_grad(set_to_none=True)
        if amp and scaler is not None and device.type=="cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = y.size(0)
        tot += float(loss.detach().cpu())
        n_seen += bs
        pbar.update(bs)
        if n_seen and (n_seen % (bs * 10) == 0 or n_seen == total_samples):
            pbar.set_postfix_str(f"{n_seen}/{total_samples} ex | loss {tot * 1.0 / (n_seen / bs):.4f}")

    pbar.close()
    return tot / max(1, n_seen / bs), time.time() - t0

def test_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Tuple[float, np.ndarray, np.ndarray, float]:
    model.train(False)
    tot = 0.0
    probs, labels = [], []
    t0 = time.time()

    for batch in loader:
        batch = _batch_to_device(batch, device)
        with torch.no_grad():
            logits, y = _forward_any(model, batch, topk_ratio=0.0)
            loss = criterion(logits, y)
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

# =============== one fold ===============
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
        esm2_dir      = str(cache_root / "esm"       / ds_cap),
        molclr_dir    = str(cache_root / "molclr"    / ds_cap),
        chemberta_dir = str(cache_root / "chemberta" / ds_cap),
    )
    dims = CacheDims(esm2=args.d_protein, molclr=args.d_molclr, chemberta=args.d_chem)

    dm = DataModule(
        DMConfig(
            train_csv=train_csv, test_csv=test_csv,
            num_workers=args.workers, batch_size=args.batch_size,
            pin_memory=True, persistent_workers=args.workers>0,
            prefetch_factor=2, drop_last=False,
            sequence=args.sequence
        ),
        cache_dirs=cache_dirs, dims=dims
    )

    train_loader = dm.train_loader()
    test_loader  = dm.test_loader()
    N = len(train_loader.dataset)
    print(f"[INFO] train size = {N} | batch_size = {args.batch_size} | sequence={args.sequence}")

    model = build_model_from_src(dims, args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine(optimizer, total_epochs=args.epochs, warmup_ratio=args.warmup_ratio) if args.scheduler == "cosine" else None
    criterion = make_criterion(args, device, train_csv)

    fold_dir = Path(args.out) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    csv_path = fold_dir / "metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            # include best-thr columns if auto
            if args.auto_thr:
                w.writerow(["epoch","train_loss","test_loss","AUROC","AUPRC","F1","ACC","SEN","MCC",
                            "THR_BEST","F1_BEST","ACC_BEST","SEN_BEST","MCC_BEST",
                            "time_train_s","time_test_s"])
            else:
                w.writerow(["epoch","train_loss","test_loss","AUROC","AUPRC","F1","ACC","SEN","MCC","time_train_s","time_test_s"])

    # resume (shape-safe loading optional)
    start_epoch = 1
    last_ckpt = fold_dir / "last.pth"
    if args.resume and Path(args.resume).exists():
        state = torch.load(str(args.resume), map_location="cpu")
        sd = state.get("state_dict", state)
        msd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
        model.load_state_dict(sd, strict=False)
        try: optimizer.load_state_dict(state.get("optimizer", {}))
        except Exception: pass
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"[RESUME] loaded {args.resume} -> start_epoch={start_epoch} (matched {len(sd)} keys)")
    elif last_ckpt.exists():
        state = torch.load(str(last_ckpt), map_location="cpu")
        sd = state.get("state_dict", state)
        msd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
        model.load_state_dict(sd, strict=False)
        try: optimizer.load_state_dict(state.get("optimizer", {}))
        except Exception: pass
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"[RESUME] auto loaded {last_ckpt} -> start_epoch={start_epoch} (matched {len(sd)} keys)")

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type=="cuda")

    # Stage-A/B gating control
    if hasattr(model, "set_gate_enabled"):
        model.set_gate_enabled(False if (args.stage=="ab" and args.stage_a_epochs>0) else True)
    if hasattr(model, "freeze_gating") and args.stage=="ab":
        model.freeze_gating(False)  # stage-A: allow other parts to learn

    best_score = -1.0
    best_row: Dict[str,float] = {}
    best_epoch = -1
    last_k_states = []  # for averaging last K

    for ep in range(start_epoch, args.epochs + 1):
        # Toggle stage
        if args.stage == "ab" and args.stage_a_epochs > 0 and ep == args.stage_a_epochs + 1:
            if hasattr(model, "set_gate_enabled"):
                model.set_gate_enabled(True)
                print("[STAGE] switch to Stage-B: enable gating")
            if hasattr(model, "freeze_gating"):
                model.freeze_gating(True)  # freeze gate params in Stage-B
                print("[STAGE] freeze gating params")

        # Top-k warmup (linearly from 0 -> target during Stage-B)
        ratio_eff = 0.0
        if args.topk_ratio > 0.0 and hasattr(model, "topk_ratio"):
            if args.stage == "ab" and ep <= args.stage_a_epochs:
                ratio_eff = 0.0
            else:
                t = max(1, args.topk_warmup_epochs)
                step = min(t, max(0, ep - max(1, args.stage_a_epochs)))
                ratio_eff = args.topk_ratio * (step / t)

        tr_loss, tr_t = train_epoch(model, train_loader, device, optimizer, criterion,
                                    tag=f"{ds_lower}/train", ep=ep, ep_total=args.epochs,
                                    gate_budget=args.gate_budget, gate_rho=args.gate_rho,
                                    evi_reg_lambda=args.evi_reg, entropy_reg_lambda=args.entropy_reg,
                                    amp=args.amp, scaler=scaler, topk_ratio=ratio_eff)
        te_loss, prob, y, te_t = test_epoch(model, test_loader, device, criterion)

        # metrics @ fixed thr
        m = compute_metrics(prob, y, thr=args.thr)

        # auto max-F1 threshold
        if args.auto_thr:
            best_thr, m_best = find_best_threshold(prob, y, step=args.thr_scan_step,
                                                   thr_min=args.thr_min, thr_max=args.thr_max)
            print(f"[THR] best F1 = {m_best['f1']:.4f} at thr={best_thr:.3f}")
            # write with extra columns
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([ep, f"{tr_loss:.6f}", f"{te_loss:.6f}",
                            f"{m['auroc']:.6f}", f"{m['auprc']:.6f}", f"{m['f1']:.6f}",
                            f"{m['acc']:.6f}", f"{m['sen']:.6f}", f"{m['mcc']:.6f}",
                            f"{best_thr:.6f}", f"{m_best['f1']:.6f}", f"{m_best['acc']:.6f}",
                            f"{m_best['sen']:.6f}", f"{m_best['mcc']:.6f}",
                            f"{tr_t:.1f}", f"{te_t:.1f}"])
        else:
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([ep, f"{tr_loss:.6f}", f"{te_loss:.6f}",
                            f"{m['auroc']:.6f}", f"{m['auprc']:.6f}", f"{m['f1']:.6f}",
                            f"{m['acc']:.6f}", f"{m['sen']:.6f}", f"{m['mcc']:.6f}",
                            f"{tr_t:.1f}", f"{te_t:.1f}"])

        print(f"[{ds_lower}] fold{fold} ep{ep:03d} | train_loss {tr_loss:.4f} | test_loss {te_loss:.4f} | {fmt1(m)} | time {tr_t:.1f}s/{te_t:.1f}s")

        save_ckpt(fold_dir / "last.pth", model, ep, m, optimizer.state_dict())

        if scheduler is not None:
            scheduler.step()

        sc = m["auroc"] if not math.isnan(m["auroc"]) else m["acc"]
        if sc > best_score:
            best_score = sc
            best_row = dict(m)
            best_epoch = ep
            save_ckpt(fold_dir / "best.pth", model, ep, m, optimizer.state_dict())

        # for last-K average
        if args.save_last_k > 0:
            last_k_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if len(last_k_states) > args.save_last_k:
                last_k_states.pop(0)

    # 保存 best 行
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if args.auto_thr:
            w.writerow([f"best@{best_epoch}", "", "",
                        f"{best_row.get('auroc', float('nan')):.6f}",
                        f"{best_row.get('auprc', float('nan')):.6f}",
                        f"{best_row.get('f1', float('nan')):.6f}",
                        f"{best_row.get('acc', float('nan')):.6f}",
                        f"{best_row.get('sen', float('nan')):.6f}",
                        f"{best_row.get('mcc', float('nan')):.6f}",
                        "", "", "", "", "", "", ""])
        else:
            w.writerow([f"best@{best_epoch}", "", "",
                        f"{best_row.get('auroc', float('nan')):.6f}",
                        f"{best_row.get('auprc', float('nan')):.6f}",
                        f"{best_row.get('f1', float('nan')):.6f}",
                        f"{best_row.get('acc', float('nan')):.6f}",
                        f"{best_row.get('sen', float('nan')):.6f}",
                        f"{best_row.get('mcc', float('nan')):.6f}",
                        "", ""])

    # 平均最后 K 个 epoch 权重
    if args.save_last_k > 0 and len(last_k_states) > 0:
        avg_state = {}
        for k in last_k_states[0]:
            avg_state[k] = sum([st[k] for st in last_k_states]) / float(len(last_k_states))
        model.load_state_dict(avg_state, strict=False)
        save_ckpt(fold_dir / "lastK_avg.pth", model, best_epoch, best_row, optimizer.state_dict())
        print(f"[SAVE] lastK_avg.pth saved over last {len(last_k_states)} epochs.")

    print(f"[{ds_lower}] fold{fold} best | {fmt1(best_row)} (epoch={best_epoch})")
    return best_row

def summarize(rows: List[Dict[str,float]], out_dir: Path):
    keys = ["auroc","auprc","f1","acc","sen","mcc"]
    fold_best_csv = out_dir / "fold_best.csv"
    with open(fold_best_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold"] + [k.upper() for k in keys])
        for i, r in enumerate(rows, 1):
            w.writerow([i] + [f"{r.get(k, float('nan')):.6f}" for k in keys])

    mean = {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys}
    std  = {k: float(np.nanstd ([r.get(k, np.nan) for r in rows])) for k in keys}
    summary_csv = out_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric","mean","std"])
        for k in keys:
            w.writerow([k.upper(), f"{mean[k]:.6f}", f"{std[k]:.6f}"])

    s = " | ".join([f"{k.upper()} {mean[k]:>7.4f}±{std[k]:<7.4f}" for k in keys])
    print(f"[SUMMARY over 5 folds] {s}")
    return mean, std

# =============== CLI ===============
def parse_args():
    ap = argparse.ArgumentParser()
    # Data & IO
    ap.add_argument("--dataset", required=True, help="bindingdb / davis / biosnap（不区分大小写）")
    ap.add_argument("--dataset-dirname", required=True, help="如 bindingdb / davis / biosnap")
    ap.add_argument("--data-root", required=True, help="如 /root/lanyun-tmp")
    ap.add_argument("--out", required=True, help="如 /root/lanyun-tmp/ugca-runs/dataset")
    ap.add_argument("--resume", default="", help="可指定 ckpt 路径；为空时自动找 foldX/last.pth")

    # Train
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--scheduler", default="cosine", choices=["none","cosine"])
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--amp", action="store_true", help="启用 AMP（3090/4090 推荐）")

    # Model dims
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--d-protein", type=int, default=1280)
    ap.add_argument("--d-molclr", type=int, default=300)
    ap.add_argument("--d-chem", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--head-hidden", type=int, default=512)
    ap.add_argument("--mutan-rank", type=int, default=20)
    ap.add_argument("--mutan-dim", type=int, default=256)

    # UGCA layers
    ap.add_argument("--sequence", action="store_true", help="启用 per-token（序列级）模式")
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)

    # Gating
    ap.add_argument("--gate-type", default="evidential", choices=["evidential","mlp"])
    ap.add_argument("--gate-mode", default="evi_x_mu", choices=["evi_x_mu","evi_only"])
    ap.add_argument("--gate-lambda", type=float, default=2.0)
    ap.add_argument("--g-min", type=float, default=0.05)
    ap.add_argument("--smooth-g", action="store_true", help="邻接平滑 gate")

    # Budget reg
    ap.add_argument("--gate-budget", type=float, default=1e-2, help="门控预算正则系数 λb（0 关闭）")
    ap.add_argument("--gate-rho", type=float, default=0.6, help="目标平均开度 ρ（一般 0.6 左右）")

    # Stage A/B
    ap.add_argument("--stage", default="ab", choices=["a","ab"], help="a: 单阶段；ab: A 后开启门控进入 B")
    ap.add_argument("--stage-a-epochs", type=int, default=15, help="阶段A 训练轮数（仅 --stage ab 生效）")

    # Top-k sparsity
    ap.add_argument("--topk-ratio", type=float, default=0.0, help=">0 启用逐层 Top-k 稀疏门控比例")
    ap.add_argument("--topk-warmup-epochs", type=int, default=10, help="Top-k 线性 warm-up 轮数（Stage-B）")

    # Pooling & attention regularization
    ap.add_argument("--pooling", default="attn", choices=["mean","attn"])
    ap.add_argument("--attn-temp", type=float, default=1.0, help="注意力温度（>1 更平滑）")
    ap.add_argument("--entropy-reg", type=float, default=0.0, help="注意力熵正则权重")

    # Loss
    ap.add_argument("--loss-type", default="bce", choices=["bce","bce_weighted","focal"])
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--evi-reg", type=float, default=0.0, help="Evidential 正则权重")

    # Thresholding
    ap.add_argument("--thr", type=float, default=0.5, help="固定阈值（auto 关闭时生效）")
    ap.add_argument("--auto-thr", action="store_true", help="自动网格搜索使 F1 最大的阈值")
    ap.add_argument("--thr-scan-step", type=float, default=0.01)
    ap.add_argument("--thr-min", type=float, default=0.0)
    ap.add_argument("--thr-max", type=float, default=1.0)

    # Save
    ap.add_argument("--save-last-k", type=int, default=5, help="平均最后 K 个 epoch 的权重保存为 lastK_avg.pth")

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
