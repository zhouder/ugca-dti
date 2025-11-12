# src/train.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, math, argparse, importlib, inspect, csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datamodule import DataModule, DMConfig, CacheDirs, CacheDims, _read_smiles_protein_label

# ===== sklearn =====
try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        accuracy_score, recall_score, matthews_corrcoef
    )
    from sklearn.model_selection import GroupKFold
    SKLEARN = True
except Exception:
    SKLEARN = False
    print("[WARN] scikit-learn 不可用，将退化到少量指标。")

# ---------- metrics ----------
def compute_metrics(prob: np.ndarray, y_true: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if SKLEARN:
        pred = (prob >= thr).astype(np.int64)
        out["acc"] = float(accuracy_score(y_true, pred))
        out["sen"] = float(recall_score(y_true, pred))
        out["f1"]  = float(f1_score(y_true, pred))
        out["mcc"] = float(matthews_corrcoef(y_true, pred))
        try:    out["auroc"] = float(roc_auc_score(y_true, prob))
        except Exception: out["auroc"] = float("nan")
        try:    out["auprc"] = float(average_precision_score(y_true, prob))
        except Exception: out["auprc"] = float("nan")
    else:
        pred = (prob >= thr).astype(np.int64)
        out["acc"] = float((pred == y_true).mean())
        out["sen"] = float((pred[y_true == 1] == 1).mean() if (y_true == 1).any() else 0.0)
        out["f1"]  = float("nan")
        out["mcc"] = float("nan")
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out

def fmt1(m: Dict[str, float]) -> str:
    return (f"AUROC {m['auroc']:>7.4f} | AUPRC {m['auprc']:>7.4f} | "
            f"F1 {m['f1']:>7.4f} | ACC {m['acc']:>7.4f} | "
            f"SEN {m['sen']:>7.4f} | MCC {m['mcc']:>7.4f}")

def find_best_threshold(prob: np.ndarray, y_true: np.ndarray, grid: np.ndarray | None = None) -> float:
    """在验证集上搜索使 F1 最大的阈值"""
    if grid is None:
        uniq = np.unique(np.clip(prob, 0.0, 1.0))
        grid = uniq if uniq.size <= 500 else np.linspace(0.01, 0.99, 200)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (prob >= t).astype(np.int64)
        try:
            f1 = f1_score(y_true, pred) if SKLEARN else float("nan")
        except Exception:
            f1 = float("nan")
        if not np.isnan(f1) and f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return float(best_t)

# ---------- model loader ----------
def build_model_from_src(dims: CacheDims, prefer_class: str | None = None, sequence: bool = False) -> nn.Module:
    model_mod = importlib.import_module("src.model")
    if hasattr(model_mod, "build_model"):
        print("[Model] 使用 src.model.build_model(cfg)")
        cfg = {
            "d_protein": int(dims.esm2),
            "d_molclr":  int(dims.molclr),
            "d_chem":    int(dims.chemberta),
            "d_model":   512, "dropout": 0.1, "act": "silu",
            "mutan_rank": 10, "mutan_dim": 512, "head_hidden": 512,
            "sequence": bool(sequence),
            "nhead": 4, "nlayers": 2
        }
        return getattr(model_mod, "build_model")(cfg)

    cand_names = [prefer_class] if prefer_class else []
    cand_names += ["UGCAModel", "UGCASeqModel", "UGCA", "Model"]
    for name in cand_names:
        if not name: continue
        c = getattr(model_mod, name, None)
        if inspect.isclass(c) and issubclass(c, nn.Module):
            print(f"[Model] 使用类 {name}")
            try:
                return c({"d_protein": dims.esm2, "d_molclr": dims.molclr, "d_chem": dims.chemberta})
            except TypeError:
                try: return c(dims)
                except Exception: return c({"d_protein": dims.esm2, "d_molclr": dims.molclr, "d_chem": dims.chemberta})
    raise RuntimeError("在 src/model.py 中没找到可用的模型类。")

# ---------- train / eval ----------
def _batch_to_device(batch, device: torch.device):
    if isinstance(batch, (list, tuple)):
        return tuple(x.to(device) if torch.is_tensor(x) else x for x in batch)
    return batch.to(device) if torch.is_tensor(batch) else batch

def _forward_any(model: nn.Module, batch):
    # V1: (Dp, Dd1, Dc, y)       → logits, y
    # V2: (P, Pm, D, Dm, C, y)   → logits, y
    if len(batch) == 4:
        Dp, Dd1, Dc, y = batch
        logits = model(Dp, Dd1, Dc)
        return logits, y
    else:
        P, Pm, D, Dm, C, y = batch
        logits = model(P, Pm, D, Dm, C)
        return logits, y

def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: optim.Optimizer,
                tag: str, ep: int, ep_total: int, gate_budget: float = 0.0, gate_rho: float = 0.6) -> Tuple[float, float]:
    model.train(True)
    bce = nn.BCEWithLogitsLoss()
    tot = 0.0
    t0 = time.time()
    n_seen = 0
    total_samples = len(loader.dataset)
    pbar = tqdm(total=total_samples, ncols=120, unit="ex",
                desc=f"[{tag}] epoch {ep}/{ep_total}", leave=True, position=0)

    for batch in loader:
        batch = _batch_to_device(batch, device)
        logits, y = _forward_any(model, batch)
        loss = bce(logits, y)

        # 门控预算正则（若模型实现了 last_gates 钩子）
        if gate_budget > 0.0 and hasattr(model, "last_gates"):
            gd, gp = model.last_gates()
            if gd is not None and gp is not None:
                def _mean1d(g): return g.mean(dim=1).mean()
                Lb = ((_mean1d(gd) - gate_rho) ** 2 + (_mean1d(gp) - gate_rho) ** 2) * 0.5
                loss = loss + gate_budget * Lb

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs = y.size(0)
        tot += float(loss.detach().cpu())
        n_seen += bs
        pbar.update(bs)
        if n_seen and (n_seen % (bs * 10) == 0 or n_seen == total_samples):
            pbar.set_postfix_str(f"{n_seen}/{total_samples} ex | loss {tot * 1.0 / (n_seen / bs):.4f}")

    pbar.close()
    return tot / max(1, n_seen / bs), time.time() - t0

def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, float]:
    model.train(False)
    bce = nn.BCEWithLogitsLoss()
    tot = 0.0
    probs, labels = [], []
    t0 = time.time()

    for batch in loader:
        batch = _batch_to_device(batch, device)
        with torch.no_grad():
            logits, y = _forward_any(model, batch)
            loss = bce(logits, y)
        tot += float(loss.detach().cpu())
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    prob = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
    lab  = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.float32)
    return tot / max(1, len(loader)), prob, lab, time.time() - t0

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="如 DAVIS / BindingDB / BioSNAP（大小写不敏感）")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--model-class", default="", help="通常不用；除非你显式选类（如 UGCASeqModel）")
    ap.add_argument("--seed", type=int, default=42)

    # 模型形态与门控
    ap.add_argument("--sequence", action="store_true", help="启用 per-token（序列级）模式")
    ap.add_argument("--gate-budget", type=float, default=0.0, help="门控预算正则系数 λb（0 关闭）")
    ap.add_argument("--gate-rho", type=float, default=0.6, help="目标平均开度 ρ（一般 0.6 左右）")

    # 冷启动设置：默认 5 折；整体比例默认 0.7/0.1/0.2
    ap.add_argument("--split-mode", choices=["cold-protein", "cold-drug"], default="cold-protein")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--overall-train", type=float, default=0.7)
    ap.add_argument("--overall-val",   type=float, default=0.1)
    ap.add_argument("--thr", default="auto",
                    help="决策阈值；'auto'=在验证集上选择使F1最大的阈值；或显式给出如 0.35")

    # —— 新增：稳定命名 & 断点续训 —— #
    ap.add_argument("--out-dir", default="",
                    help="可选：自定义输出根目录；不设则默认 /root/lanyun-tmp/ugca-runs/<DATASET>-cold-{protein|drug}[-seq][-suffix]")
    ap.add_argument("--suffix", default="", help="可选：给目录添加自定义后缀，如 '-seq'、'-s42'")
    ap.add_argument("--resume", action="store_true", help="断点续训：已完成的折自动跳过；存在 last.pth 时从其 epoch+1 继续")
    return ap.parse_args()

# ---------- helpers for split ----------
def sample_val_from_pool_by_groups(pool_idx: np.ndarray, groups: np.ndarray,
                                   val_frac_in_pool: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """在候选训练池(不含test)里，按 group 采样一部分作为 val（组不交叠）"""
    pool_groups = groups[pool_idx]
    uniq = np.unique(pool_groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val_groups = max(1, int(round(val_frac_in_pool * len(uniq))))
    val_set = set(uniq[:n_val_groups])
    mask = np.array([g in val_set for g in pool_groups], dtype=bool)
    va_idx = pool_idx[mask]
    tr_idx = pool_idx[~mask]
    return tr_idx, va_idx

# ---------- main ----------
if __name__ == "__main__":
    args = parse_args()
    print(f"[ENV] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device: {'cuda' if use_cuda else 'cpu'} | cuda_available: {use_cuda}")
    if use_cuda:
        try: print("gpu:", torch.cuda.get_device_name(0))
        except Exception: pass

    # ---- 自动路径 ----
    ds_lower = args.dataset.lower()
    ds_cap   = {"bindingdb":"BindingDB", "davis":"DAVIS", "biosnap":"BioSNAP"}.get(ds_lower, args.dataset)
    data_root = Path("/root/lanyun-tmp")
    all_csv   = data_root / ds_cap / "all.csv"
    if not all_csv.exists():
        raise FileNotFoundError(f"未找到 {all_csv}，请确认你已把 all.csv 放到该路径。")

    cache_root = data_root / "cache"
    cache_dirs = CacheDirs(
        esm_dir      = str(cache_root / "esm2"     / ds_cap),   # 自动在 esm2/esm 间回退
        molclr_dir   = str(cache_root / "molclr"   / ds_cap),
        chemberta_dir= str(cache_root / "chemberta"/ ds_cap),
    )
    dims = CacheDims(esm2=1280, molclr=300, chemberta=384)

    # ---- 读 all.csv ----
    smiles, proteins, labels = _read_smiles_protein_label(str(all_csv))
    N = len(labels)
    print(f"[ALL] loaded: {N} rows from {all_csv}")

    # ---- 外层：按 group 的 K 折，用于 test ----
    groups_all = np.asarray(proteins if args.split_mode == "cold-protein" else smiles)
    gkf = GroupKFold(n_splits=args.cv_folds if args.cv_folds > 1 else 5)
    outer_splits = list(gkf.split(np.arange(N), groups=groups_all))
    overall_test = 1.0 / len(outer_splits)
    # 在候选训练池(80%)里抽取 val，使整体比例达到 overall_val（默认 0.1）
    val_frac_in_pool = args.overall_val / (1.0 - overall_test)  # 0.1 / 0.8 = 0.125
    print(f"[SPLIT] 模式={args.split_mode} | 折数={len(outer_splits)} | 目标整体比例 train:val:test = {args.overall_train}:{args.overall_val}:{overall_test:.1f}")
    print(f"[SPLIT] 实施策略：每折 test=1/5；在其余 4/5 的候选池按组随机抽取 {val_frac_in_pool*100:.1f}% 的组做 val，其余做 train（保持组不交叠）")

    # ---- 输出主目录（稳定命名，不含时间戳）----
    split_tag = "cold-protein" if args.split_mode == "cold-protein" else "cold-drug"
    base = args.out_dir if args.out_dir else f"/root/lanyun-tmp/ugca-runs/{ds_cap}-{split_tag}"
    if args.sequence:
        base += "-seq"
    if args.suffix:
        base += (args.suffix if args.suffix.startswith("-") else f"-{args.suffix}")
    out_dir = Path(base)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[OUT]", out_dir)

    # ---- 逐折训练 ----
    test_metrics_all: List[Dict[str, float]] = []
    for t, (_, te_idx) in enumerate(outer_splits, 1):
        te_idx = np.asarray(te_idx)
        pool_idx = np.setdiff1d(np.arange(N), te_idx)
        tr_idx, va_idx = sample_val_from_pool_by_groups(pool_idx, groups_all, val_frac_in_pool, seed=args.seed + t)

        def slice_trip(idx):
            return [smiles[i] for i in idx], [proteins[i] for i in idx], [labels[i] for i in idx]
        tr = slice_trip(tr_idx); va = slice_trip(va_idx); te = slice_trip(te_idx)

        # 打印比例
        def _stat(ix):
            ys = [labels[i] for i in ix]; pos = sum(1 for v in ys if float(v) >= 0.5)
            return len(ix), pos, (pos/len(ix) if len(ix) else 0.0)
        ntr, ptr, rtr = _stat(tr_idx); nva, pva, rva = _stat(va_idx); nte, pte, rte = _stat(te_idx)
        print(f"[FOLD{t}] train={ntr} (pos={ptr}, {rtr:.3f}) | val={nva} (pos={pva}, {rva:.3f}) | test={nte} (pos={pte}, {rte:.3f})")

        # ---- Data & loaders ----
        dm = DataModule(
            DMConfig(
                train_data=tr, val_data=va, test_data=te,
                num_workers=args.workers, batch_size=args.batch_size,
                pin_memory=True, persistent_workers=args.workers>0,
                prefetch_factor=2, drop_last=False,
                sequence=args.sequence
            ),
            cache_dirs=cache_dirs, dims=dims
        )
        train_loader = dm.train_loader()
        val_loader   = dm.val_loader()
        test_loader  = dm.test_loader()

        # ---- Model / Optim ----
        model = build_model_from_src(dims, args.model_class, sequence=args.sequence).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        fold_dir = out_dir / f"fold{t}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        ckpt_best = fold_dir / "best.pth"
        ckpt_last = fold_dir / "last.pth"
        csv_path  = fold_dir / "metrics.csv"

        # 断点续训：若 summary.csv 存在 → 视为该折完成；否则若 last.pth 存在 → 从其 epoch+1 继续
        if args.resume:
            if (fold_dir / "summary.csv").exists():
                print(f"[RESUME] fold{t} 已完成，跳过。")
                test_metrics_all.append({k: float("nan") for k in ["auroc","auprc","f1","acc","sen","mcc"]})
                continue

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_loss","val_loss","AUROC","AUPRC","F1","ACC","SEN","MCC","time_train_s","time_val_s","thr"])

        best_score = -1.0
        best_epoch = -1
        best_metrics = {}
        best_thr = 0.5
        start_epoch = 1

        if args.resume and ckpt_last.exists():
            sd = torch.load(str(ckpt_last), map_location="cpu")
            try:
                model.load_state_dict(sd["state_dict"])
                if "optimizer" in sd and sd["optimizer"]:
                    optimizer.load_state_dict(sd["optimizer"])
            except Exception:
                pass
            start_epoch = int(sd.get("epoch", 0)) + 1
            best_thr    = float(sd.get("best_thr", best_thr))
            best_score  = float(sd.get("best_score", best_score))
            best_epoch  = int(sd.get("best_epoch", best_epoch))
            best_metrics= sd.get("best_metrics", best_metrics)
            print(f"[RESUME] fold{t} 从 epoch {start_epoch} 继续")

        for ep in range(start_epoch, args.epochs + 1):
            tr_loss, tr_t = train_epoch(model, train_loader, device, optimizer,
                                        tag=f"{ds_lower}/train/fold{t}", ep=ep, ep_total=args.epochs,
                                        gate_budget=args.gate_budget, gate_rho=args.gate_rho)
            va_loss, prob, y, va_t = eval_epoch(model, val_loader, device)
            thr_now = find_best_threshold(prob, y) if (str(args.thr).lower() == "auto") else float(args.thr)
            m = compute_metrics(prob, y, thr=thr_now)

            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([ep, f"{tr_loss:.6f}", f"{va_loss:.6f}",
                            f"{m['auroc']:.6f}", f"{m['auprc']:.6f}", f"{m['f1']:.6f}",
                            f"{m['acc']:.6f}", f"{m['sen']:.6f}", f"{m['mcc']:.6f}",
                            f"{tr_t:.1f}", f"{va_t:.1f}", f"{thr_now:.6f}"])
            print(f"[{ds_lower}/fold{t}] ep{ep:03d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | {fmt1(m)} | thr {thr_now:.3f}")

            score = (m["auroc"] if not math.isnan(m["auroc"]) else m["acc"])
            if score > best_score:
                best_score  = score
                best_epoch  = ep
                best_metrics= dict(m)
                best_thr    = float(thr_now)
                torch.save({"epoch": ep, "state_dict": model.state_dict(),
                            "metrics": m, "optimizer": optimizer.state_dict()}, str(ckpt_best))

            # 保存 last.pth（每个 epoch 都覆盖）
            torch.save({
                "epoch": ep,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_thr": best_thr,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
            }, str(ckpt_last))

        print(f"[VAL/fold{t}] best@epoch={best_epoch} | thr*={best_thr:.4f} | {fmt1(best_metrics)}")

        # ---- 测试（仅一次；用 thr*）----
        if ckpt_best.exists():
            sd = torch.load(str(ckpt_best), map_location="cpu")
            model.load_state_dict(sd["state_dict"])
        te_loss, prob, y, te_t = eval_epoch(model, test_loader, device)
        te_m = compute_metrics(prob, y, thr=best_thr)
        print(f"[TEST/fold{t}] thr={best_thr:.4f} | {fmt1(te_m)} | loss {te_loss:.4f} | time {te_t:.1f}s")

        with open(fold_dir / "summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["VAL_BEST_EPOCH", best_epoch])
            w.writerow(["VAL_BEST_THR",   f"{best_thr:.6f}"])
            for k, v in best_metrics.items():
                w.writerow([f"VAL_{k.upper()}", f"{v:.6f}"])
            for k, v in te_m.items():
                w.writerow([f"TEST_{k.upper()}", f"{v:.6f}"])
        test_metrics_all.append(te_m)

    # ---- 汇总五折 ----
    keys = ["auroc","auprc","f1","acc","sen","mcc"]
    mean = {k: float(np.mean([m[k] for m in test_metrics_all])) for k in keys}
    std  = {k: float(np.std ([m[k] for m in test_metrics_all])) for k in keys}
    with open(out_dir / "cv_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric","mean","std"])
        for k in keys: w.writerow([k.upper()],)
        for k in keys: w.writerow([k.upper(), f"{mean[k]:.6f}", f"{std[k]:.6f}"])
    print("[CV] " + " | ".join([f"{k.upper()} {mean[k]:.4f}±{std[k]:.4f}" for k in keys]))