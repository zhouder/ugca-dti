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

# ===== 你的数据模块 =====
from src.datamodule import DataModule, DMConfig, CacheDirs, CacheDims

# ===== sklearn 指标 =====
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
    out["sen"] = float(recall_score(y_true, pred)) if SKLEARN else float((pred[y_true == 1] == 1).mean() if (y_true == 1).any() else 0.0)
    out["f1"]  = float(f1_score(y_true, pred)) if SKLEARN else float("nan")
    out["mcc"] = float(matthews_corrcoef(y_true, pred)) if SKLEARN else float("nan")
    try:
        out["auroc"] = float(roc_auc_score(y_true, prob)) if SKLEARN else float("nan")
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, prob)) if SKLEARN else float("nan")
    except Exception:
        out["auprc"] = float("nan")
    return out

def fmt(m: Dict[str, float]) -> str:
    return (f"AUROC {m['auroc']:.4f} AUPRC {m['auprc']:.4f} F1 {m['f1']:.4f} "
            f"ACC {m['acc']:.4f} SEN {m['sen']:.4f} MCC {m['mcc']:.4f}")

# ===== 模型加载：优先使用 build_model(cfg)，否则用 UGCAModel =====
def build_model_from_src(dims: CacheDims, prefer_class: str | None = None) -> nn.Module:
    model_mod = importlib.import_module("src.model")  # 对应 src/model.py

    # 优先工厂函数
    if hasattr(model_mod, "build_model"):
        print("[Model] 使用 src.model.build_model(cfg) 构建模型")
        cfg = {
            "d_protein": int(dims.esm2),
            "d_molclr":  int(dims.molclr),
            "d_chem":    int(dims.chemberta),
            # 下方是默认值；如需在 CLI 里开放，可在 argparse 里加参数再传进来
            "d_model":   512,
            "dropout":   0.1,
            "act":       "silu",
            "mutan_rank": 10,
            "mutan_dim":  512,
            "head_hidden": 512,
        }
        return getattr(model_mod, "build_model")(cfg)

    # 否则找 UGCAModel；如果用户传了 prefer_class，也优先按那个找
    cand_names = [prefer_class] if prefer_class else []
    cand_names += ["UGCAModel", "UGCA", "Model"]
    for name in cand_names:
        if not name:
            continue
        c = getattr(model_mod, name, None)
        if inspect.isclass(c) and issubclass(c, nn.Module):
            print(f"[Model] 使用类 {name}")
            try:
                # 你的 UGCAModel 期望传入 ModelCfg，但也兼容 dict
                return c({"d_protein": dims.esm2, "d_molclr": dims.molclr, "d_chem": dims.chemberta})
            except TypeError:
                # 有些实现是直接接收 dims / 或三元组，这里尽量兜底
                try:
                    return c(dims)
                except Exception:
                    return c({"d_protein": dims.esm2, "d_molclr": dims.molclr, "d_chem": dims.chemberta})

    # 自动挑选一个 nn.Module 子类（避开 LayerNorm/Block/Head）
    candidates = [
        (name, obj) for name, obj in inspect.getmembers(model_mod, inspect.isclass)
        if obj.__module__ == model_mod.__name__ and issubclass(obj, nn.Module)
    ]
    for name, cls in candidates:
        if any(tok in name.lower() for tok in ["norm", "block", "head", "layer", "unit"]):
            continue
        print(f"[INFO] 未找到指定类，自动使用 {name}")
        try:
            return cls({"d_protein": dims.esm2, "d_molclr": dims.molclr, "d_chem": dims.chemberta})
        except TypeError:
            return cls(dims)
    raise RuntimeError("在 src/model.py 中没找到可用的模型类。")

# ===== 一个 epoch =====
def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, train: bool,
              optimizer: optim.Optimizer | None) -> Tuple[float, np.ndarray, np.ndarray, float]:
    t0 = time.time()
    model.train(train)
    bce = nn.BCEWithLogitsLoss()
    tot = 0.0
    probs, labels = [], []

    for i, (v1, v2, v3, y) in enumerate(loader, 1):
        v1, v2, v3, y = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True), v3.to(device, non_blocking=True), y.to(device)
        if i == 1:
            try:
                mdev = next(model.parameters()).device
            except StopIteration:
                mdev = torch.device("cpu")
            print(f"[DEBUG] device check | model={mdev} | v1={v1.device} v2={v2.device} v3={v3.device} y={y.device}")

        logits = model(v1, v2, v3)
        loss = bce(logits, y)
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

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
        esm2_dir      = str(cache_root / "esm2"      / ds_cap),
        molclr_dir    = str(cache_root / "molclr"    / ds_cap),
        chemberta_dir = str(cache_root / "chemberta" / ds_cap),
    )

    # 关键修正：CacheDims 需要显式传入三维度
    dims = CacheDims(esm2=1280, molclr=300, chemberta=384)

    dm = DataModule(
        DMConfig(
            train_csv=train_csv, test_csv=test_csv,
            num_workers=args.workers, batch_size=args.batch_size,
            pin_memory=True, persistent_workers=args.workers>0,
            prefetch_factor=2, drop_last=False
        ),
        cache_dirs=cache_dirs,
        dims=dims
    )

    train_loader = dm.train_loader()
    test_loader  = dm.test_loader()
    N = len(train_loader.dataset)
    print(f"[INFO] train size = {N} | batch_size = {args.batch_size}")

    # 模型：优先 build_model(cfg)
    model = build_model_from_src(dims, args.model_class).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    fold_dir = Path(args.out) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    csv_path = fold_dir / "metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_loss","test_loss","AUROC","AUPRC","F1","ACC","SEN","MCC","time_train_s","time_test_s"])

    start_epoch = 1
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[RESUME] loaded {args.resume} -> start from epoch {start_epoch}")

    def score_for_best(m: Dict[str,float]) -> float:
        return m["auroc"] if not math.isnan(m["auroc"]) else m["acc"]

    best_score = -1.0
    best_row: Dict[str,float] = {}

    pbar = tqdm(range(start_epoch, args.epochs + 1), ncols=120, desc=f"[{ds_lower}] fold{fold} epochs")
    for ep in pbar:
        tr_loss, _, _, tr_t = run_epoch(model, train_loader, device, train=True,  optimizer=optimizer)
        te_loss, prob, y, te_t = run_epoch(model, test_loader,  device, train=False, optimizer=None)
        m = compute_metrics(prob, y)

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, f"{tr_loss:.6f}", f"{te_loss:.6f}",
                        f"{m['auroc']:.6f}", f"{m['auprc']:.6f}", f"{m['f1']:.6f}",
                        f"{m['acc']:.6f}", f"{m['sen']:.6f}", f"{m['mcc']:.6f}",
                        f"{tr_t:.1f}", f"{te_t:.1f}"])
        pbar.set_postfix_str(f"train_loss {tr_loss:.4f} | test_loss {te_loss:.4f} | {fmt(m)} | time train {tr_t:.1f}s test {te_t:.1f}s")

        save_ckpt(fold_dir / "last.pth", model, ep, m, optimizer.state_dict())

        sc = score_for_best(m)  # AUROC 优先
        if sc > best_score:
            best_score = sc
            best_row = dict(m)
            save_ckpt(fold_dir / "best.pth", model, ep, m, optimizer.state_dict())

        torch.save(model.state_dict(), str(fold_dir / f"epoch{ep:03d}.pth"))

    print(f"[{ds_lower}] fold{fold} best: {fmt(best_row)}")
    return best_row

def summarize(rows: List[Dict[str,float]]):
    keys = ["auroc","auprc","f1","acc","sen","mcc"]
    mean = {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys}
    std  = {k: float(np.nanstd ([r.get(k, np.nan) for r in rows])) for k in keys}
    s = " | ".join([f"{k.upper()} {mean[k]:.4f}±{std[k]:.4f}" for k in keys])
    print(f"[SUMMARY over 5 folds] {s}")
    return mean, std

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="bindingdb / davis / biosnap（不区分大小写）")
    ap.add_argument("--dataset-dirname", required=True, help="如 bindingdb_k5 / davis_k5 / biosnap_k5")
    ap.add_argument("--data-root", required=True, help="如 /root/lanyun-tmp")
    ap.add_argument("--out", required=True, help="如 /root/lanyun-tmp/ugca-runs/bindingdb")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resume", default="")
    ap.add_argument("--model-class", default="UGCAModel", help="src/model.py 的类名（若无需，可忽略；默认 UGCAModel）")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # CUDA 可见性提示（帮助排查被强制到 CPU 的情况）
    print(f"[ENV] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device: {'cuda' if use_cuda else 'cpu'} | cuda_available: {use_cuda}")
    if use_cuda:
        try:
            print("gpu:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)

    all_best: List[Dict[str,float]] = []
    for fold in range(1, 6):
        best = run_one_fold(args, fold, device)
        all_best.append(best)
    summarize(all_best)
