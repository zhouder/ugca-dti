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
from tqdm import tqdm

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
    out["sen"] = float(recall_score(y_true, pred))    if SKLEARN else float((pred[y_true == 1] == 1).mean() if (y_true == 1).any() else 0.0)
    out["f1"]  = float(f1_score(y_true, pred))        if SKLEARN else float("nan")
    out["mcc"] = float(matthews_corrcoef(y_true, pred)) if SKLEARN else float("nan")
    try:    out["auroc"] = float(roc_auc_score(y_true, prob)) if SKLEARN else float("nan")
    except Exception: out["auroc"] = float("nan")
    try:    out["auprc"] = float(average_precision_score(y_true, prob)) if SKLEARN else float("nan")
    except Exception: out["auprc"] = float("nan")
    return out

def fmt1(m: Dict[str, float]) -> str:
    return (f"AUROC {m['auroc']:>7.4f} | AUPRC {m['auprc']:>7.4f} | "
            f"F1 {m['f1']:>7.4f} | ACC {m['acc']:>7.4f} | "
            f"SEN {m['sen']:>7.4f} | MCC {m['mcc']:>7.4f}")

# ===== 模型装载（优先 build_model(cfg)） =====
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
            # V2 相关
            "sequence": bool(sequence),
            "nhead": 4, "nlayers": 2,
            # 门控默认（需要时可在这里修改，也可在你仓库里接 CLI/配置）
            "gate_type": "evidential",
            "gate_lambda": 2.0,
            "topk_ratio": 0.0,
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

# ===== 训练/测试 =====
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

def _forward_any(model: nn.Module, batch_tensors):
    if len(batch_tensors) == 4:
        v1, v2, v3, y = batch_tensors
        logits = model(v1, v2, v3)
        return logits, y
    else:
        P, Pm, D, Dm, C, y = batch_tensors
        # UGCASeqModel.forward(P, Pm, D, Dm, C)
        logits = model(P, Pm, D, Dm, C)
        return logits, y

def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
                optimizer: optim.Optimizer, tag: str, ep: int, ep_total: int,
                gate_budget: float = 0.0, gate_rho: float = 0.6,
                amp: bool = True, scaler: Optional["torch.cuda.amp.GradScaler"] = None
                ) -> Tuple[float, float]:
    model.train(True)
    bce = nn.BCEWithLogitsLoss()
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
            logits, y = _forward_any(model, batch)
            loss = bce(logits, y)

            # ===== 门控预算正则（可选，V2 有效；V1 返回 None） =====
            if gate_budget > 0.0 and hasattr(model, "last_gates"):
                gd, gp = model.last_gates()
                if gd is not None and gp is not None:
                    # 对每个样本计算 token 级均值，再对 batch 求均值
                    def _mean1d(g): return g.mean(dim=1).mean()
                    Lb = ((_mean1d(gd) - gate_rho) ** 2 + (_mean1d(gp) - gate_rho) ** 2) * 0.5
                    loss = loss + gate_budget * Lb

        optimizer.zero_grad(set_to_none=True)
        if amp and scaler is not None and device.type=="cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
        else:
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

def test_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, float]:
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
        # 传 esm，DataModule 内部会在 esm/esm2 之间自动回退
        esm2_dir      = str(cache_root / "esm"       / ds_cap),
        molclr_dir    = str(cache_root / "molclr"    / ds_cap),
        chemberta_dir = str(cache_root / "chemberta" / ds_cap),
    )
    dims = CacheDims(esm2=1280, molclr=300, chemberta=384)

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

    model = build_model_from_src(dims, args.model_class, sequence=args.sequence).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    fold_dir = Path(args.out) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    csv_path = fold_dir / "metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_loss","test_loss","AUROC","AUPRC","F1","ACC","SEN","MCC","time_train_s","time_test_s"])

    def score_for_best(m: Dict[str,float]) -> float:
        return m["auroc"] if not math.isnan(m["auroc"]) else m["acc"]

    # 断点续训
    start_epoch = 1
    if args.resume:
        ck = Path(args.resume)
        if ck.is_file():
            state = torch.load(str(ck), map_location="cpu")
            model.load_state_dict(state.get("state_dict", state), strict=False)
            try: optimizer.load_state_dict(state.get("optimizer", {}))
            except Exception: pass
            start_epoch = int(state.get("epoch", 0)) + 1
            print(f"[RESUME] loaded {ck} -> start_epoch={start_epoch}")
    else:
        last = fold_dir / "last.pth"
        if last.exists():
            state = torch.load(str(last), map_location="cpu")
            model.load_state_dict(state.get("state_dict", state), strict=False)
            try: optimizer.load_state_dict(state.get("optimizer", {}))
            except Exception: pass
            start_epoch = int(state.get("epoch", 0)) + 1
            print(f"[RESUME] auto loaded {last} -> start_epoch={start_epoch}")

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type=="cuda")

    # 两阶段：到 A 期结束后冻结门控（仅 V2 有效）
    def _maybe_freeze_after_a(ep_now: int):
        if args.stage == "ab" and args.stage_a_epochs > 0 and ep_now == args.stage_a_epochs + 1:
            if hasattr(model, "freeze_gating"):
                print("[STAGE] switch to Stage-B: freeze gating")
                model.freeze_gating(True)

    best_score = -1.0
    best_row: Dict[str,float] = {}
    best_epoch = -1

    for ep in range(start_epoch, args.epochs + 1):
        _maybe_freeze_after_a(ep)
        tr_loss, tr_t = train_epoch(model, train_loader, device, optimizer,
                                    tag=f"{ds_lower}/train", ep=ep, ep_total=args.epochs,
                                    gate_budget=args.gate_budget, gate_rho=args.gate_rho,
                                    amp=args.amp, scaler=scaler)
        te_loss, prob, y, te_t = test_epoch(model, test_loader, device)
        m = compute_metrics(prob, y)

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, f"{tr_loss:.6f}", f"{te_loss:.6f}",
                        f"{m['auroc']:.6f}", f"{m['auprc']:.6f}", f"{m['f1']:.6f}",
                        f"{m['acc']:.6f}", f"{m['sen']:.6f}", f"{m['mcc']:.6f}",
                        f"{tr_t:.1f}", f"{te_t:.1f}"])

        print(f"[{ds_lower}] fold{fold} ep{ep:03d} | train_loss {tr_loss:.4f} | test_loss {te_loss:.4f} | {fmt1(m)} | time {tr_t:.1f}s/{te_t:.1f}s")

        save_ckpt(fold_dir / "last.pth", model, ep, m, optimizer.state_dict())

        sc = score_for_best(m)
        if sc > best_score:
            best_score = sc
            best_row = dict(m)
            best_epoch = ep
            save_ckpt(fold_dir / "best.pth", model, ep, m, optimizer.state_dict())

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
                    "", ""])

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

    # 计算 mean/std 并写 summary
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
    ap.add_argument("--dataset-dirname", required=True, help="如 bindingdb / davis / biosnap")
    ap.add_argument("--data-root", required=True, help="如 /root/lanyun-tmp")
    ap.add_argument("--out", required=True, help="如 /root/lanyun-tmp/ugca-runs/bindingdb")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--resume", default="", help="可指定 ckpt 路径；为空时会自动找 foldX/last.pth")
    ap.add_argument("--model-class", default="", help="通常不用；除非你显式选类（如 UGCASeqModel）")
    # 新增：V2 序列级与门控预算
    ap.add_argument("--sequence", action="store_true", help="启用 per-token（序列级）模式")
    ap.add_argument("--gate-budget", type=float, default=0.0, help="门控预算正则系数 λb（0 关闭）")
    ap.add_argument("--gate-rho", type=float, default=0.6, help="目标平均开度 ρ（一般 0.6 左右）")
    # AMP 与两阶段
    ap.add_argument("--amp", action="store_true", help="启用 AMP（3090/4090 推荐）")
    ap.add_argument("--stage", default="a", choices=["a","ab"], help="a: 单阶段；ab: A 后冻结门控进入 B")
    ap.add_argument("--stage-a-epochs", type=int, default=0, help="阶段A 训练轮数（仅 --stage ab 生效）")
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
