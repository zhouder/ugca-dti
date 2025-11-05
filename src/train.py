# src/train.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.datamodule import DataModule, DMConfig, CacheDirs, CacheDims

# ----------------- 配置解析 -----------------
def load_yaml_like(path: str) -> Dict[str, Any]:
    """
    简单 YAML/JSON 读取器：只支持 k: v / 缩进两层内的 dict/list。
    你也可以直接用 pyyaml；为了最少依赖，这里用一个最小实现。
    """
    import yaml  # 如果你的环境没有，可以 `pip install pyyaml`
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ----------------- 简易模型（占位） -----------------
class SimpleMLP(nn.Module):
    def __init__(self, d1: int, d2: int, d3: int, d_h: int = 256):
        super().__init__()
        in_dim = d1 + d2 + d3
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_h),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(d_h, d_h),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(d_h, 1),
        )

    def forward(self, v1, v2, v3):
        x = torch.cat([v1, v2, v3], dim=-1)
        return self.net(x).squeeze(-1)

def maybe_import_build_model():
    for name in ("src.model", "model"):
        try:
            mod = __import__(name, fromlist=["build_model"])
            return getattr(mod, "build_model")
        except Exception:
            continue
    return None

# ----------------- 训练主程 -----------------
def train_one_fold(cfg: Dict[str, Any], ds: str, fold: int, device: torch.device):
    data_cfg = cfg["data"]
    cache_cfg = cfg["cache"]
    dims_cfg  = cfg["cache_dims"]
    train_csv = data_cfg["train_csv_tmpl"].replace("${dataset}", ds).replace("${fold}", str(fold))
    test_csv  = data_cfg["test_csv_tmpl" ].replace("${dataset}", ds).replace("${fold}", str(fold))

    print("=== dataset:", ds, "===")
    print("[paths] train=", train_csv)
    print("[paths] test =", test_csv)

    dm = DataModule(
        DMConfig(
            train_csv=train_csv,
            test_csv=test_csv,
            num_workers=int(data_cfg.get("num_workers", 8)),
            batch_size=int(data_cfg.get("batch_size", 64)),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=False,
        ),
        CacheDirs(
            esm2_dir=cache_cfg["esm2_dir"],
            molclr_dir=cache_cfg["molclr_dir"],
            chemberta_dir=cache_cfg["chemberta_dir"],
        ),
        CacheDims(
            esm2=int(dims_cfg["esm2"]),
            molclr=int(dims_cfg["molclr"]),
            chemberta=int(dims_cfg["chemberta"]),
        ),
    )

    train_loader = dm.train_loader()
    test_loader  = dm.test_loader()

    # 构建模型
    build_model = maybe_import_build_model()
    if build_model is not None:
        model = build_model(cfg["model"]).to(device)
    else:
        print("[Model] use SimpleMLP placeholder")
        d1, d2, d3 = dm.dims.esm2, dm.dims.molclr, dm.dims.chemberta
        model = SimpleMLP(d1, d2, d3, d_h=int(cfg["model"].get("d", 256))).to(device)

    # 优化器 / 调度
    lr = float(cfg["train"].get("lr", 3e-4))
    wd = float(cfg["train"].get("weight_decay", 1e-4))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    total_epochs = int(cfg["train"].get("total_epochs", 50))
    use_amp = (str(cfg.get("precision", "amp")) == "amp")

    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, total_epochs + 1):
        pbar = enumerate(train_loader, 1)
        for step, (v1, v2, v3, y) in pbar:
            v1, v2, v3, y = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True), v3.to(device, non_blocking=True), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(v1, v2, v3)
                    loss = bce(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(v1, v2, v3)
                loss = bce(logits, y)
                loss.backward()
                optimizer.step()

            if step % 5 == 0:
                print(f"[{ds}] fold{fold} epoch{epoch}/{total_epochs} | step {step}/{len(train_loader)} | loss={loss.item():.4f}")

    # 一个非常简易的评估（仅示范）
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for v1, v2, v3, y in test_loader:
            v1, v2, v3 = v1.to(device), v2.to(device), v3.to(device)
            logits = model(v1, v2, v3)
            prob = torch.sigmoid(logits).cpu().numpy()
            ys.append(y.numpy())
            ps.append(prob)
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    # 打个概览
    print(f"[{ds}] test mean prob = {ps.mean():.4f} | pos ratio (label) = {ys.mean():.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True, choices=["DAVIS", "BindingDB", "BioSNAP"])
    ap.add_argument("--fold", type=int, default=1)
    args = ap.parse_args()

    cfg = load_yaml_like(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  |  cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    train_one_fold(cfg, args.dataset, int(args.fold), device)

if __name__ == "__main__":
    main()
