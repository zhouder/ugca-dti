# src/model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====== 小工具 ======
def _act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    return {
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
        "tanh": nn.Tanh(),
    }.get(name, nn.ReLU(inplace=True))


class LayerNorm1d(nn.Module):
    """对最后一维做 LayerNorm（float32 计算更稳）"""
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(d, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            return self.ln(x.float()).type_as(x)
        return self.ln(x)


# ====== MUTAN（低秩双线性池化）======
class Mutan(nn.Module):
    """
    设 rank=R，先把 x,y 投到 R 组子空间：
        X_i = Wxi x,  Y_i = Wyi y
    然后逐组做 Hadamard：Z_i = tanh(X_i) ⊙ tanh(Y_i)
    最后拼接/加权聚合成 fused。
    """
    def __init__(self, dx: int, dy: int, dz: int, rank: int = 10, dropout: float = 0.1, act: str = "tanh"):
        super().__init__()
        self.rank = rank
        self.proj_x = nn.Sequential(nn.Linear(dx, dz * rank, bias=False), nn.Dropout(dropout))
        self.proj_y = nn.Sequential(nn.Linear(dy, dz * rank, bias=False), nn.Dropout(dropout))
        self.out = nn.Sequential(LayerNorm1d(dz), nn.Dropout(dropout))
        self.nonlin = _act(act)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: [B, dx], y: [B, dy]
        bx = self.proj_x(x)       # [B, dz*R]
        by = self.proj_y(y)       # [B, dz*R]
        B, _ = bx.shape
        R = self.rank
        # [B,R,dz]
        bx = bx.view(B, R, -1)
        by = by.view(B, R, -1)
        z = self.nonlin(bx) * self.nonlin(by)   # 逐组 Hadamard
        z = z.sum(dim=1)                        # 聚合 R 组 -> [B,dz]
        return self.out(z)


# ====== UGCA（门控 + 互导向聚合）======
class UGCAUnit(nn.Module):
    """
    对两路输入 a,b（如 molclr 与 chemberta 或 drug 与 protein）做门控聚合：
      g = σ( Wg [a||b] )
      a' = LayerNorm( a + Drop( W1 a ) )
      b' = LayerNorm( b + Drop( W2 b ) )
      out = g * a' + (1-g) * b'
    可堆叠多层，配合跨路的“引导信息”（这里简化为残差+LayerNorm）。
    """
    def __init__(self, da: int, db: int, d: int, dropout: float = 0.1, act: str = "silu"):
        super().__init__()
        self.pa = nn.Linear(da, d, bias=False)
        self.pb = nn.Linear(db, d, bias=False)
        self.ln_a = LayerNorm1d(d)
        self.ln_b = LayerNorm1d(d)
        self.ff_a = nn.Sequential(nn.Linear(d, d), _act(act), nn.Dropout(dropout))
        self.ff_b = nn.Sequential(nn.Linear(d, d), _act(act), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # 投影到同维
        A = self.pa(a)
        B = self.pb(b)
        # 各自做一层前馈 + 残差
        A2 = self.ln_a(A + self.ff_a(A))
        B2 = self.ln_b(B + self.ff_b(B))
        # 门控融合
        g = self.gate(torch.cat([A2, B2], dim=-1))
        out = g * A2 + (1.0 - g) * B2
        return out


# ====== 顶层模型 ======
@dataclass
class ModelCfg:
    # 维度（来自缓存特征）
    d_protein: int = 1280   # ESM2
    d_molclr: int = 300
    d_chem: int = 384

    # 统一隐藏维度
    d_model: int = 512
    dropout: float = 0.1
    act: str = "silu"

    # MUTAN
    mutan_rank: int = 10
    mutan_dim: int = 512    # 输出维度

    # 分类头
    head_hidden: int = 512


class UGCAModel(nn.Module):
    """
    输入：
        v_protein:  [B, d_protein]   (ESM2)
        v_molclr:   [B, d_molclr]    (MolCLR)
        v_chem:     [B, d_chem]      (ChemBERTa)
    流程：
      1) 先做药物模态内部融合：UGCA(molclr, chemberta) -> drug_fused
      2) 再做药物-蛋白互作用：MUTAN(drug_fused, protein) -> joint
      3) 分类头输出 logit
    """
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg = ModelCfg(**cfg) if isinstance(cfg, dict) else cfg

        d = cfg.d_model
        self.dropout = nn.Dropout(cfg.dropout)

        # 1) 药物模态融合（UGCA）
        self.ugca_drug = UGCAUnit(
            da=cfg.d_molclr, db=cfg.d_chem, d=d,
            dropout=cfg.dropout, act=cfg.act
        )

        # 2) 药物-蛋白互作用（MUTAN 低秩双线性）
        self.proj_prot = nn.Linear(cfg.d_protein, d, bias=False)
        self.mutan_dp = Mutan(dx=d, dy=d, dz=cfg.mutan_dim, rank=cfg.mutan_rank, dropout=cfg.dropout, act="tanh")

        # 3) 分类头
        self.head = nn.Sequential(
            LayerNorm1d(cfg.mutan_dim),
            nn.Linear(cfg.mutan_dim, cfg.head_hidden),
            _act(cfg.act),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

        # init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, v_protein: torch.Tensor, v_molclr: torch.Tensor, v_chem: torch.Tensor) -> torch.Tensor:
        # 1) 药物内部融合（UGCA）
        drug = self.ugca_drug(v_molclr, v_chem)               # [B, d]

        # 2) 药物-蛋白互作用（MUTAN）
        prot = self.proj_prot(v_protein)                      # [B, d]
        joint = self.mutan_dp(drug, prot)                     # [B, mutan_dim]

        # 3) 分类
        logit = self.head(joint).squeeze(-1)                  # [B]
        return logit


# ====== 训练脚本会优先调用这个工厂函数 ======
def build_model(cfg: Dict) -> nn.Module:
    """
    train.py 会先尝试 from src.model import build_model
    我们把 YAML/CLI 的 model 字段（字典）转成 ModelCfg 实例。
    """
    mcfg = ModelCfg(
        d_protein=int(cfg.get("d_protein", 1280)),
        d_molclr=int(cfg.get("d_molclr", 300)),
        d_chem=int(cfg.get("d_chem", 384)),
        d_model=int(cfg.get("d_model", 512)),
        dropout=float(cfg.get("dropout", 0.1)),
        act=str(cfg.get("act", "silu")),
        mutan_rank=int(cfg.get("mutan_rank", 10)),
        mutan_dim=int(cfg.get("mutan_dim", 512)),
        head_hidden=int(cfg.get("head_hidden", 512)),
    )
    return UGCAModel(mcfg)
