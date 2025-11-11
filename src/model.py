# src/model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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
    def __init__(self, dx: int, dy: int, dz: int, rank: int = 10, dropout: float = 0.1, act: str = "tanh"):
        super().__init__()
        self.rank = rank
        self.proj_x = nn.Sequential(nn.Linear(dx, dz * rank, bias=False), nn.Dropout(dropout))
        self.proj_y = nn.Sequential(nn.Linear(dy, dz * rank, bias=False), nn.Dropout(dropout))
        self.out = nn.Sequential(LayerNorm1d(dz), nn.Dropout(dropout))
        self.nonlin = _act(act)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bx = self.proj_x(x)       # [B, dz*R]
        by = self.proj_y(y)       # [B, dz*R]
        B, _ = bx.shape; R = self.rank
        bx = bx.view(B, R, -1)
        by = by.view(B, R, -1)
        z = self.nonlin(bx) * self.nonlin(by)   # [B,R,dz]
        z = z.sum(dim=1)                        # [B,dz]
        return self.out(z)

# ====== UGCA（V1：向量级门控融合）======
class UGCAUnit(nn.Module):
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
        A = self.pa(a); B = self.pb(b)
        A2 = self.ln_a(A + self.ff_a(A))
        B2 = self.ln_b(B + self.ff_b(B))
        g = self.gate(torch.cat([A2, B2], dim=-1))
        out = g * A2 + (1.0 - g) * B2
        return out

# ====== V1 顶层模型 ======
@dataclass
class ModelCfg:
    d_protein: int = 1280   # ESM2
    d_molclr: int = 300
    d_chem: int = 384
    d_model: int = 512
    dropout: float = 0.1
    act: str = "silu"
    mutan_rank: int = 10
    mutan_dim: int = 512
    head_hidden: int = 512

class UGCAModel(nn.Module):
    """
    V1（向量级）：molclr + chemberta -> UGCA -> 与 protein 做 MUTAN -> 分类
    输入：
        v_protein:  [B, d_protein]
        v_molclr:   [B, d_molclr]
        v_chem:     [B, d_chem]
    """
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg = ModelCfg(**cfg) if isinstance(cfg, dict) else cfg
        d = cfg.d_model
        self.dropout = nn.Dropout(cfg.dropout)
        self.ugca_drug = UGCAUnit(da=cfg.d_molclr, db=cfg.d_chem, d=d, dropout=cfg.dropout, act=cfg.act)
        self.proj_prot = nn.Linear(cfg.d_protein, d, bias=False)
        self.mutan_dp = Mutan(dx=d, dy=d, dz=cfg.mutan_dim, rank=cfg.mutan_rank, dropout=cfg.dropout, act="tanh")
        self.head = nn.Sequential(
            LayerNorm1d(cfg.mutan_dim),
            nn.Linear(cfg.mutan_dim, cfg.head_hidden),
            _act(cfg.act),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )
        self.apply(self._init_weights)
        # 为了与训练里门控预算接口兼容（V1 没有 token 级门控，这里返回 None）
        self._last_gd = None
        self._last_gp = None

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, v_protein: torch.Tensor, v_molclr: torch.Tensor, v_chem: torch.Tensor) -> torch.Tensor:
        drug = self.ugca_drug(v_molclr, v_chem)               # [B, d]
        prot = self.proj_prot(v_protein)                      # [B, d]
        joint = self.mutan_dp(drug, prot)                     # [B, mutan_dim]
        logit = self.head(joint).squeeze(-1)                  # [B]
        # V1：没有 token 级门控，清空
        self._last_gd, self._last_gp = None, None
        return logit

    def last_gates(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._last_gd, self._last_gp

# ====== V2：序列级 UGCA（per-token）======
class TokenGate(nn.Module):
    """给每个 token 产生一个门控 g∈(0,1)"""
    def __init__(self, d: int):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(x))  # (B,T,1)

class EvidentialTokenGate(nn.Module):
    """
    基于 Normal-Inverse-Gamma 的证据门控：
      x -> (mu, v, alpha, beta) 经 softplus 约束；以“epistemic 方差”抑制置信：
        Var_ep = beta / (v * (alpha - 1))
        g = sigmoid(mu) * exp(-lam * Var_ep)
    """
    def __init__(self, d: int, lam: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(d, 4, bias=True)
        self.lam = float(lam)
        self.eps = float(eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)                    # (B,T,4)
        mu, v_raw, a_raw, b_raw = torch.chunk(h, 4, dim=-1)
        v = F.softplus(v_raw) + self.eps    # >0
        alpha = F.softplus(a_raw) + 1.0 + self.eps
        beta  = F.softplus(b_raw) + self.eps
        var_ep = beta / (v * (alpha - 1.0))
        g = torch.sigmoid(mu) * torch.exp(-self.lam * var_ep)
        return g.clamp_min(1e-6)            # (B,T,1)

class GatedCrossAttn(nn.Module):
    """
    x<->y 的对称门控 cross-attn：
      score_xy = Qx Ky^T / sqrt(dh) + log(gx) + log(gy)^T
      attn_xy  = softmax(score_xy)
      out_x    = attn_xy Vy
    """
    def __init__(self, d: int, nhead: int = 4, dropout: float = 0.1,
                 gate_type: str = "evidential", gate_lambda: float = 2.0):
        super().__init__()
        assert d % nhead == 0
        self.d, self.h = d, nhead
        self.dk = d // nhead
        self.qx = nn.Linear(d, d, bias=False)
        self.kx = nn.Linear(d, d, bias=False)
        self.vx = nn.Linear(d, d, bias=False)
        self.qy = nn.Linear(d, d, bias=False)
        self.ky = nn.Linear(d, d, bias=False)
        self.vy = nn.Linear(d, d, bias=False)
        # 门控类型
        if (gate_type or "evidential").lower().startswith("evi"):
            self.gx = EvidentialTokenGate(d, lam=gate_lambda)
            self.gy = EvidentialTokenGate(d, lam=gate_lambda)
        else:
            self.gx = TokenGate(d)
            self.gy = TokenGate(d)
        self.out_x = nn.Linear(d, d, bias=False)
        self.out_y = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln_x = nn.LayerNorm(d)
        self.ln_y = nn.LayerNorm(d)
        self.ff_x = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Dropout(dropout))
        self.ff_y = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Dropout(dropout))

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        # (B,T,d) -> (B,h,T,dk)
        B, T, _ = t.shape
        return t.view(B, T, self.h, self.dk).transpose(1, 2)

    def _merge(self, t: torch.Tensor) -> torch.Tensor:
        # (B,h,T,dk) -> (B,T,d)
        B, _, T, _ = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, self.d)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor,
                      y: torch.Tensor, y_mask: torch.Tensor):
        B, Nx, d = x.shape
        Ny = y.shape[1]
        qx, kx, vx = self._split(self.qx(x)), self._split(self.kx(x)), self._split(self.vx(x))  # (B,h,Nx,dk)
        qy, ky, vy = self._split(self.qy(y)), self._split(self.ky(y)), self._split(self.vy(y))  # (B,h,Ny,dk)

        gx = self.gx(x).clamp_min(1e-6)  # (B,Nx,1)
        gy = self.gy(y).clamp_min(1e-6)  # (B,Ny,1)

        # x -> y
        scores_xy = torch.matmul(qx, ky.transpose(-2, -1)) / math.sqrt(self.dk)  # (B,h,Nx,Ny)
        scores_xy = scores_xy + torch.log(gx).unsqueeze(1) + torch.log(gy.transpose(1, 2)).unsqueeze(1)  # broadcast
        # mask y 端
        if y_mask is not None:
            mask_y = (~y_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,Ny)
            scores_xy = scores_xy.masked_fill(mask_y, float("-inf"))
        attn_xy = torch.softmax(scores_xy, dim=-1)
        attn_xy = self.dropout(attn_xy)
        out_x = torch.matmul(attn_xy, vy)  # (B,h,Nx,dk)
        out_x = self._merge(out_x)
        out_x = self.out_x(out_x)
        x2 = self.ln_x(x + out_x)
        x2 = self.ln_x(x2 + self.ff_x(x2))

        # y -> x
        scores_yx = torch.matmul(qy, kx.transpose(-2, -1)) / math.sqrt(self.dk)  # (B,h,Ny,Nx)
        scores_yx = scores_yx + torch.log(gy).unsqueeze(1) + torch.log(gx.transpose(1, 2)).unsqueeze(1)
        if x_mask is not None:
            mask_x = (~x_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,Nx)
            scores_yx = scores_yx.masked_fill(mask_x, float("-inf"))
        attn_yx = torch.softmax(scores_yx, dim=-1)
        attn_yx = self.dropout(attn_yx)
        out_y = torch.matmul(attn_yx, vx)
        out_y = self._merge(out_y)
        out_y = self.out_y(out_y)
        y2 = self.ln_y(y + out_y)
        y2 = self.ln_y(y2 + self.ff_y(y2))

        return x2, y2, gx.squeeze(-1), gy.squeeze(-1)  # (B,Nx), (B,Ny)

def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    # x:(B,T,d) m:(B,T) -> (B,d)
    m = m.float()
    s = (x * m.unsqueeze(-1)).sum(dim=1)
    d = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    return s / d

@dataclass
class SeqCfg:
    d_protein: int = 1280
    d_molclr: int = 300
    d_chem: int = 384
    d_model: int = 512
    nhead: int = 4
    nlayers: int = 2
    dropout: float = 0.1
    mutan_dim: int = 512
    mutan_rank: int = 10
    head_hidden: int = 512
    # 门控
    gate_type: str = "evidential"   # evidential | mlp
    gate_lambda: float = 2.0        # 不确定性抑制系数 λ
    topk_ratio: float = 0.0         # >0 启用逐层 top-k 稀疏门控

class UGCASeqModel(nn.Module):
    """
    V2（序列级）：
      Protein tokens (ESM2)  <->  Drug tokens (MolCLR+ChemBERTa)  通过门控 cross-attn
      然后池化 -> MUTAN -> 分类
    forward 输入：
      v_protein:(B,M,dp), mask_p:(B,M), v_mol:(B,N,dd), mask_d:(B,N), v_chem:(B,dc)
    """
    def __init__(self, cfg: SeqCfg):
        super().__init__()
        self.cfg = cfg = SeqCfg(**cfg) if isinstance(cfg, dict) else cfg
        d = cfg.d_model
        # 投影
        self.proj_p = nn.Linear(cfg.d_protein, d, bias=False)
        self.proj_d = nn.Linear(cfg.d_molclr,  d, bias=False)
        self.proj_c = nn.Linear(cfg.d_chem,    d, bias=False)
        # 门控 cross-attn 层堆叠
        self.layers = nn.ModuleList([
            GatedCrossAttn(d=d, nhead=cfg.nhead, dropout=cfg.dropout,
                           gate_type=cfg.gate_type, gate_lambda=cfg.gate_lambda)
            for _ in range(cfg.nlayers)
        ])
        self.topk_ratio = float(getattr(cfg, "topk_ratio", 0.0))
        # MUTAN + 头
        self.mutan = Mutan(dx=d, dy=d, dz=cfg.mutan_dim, rank=cfg.mutan_rank, dropout=cfg.dropout, act="tanh")
        self.head = nn.Sequential(
            LayerNorm1d(cfg.mutan_dim),
            nn.Linear(cfg.mutan_dim, cfg.head_hidden),
            nn.SiLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1)
        )
        # 保存门控以便预算正则
        self._last_gd = None
        self._last_gp = None

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight);
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, v_prot: torch.Tensor, m_prot: torch.Tensor,
                      v_mol: torch.Tensor,  m_mol: torch.Tensor,
                      v_chem: torch.Tensor) -> torch.Tensor:
        # 1) 投影到同维，并用 ChemBERTa 提升药物 token
        P = self.proj_p(v_prot)                                   # (B,M,d)
        D = self.proj_d(v_mol) + self.proj_c(v_chem).unsqueeze(1) # (B,N,d) + broadcast

        # 2) 多层门控 cross-attn（可选逐层 top-k 稀疏门控）
        gd = gp = None
        for li, layer in enumerate(self.layers):
            D, P, gd, gp = layer(D, m_mol, P, m_prot)             # D<->P
            if self.topk_ratio > 0.0:
                # 逐层更新 mask：保留门控较大的 token
                with torch.no_grad():
                    def _update_mask(g: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
                        # g:(B,T) m:(B,T)[bool]
                        B, T = g.shape
                        new_m = m.clone()
                        for i in range(B):
                            valid = torch.nonzero(m[i], as_tuple=False).squeeze(-1)
                            if valid.numel() == 0:
                                continue
                            k = max(1, int(valid.numel() * self.topk_ratio + 1e-6))
                            vals = g[i, valid]
                            topk = torch.topk(vals, k=min(k, valid.numel()), largest=True).indices
                            keep = valid[topk]
                            nm = torch.zeros_like(m[i])
                            nm[keep] = True
                            new_m[i] = nm
                        return new_m
                    m_mol = _update_mask(gd, m_mol)
                    m_prot = _update_mask(gp, m_prot)

        self._last_gd, self._last_gp = gd, gp                     # (B,N), (B,M)

        # 3) 池化 + MUTAN + 分类
        hd = _masked_mean(D, m_mol)                               # (B,d)
        hp = _masked_mean(P, m_prot)                              # (B,d)
        z  = self.mutan(hd, hp)                                   # (B,mutan_dim)
        logit = self.head(z).squeeze(-1)                          # (B,)
        return logit

    def last_gates(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._last_gd, self._last_gp

    # 两阶段训练：冻结/解冻门控
    def freeze_gating(self, freeze: bool = True):
        for layer in self.layers:
            for mod in (layer.gx, layer.gy):
                for p in mod.parameters():
                    p.requires_grad = (not freeze)

# ====== 工厂函数（训练脚本优先调用）======
def build_model(cfg: Dict) -> nn.Module:
    """
    train.py 会先尝试 from src.model import build_model
    """
    # 兼容：若传了 sequence=True，我就建 UGCASeqModel；否则建 V1
    if bool(cfg.get("sequence", False)):
        mcfg = SeqCfg(
            d_protein=int(cfg.get("d_protein", 1280)),
            d_molclr=int(cfg.get("d_molclr", 300)),
            d_chem=int(cfg.get("d_chem", 384)),
            d_model=int(cfg.get("d_model", 512)),
            nhead=int(cfg.get("nhead", 4)),
            nlayers=int(cfg.get("nlayers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            mutan_rank=int(cfg.get("mutan_rank", 10)),
            mutan_dim=int(cfg.get("mutan_dim", 512)),
            head_hidden=int(cfg.get("head_hidden", 512)),
            gate_type=str(cfg.get("gate_type", "evidential")),
            gate_lambda=float(cfg.get("gate_lambda", 2.0)),
            topk_ratio=float(cfg.get("topk_ratio", 0.0)),
        )
        return UGCASeqModel(mcfg)
    else:
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
