# src/model.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

<<<<<<< HEAD
# =========================
# Utils
# =========================
=======
# ====== 小工具 ======
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
def _act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    return {
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
        "tanh": nn.Tanh(),
    }.get(name, nn.ReLU(inplace=True))

class LayerNorm1d(nn.Module):
    """LayerNorm on last dim with safe fp32 compute."""
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(d, eps=eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            return self.ln(x.float()).type_as(x)
        return self.ln(x)

<<<<<<< HEAD
# =========================
# MUTAN (low-rank bilinear)
# =========================
=======
# ====== MUTAN（低秩双线性池化）======
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
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

<<<<<<< HEAD
# =========================
# UGCA (V1: vector-level)
# =========================
class UGCAUnit(nn.Module):
    """Gated fusion of two global vectors -> d-model"""
=======
# ====== UGCA（V1：向量级门控融合）======
class UGCAUnit(nn.Module):
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
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

<<<<<<< HEAD
=======
# ====== V1 顶层模型 ======
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
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
<<<<<<< HEAD
    """V1 (vector-level)"""
=======
    """
    V1（向量级）：molclr + chemberta -> UGCA -> 与 protein 做 MUTAN -> 分类
    输入：
        v_protein:  [B, d_protein]
        v_molclr:   [B, d_molclr]
        v_chem:     [B, d_chem]
    """
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
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
<<<<<<< HEAD
=======
        # 为了与训练里门控预算接口兼容（V1 没有 token 级门控，这里返回 None）
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
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
<<<<<<< HEAD
=======
        # V1：没有 token 级门控，清空
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
        self._last_gd, self._last_gp = None, None
        return logit

    def last_gates(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._last_gd, self._last_gp

<<<<<<< HEAD
# =========================
# V2: sequence-level UGCA
# =========================
class TokenGate(nn.Module):
    """MLP token gate g∈(0,1)"""
    def __init__(self, d: int, g_min: float = 1e-6):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        self.g_min = float(g_min)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.mlp(x))
        return g.clamp_min(self.g_min)  # (B,T,1)

class EvidentialTokenGate(nn.Module):
    """
    NIG-based gate:
      Parameters: (mu, v, alpha, beta)
      Epistemic variance: var_ep = beta / (v * (alpha - 1))
      Modes:
        - "evi_only":       g = exp(-lam * var_ep)
        - "evi_x_mu":       g = sigmoid(mu) * exp(-lam * var_ep)
      Then clamp to [g_min, 1].
    """
    def __init__(self, d: int, lam: float = 2.0, mode: str = "evi_x_mu", g_min: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(d, 4, bias=True)
        self.lam = float(lam)
        self.eps = float(eps)
        self.mode = (mode or "evi_x_mu").lower()
        self.g_min = float(g_min)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)                    # (B,T,4)
        mu, v_raw, a_raw, b_raw = torch.chunk(h, 4, dim=-1)
        v = F.softplus(v_raw) + self.eps    # >0
        alpha = F.softplus(a_raw) + 1.0 + self.eps
        beta  = F.softplus(b_raw) + self.eps
        var_ep = beta / (v * (alpha - 1.0))
        if self.mode.startswith("evi_only"):
            g = torch.exp(-self.lam * var_ep)
        else:
            g = torch.sigmoid(mu) * torch.exp(-self.lam * var_ep)
        return g.clamp_min(self.g_min)      # (B,T,1)

def _smooth_gate(g: torch.Tensor) -> torch.Tensor:
    """Lightweight 1D smoothing on token axis with kernel [0.25, 0.5, 0.25]. g:(B,T,1)"""
    if g.dim() != 3 or g.size(-1) != 1:
        return g
    B, T, _ = g.shape
    if T < 3:
        return g
    weight = torch.tensor([0.25, 0.5, 0.25], dtype=g.dtype, device=g.device).view(1,1,3)
    # conv1d expects (B,C,T)
    x = g.transpose(1,2)  # (B,1,T)
    pad = (1,1)
    x = F.pad(x, pad, mode="replicate")
    y = F.conv1d(x, weight, padding=0)
    return y.transpose(1,2)

class GatedCrossAttn(nn.Module):
    """
    Bidirectional cross-attn with pre-softmax log-gate bias.
    score_xy = (Qx Ky^T)/sqrt(dh) + log(gx) + log(gy)^T
    """
    def __init__(self, d: int, nhead: int = 4, dropout: float = 0.1,
                 gate_type: str = "evidential", gate_lambda: float = 2.0,
                 gate_mode: str = "evi_x_mu", g_min: float = 0.05,
                 smooth_g: bool = False, attn_temp: float = 1.0):
=======
# ====== V2：序列级 UGCA（per-token）======
class TokenGate(nn.Module):
    """给每个 token 产生一个门控 g∈(0,1)"""
    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(x))  # (B,T,1)

class GatedCrossAttn(nn.Module):
    """
    x<->y 的对称门控 cross-attn：
      score_xy = Qx Ky^T / sqrt(dh) + log(gx) + log(gy)^T
      attn_xy  = softmax(score_xy)
      out_x    = attn_xy Vy
    """
    def __init__(self, d: int, nhead: int = 4, dropout: float = 0.1):
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
        super().__init__()
        assert d % nhead == 0
        self.d, self.h = d, nhead
        self.dk = d // nhead
<<<<<<< HEAD
        self.attn_temp = float(attn_temp)
        # QKV
=======
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
        self.qx = nn.Linear(d, d, bias=False)
        self.kx = nn.Linear(d, d, bias=False)
        self.vx = nn.Linear(d, d, bias=False)
        self.qy = nn.Linear(d, d, bias=False)
        self.ky = nn.Linear(d, d, bias=False)
        self.vy = nn.Linear(d, d, bias=False)
<<<<<<< HEAD
        # Gate
        gt = (gate_type or "evidential").lower()
        if gt.startswith("evi"):
            self.gx = EvidentialTokenGate(d, lam=gate_lambda, mode=gate_mode, g_min=g_min)
            self.gy = EvidentialTokenGate(d, lam=gate_lambda, mode=gate_mode, g_min=g_min)
        else:
            self.gx = TokenGate(d, g_min=g_min)
            self.gy = TokenGate(d, g_min=g_min)
        self.smooth_g = bool(smooth_g)
        # Output
=======
        self.gx = TokenGate(d)
        self.gy = TokenGate(d)
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
        self.out_x = nn.Linear(d, d, bias=False)
        self.out_y = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln_x = nn.LayerNorm(d)
        self.ln_y = nn.LayerNorm(d)
        self.ff_x = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Dropout(dropout))
        self.ff_y = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Dropout(dropout))
<<<<<<< HEAD
        # Buffers to expose attention entropy (for optional regularization)
        self.last_entropy_x = None
        self.last_entropy_y = None

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = t.shape
        return t.view(B, T, self.h, self.dk).transpose(1, 2)  # (B,h,T,dk)

    def _merge(self, t: torch.Tensor) -> torch.Tensor:
=======

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        # (B,T,d) -> (B,h,T,dk)
        B, T, _ = t.shape
        return t.view(B, T, self.h, self.dk).transpose(1, 2)

    def _merge(self, t: torch.Tensor) -> torch.Tensor:
        # (B,h,T,dk) -> (B,T,d)
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
        B, _, T, _ = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, self.d)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor,
<<<<<<< HEAD
                      y: torch.Tensor, y_mask: torch.Tensor,
                      gate_enabled: bool = True):
        B, Nx, d = x.shape
        Ny = y.shape[1]
        qx, kx, vx = self._split(self.qx(x)), self._split(self.kx(x)), self._split(self.vx(x))
        qy, ky, vy = self._split(self.qy(y)), self._split(self.ky(y)), self._split(self.vy(y))

        if gate_enabled:
            gx = self.gx(x).clamp_min(1e-6)  # (B,Nx,1)
            gy = self.gy(y).clamp_min(1e-6)  # (B,Ny,1)
            if self.smooth_g:
                gx = _smooth_gate(gx)
                gy = _smooth_gate(gy)
            log_gx = torch.log(gx)
            log_gy = torch.log(gy)
        else:
            log_gx = log_gy = 0.0

        # x -> y
        scores_xy = torch.matmul(qx, ky.transpose(-2, -1)) / (math.sqrt(self.dk) * max(self.attn_temp, 1e-6))
        if gate_enabled:
            scores_xy = scores_xy + log_gx.unsqueeze(1) + log_gy.transpose(1,2).unsqueeze(1)
=======
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
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
        if y_mask is not None:
            mask_y = (~y_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,Ny)
            scores_xy = scores_xy.masked_fill(mask_y, float("-inf"))
        attn_xy = torch.softmax(scores_xy, dim=-1)
<<<<<<< HEAD
        self.last_entropy_x = (-attn_xy * (torch.log(attn_xy + 1e-8))).sum(dim=-1).mean()  # mean over (B,h,Nx)
        attn_xy = self.dropout(attn_xy)
        out_x = torch.matmul(attn_xy, vy)
=======
        attn_xy = self.dropout(attn_xy)
        out_x = torch.matmul(attn_xy, vy)  # (B,h,Nx,dk)
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
        out_x = self._merge(out_x)
        out_x = self.out_x(out_x)
        x2 = self.ln_x(x + out_x)
        x2 = self.ln_x(x2 + self.ff_x(x2))

        # y -> x
<<<<<<< HEAD
        scores_yx = torch.matmul(qy, kx.transpose(-2, -1)) / (math.sqrt(self.dk) * max(self.attn_temp, 1e-6))
        if gate_enabled:
            scores_yx = scores_yx + log_gy.unsqueeze(1) + log_gx.transpose(1, 2).unsqueeze(1)
=======
        scores_yx = torch.matmul(qy, kx.transpose(-2, -1)) / math.sqrt(self.dk)  # (B,h,Ny,Nx)
        scores_yx = scores_yx + torch.log(gy).unsqueeze(1) + torch.log(gx.transpose(1, 2)).unsqueeze(1)
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
        if x_mask is not None:
            mask_x = (~x_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,Nx)
            scores_yx = scores_yx.masked_fill(mask_x, float("-inf"))
        attn_yx = torch.softmax(scores_yx, dim=-1)
<<<<<<< HEAD
        self.last_entropy_y = (-attn_yx * (torch.log(attn_yx + 1e-8))).sum(dim=-1).mean()
=======
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
        attn_yx = self.dropout(attn_yx)
        out_y = torch.matmul(attn_yx, vx)
        out_y = self._merge(out_y)
        out_y = self.out_y(out_y)
        y2 = self.ln_y(y + out_y)
        y2 = self.ln_y(y2 + self.ff_y(y2))

<<<<<<< HEAD
        # Return also gates for budget reg; if disabled, return ones to signal "inactive"
        if gate_enabled:
            gd = gx.squeeze(-1); gp = gy.squeeze(-1)
        else:
            gd = gp = None
        return x2, y2, gd, gp

def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
=======
        return x2, y2, gx.squeeze(-1), gy.squeeze(-1)  # (B,Nx), (B,Ny)

def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    # x:(B,T,d) m:(B,T) -> (B,d)
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
    m = m.float()
    s = (x * m.unsqueeze(-1)).sum(dim=1)
    d = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    return s / d

<<<<<<< HEAD
class AttnPool(nn.Module):
    """Cross attention pooling: use query from the other side's global vector."""
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, query_vec: torch.Tensor):
        # keys/values: (B,T,d), mask:(B,T) bool, query_vec:(B,d)
        q = self.q(query_vec).unsqueeze(1)     # (B,1,d)
        k = self.k(keys)                       # (B,T,d)
        v = self.v(values)                     # (B,T,d)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(keys.size(-1))  # (B,1,T)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)   # (B,1,T)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).squeeze(1) # (B,d)
        return out

=======
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
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
<<<<<<< HEAD
    # Gate
    gate_type: str = "evidential"   # evidential | mlp
    gate_mode: str = "evi_x_mu"     # evi_x_mu | evi_only
    gate_lambda: float = 2.0        # uncertainty suppression λ
    g_min: float = 0.05             # clamp min
    smooth_g: bool = False
    topk_ratio: float = 0.0         # global ratio, can be warmed-up via train loop
    # Cross-attn
    attn_temp: float = 1.0
    # Pooling
    pool_type: str = "attn"         # mean | attn
    # Regularization
    entropy_reg: float = 0.0        # optional entropy regularizer weight

class UGCASeqModel(nn.Module):
    """
    Sequence-level UGCA:
      - Project tokens to shared dim d
      - L layers of gated bidirectional cross-attn
      - Pool to global vectors (mean or cross-attn pooling)
      - MUTAN fusion -> classifier
=======

class UGCASeqModel(nn.Module):
    """
    V2（序列级）：
      Protein tokens (ESM2)  <->  Drug tokens (MolCLR+ChemBERTa)  通过门控 cross-attn
      然后池化 -> MUTAN -> 分类
    forward 输入：
      v_protein:(B,M,dp), mask_p:(B,M), v_mol:(B,N,dd), mask_d:(B,N), v_chem:(B,dc)
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
    """
    def __init__(self, cfg: SeqCfg):
        super().__init__()
        self.cfg = cfg = SeqCfg(**cfg) if isinstance(cfg, dict) else cfg
        d = cfg.d_model
<<<<<<< HEAD
        # Projections
        self.proj_p = nn.Linear(cfg.d_protein, d, bias=False)
        self.proj_d = nn.Linear(cfg.d_molclr,  d, bias=False)
        self.proj_c = nn.Linear(cfg.d_chem,    d, bias=False)
        # Gated cross-attn layers
        self.layers = nn.ModuleList([
            GatedCrossAttn(d=d, nhead=cfg.nhead, dropout=cfg.dropout,
                           gate_type=cfg.gate_type, gate_lambda=cfg.gate_lambda,
                           gate_mode=cfg.gate_mode, g_min=cfg.g_min,
                           smooth_g=cfg.smooth_g, attn_temp=cfg.attn_temp)
            for _ in range(cfg.nlayers)
        ])
        self.topk_ratio = float(getattr(cfg, "topk_ratio", 0.0))
        self.gate_enabled = True  # Stage-A will disable
        self.entropy_reg = float(cfg.entropy_reg)
        # Pooling
        self.pool_type = str(cfg.pool_type or "attn").lower()
        if self.pool_type == "attn":
            self.pool_d = AttnPool(d, dropout=cfg.dropout)
            self.pool_p = AttnPool(d, dropout=cfg.dropout)
        # MUTAN + head
=======
        # 投影
        self.proj_p = nn.Linear(cfg.d_protein, d, bias=False)
        self.proj_d = nn.Linear(cfg.d_molclr,  d, bias=False)
        self.proj_c = nn.Linear(cfg.d_chem,    d, bias=False)
        # 门控 cross-attn 层堆叠
        self.layers = nn.ModuleList([GatedCrossAttn(d=d, nhead=cfg.nhead, dropout=cfg.dropout) for _ in range(cfg.nlayers)])
        # MUTAN + 头
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
        self.mutan = Mutan(dx=d, dy=d, dz=cfg.mutan_dim, rank=cfg.mutan_rank, dropout=cfg.dropout, act="tanh")
        self.head = nn.Sequential(
            LayerNorm1d(cfg.mutan_dim),
            nn.Linear(cfg.mutan_dim, cfg.head_hidden),
            nn.SiLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1)
        )
<<<<<<< HEAD
        # For training loop: expose latest gates and attn entropies
        self._last_gd = None
        self._last_gp = None
        self._last_entropy = 0.0
=======
        # 保存门控以便预算正则
        self._last_gd = None
        self._last_gp = None
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight);
            if m.bias is not None: nn.init.zeros_(m.bias)

<<<<<<< HEAD
    def set_gate_enabled(self, enabled: bool = True):
        self.gate_enabled = bool(enabled)

    def freeze_gating(self, freeze: bool = True):
        for layer in self.layers:
            for mod in (layer.gx, layer.gy):
                for p in mod.parameters():
                    p.requires_grad = (not freeze)

    def _apply_topk(self, g: torch.Tensor, m: torch.Tensor, ratio: float) -> torch.Tensor:
        # g:(B,T) m:(B,T)[bool] -> new mask
        B, T = g.shape
        if ratio <= 0.0 or ratio >= 1.0:
            return m
        new_m = m.clone()
        for i in range(B):
            valid = torch.nonzero(m[i], as_tuple=False).squeeze(-1)
            if valid.numel() == 0:
                continue
            k = max(1, int(valid.numel() * ratio + 1e-6))
            vals = g[i, valid]
            topk = torch.topk(vals, k=min(k, valid.numel()), largest=True).indices
            keep = valid[topk]
            nm = torch.zeros_like(m[i])
            nm[keep] = True
            new_m[i] = nm
        return new_m

    def forward(self, v_prot: torch.Tensor, m_prot: torch.Tensor,
                      v_mol: torch.Tensor,  m_mol: torch.Tensor,
                      v_chem: torch.Tensor,
                      topk_ratio: Optional[float] = None) -> torch.Tensor:
        # 1) project to d, and inject global chem vector into drug tokens
        P = self.proj_p(v_prot)                                   # (B,M,d)
        D = self.proj_d(v_mol) + self.proj_c(v_chem).unsqueeze(1) # (B,N,d)

        # 2) L layers of gated cross-attn (with optional top-k sparsification)
        gd = gp = None
        curr_ratio = self.topk_ratio if topk_ratio is None else float(topk_ratio)
        mP, mD = m_prot, m_mol
        ent_list = []
        for layer in self.layers:
            D, P, gd, gp = layer(D, mD, P, mP, gate_enabled=self.gate_enabled)  # D<->P
            # collect entropy (optional)
            if layer.last_entropy_x is not None and layer.last_entropy_y is not None:
                ent_list.append(0.5 * (layer.last_entropy_x + layer.last_entropy_y))
            # sparsify masks
            if curr_ratio > 0.0 and self.gate_enabled:
                with torch.no_grad():
                    mD = self._apply_topk(gd, mD, curr_ratio)
                    mP = self._apply_topk(gp, mP, curr_ratio)

        self._last_gd, self._last_gp = gd, gp                     # (B,N), (B,M)
        self._last_entropy = torch.stack(ent_list).mean() if len(ent_list)>0 else torch.tensor(0.0, device=P.device)

        # 3) Pooling
        if self.pool_type == "attn":
            # cross query: use mean of other side as query
            q_p = _masked_mean(P, mP)    # (B,d)
            q_d = _masked_mean(D, mD)
            hd = self.pool_d(keys=D, values=D, mask=mD, query_vec=q_p)
            hp = self.pool_p(keys=P, values=P, mask=mP, query_vec=q_d)
        else:
            hd = _masked_mean(D, mD)
            hp = _masked_mean(P, mP)

        # 4) MUTAN + head
=======
    def forward(self, v_prot: torch.Tensor, m_prot: torch.Tensor,
                      v_mol: torch.Tensor,  m_mol: torch.Tensor,
                      v_chem: torch.Tensor) -> torch.Tensor:
        # 1) 投影到同维，并用 ChemBERTa 提升药物 token
        P = self.proj_p(v_prot)                                   # (B,M,d)
        D = self.proj_d(v_mol) + self.proj_c(v_chem).unsqueeze(1) # (B,N,d) + broadcast

        # 2) 多层门控 cross-attn
        gd = gp = None
        for layer in self.layers:
            D, P, gd, gp = layer(D, m_mol, P, m_prot)             # D<->P
        self._last_gd, self._last_gp = gd, gp                     # (B,N), (B,M)

        # 3) 池化 + MUTAN + 分类
        hd = _masked_mean(D, m_mol)                               # (B,d)
        hp = _masked_mean(P, m_prot)                              # (B,d)
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
        z  = self.mutan(hd, hp)                                   # (B,mutan_dim)
        logit = self.head(z).squeeze(-1)                          # (B,)
        return logit

    def last_gates(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._last_gd, self._last_gp

<<<<<<< HEAD
# =========================
# Factory
# =========================
def build_model(cfg: Dict) -> nn.Module:
    """
    train.py expects to import build_model(cfg).
    """
=======
# ====== 工厂函数（训练脚本优先调用）======
def build_model(cfg: Dict) -> nn.Module:
    """
    train.py 会先尝试 from src.model import build_model
    """
    # 兼容：若传了 sequence=True，我就建 UGCASeqModel；否则建 V1
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
    if bool(cfg.get("sequence", False)):
        mcfg = SeqCfg(
            d_protein=int(cfg.get("d_protein", 1280)),
            d_molclr=int(cfg.get("d_molclr", 300)),
            d_chem=int(cfg.get("d_chem", 384)),
            d_model=int(cfg.get("d_model", 512)),
            nhead=int(cfg.get("nhead", 4)),
            nlayers=int(cfg.get("nlayers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
<<<<<<< HEAD
            mutan_rank=int(cfg.get("mutan_rank", 20 if cfg.get("mutan_rank") is None else cfg.get("mutan_rank"))),
            mutan_dim=int(cfg.get("mutan_dim", 256 if cfg.get("mutan_dim") is None else cfg.get("mutan_dim"))),
            head_hidden=int(cfg.get("head_hidden", 512)),
            # gate
            gate_type=str(cfg.get("gate_type", "evidential")),
            gate_mode=str(cfg.get("gate_mode", "evi_x_mu")),
            gate_lambda=float(cfg.get("gate_lambda", 2.0)),
            g_min=float(cfg.get("g_min", 0.05)),
            smooth_g=bool(cfg.get("smooth_g", False)),
            topk_ratio=float(cfg.get("topk_ratio", 0.0)),
            # cross-attn
            attn_temp=float(cfg.get("attn_temp", 1.0)),
            # pooling
            pool_type=str(cfg.get("pool_type", "attn")),
            # regularization
            entropy_reg=float(cfg.get("entropy_reg", 0.0))
=======
            mutan_rank=int(cfg.get("mutan_rank", 10)),
            mutan_dim=int(cfg.get("mutan_dim", 512)),
            head_hidden=int(cfg.get("head_hidden", 512)),
>>>>>>> c45b53e3a067a7d80114a697f327ff65c138d752
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
