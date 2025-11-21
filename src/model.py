# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Utils
# =========================
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

# =========================
# Low-rank bilinear: MUTAN
# =========================
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

# =========================
# Alternative Fusion Heads
# =========================
class ConcatMLPHead(nn.Module):
    def __init__(self, d_p: int, d_d: int, d_out: int, hidden: int = 512, dropout: float = 0.1, act: str = "silu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_p + d_d, hidden),
            _act(act),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out)
        )
    def forward(self, hp, hd):
        return self.net(torch.cat([hp, hd], dim=-1))

class BlockFusionHead(nn.Module):
    """
    简化版 BLOCK：将两路各自线性到多个 block，再做 Hadamard 并拼接。
    """
    def __init__(self, d_p: int, d_d: int, d_out: int, k_blocks: int = 8, block_dim: int = 64, dropout: float = 0.1, act: str = "silu"):
        super().__init__()
        self.k = k_blocks
        self.proj_p = nn.ModuleList([nn.Linear(d_p, block_dim) for _ in range(self.k)])
        self.proj_d = nn.ModuleList([nn.Linear(d_d, block_dim) for _ in range(self.k)])
        self.drop = nn.Dropout(dropout)
        self.act = _act(act)
        self.out = nn.Linear(self.k * block_dim, d_out)
    def forward(self, hp, hd):
        parts = []
        for i in range(self.k):
            up = self.act(self.proj_p[i](hp))
            ud = self.act(self.proj_d[i](hd))
            parts.append(up * ud)
        return self.out(self.drop(torch.cat(parts, dim=-1)))

class MCBFusionHead(nn.Module):
    """
    Multimodal Compact Bilinear Pooling (Count-Sketch + FFT)。
    """
    def __init__(self, d_p: int, d_d: int, d_out: int, sketch_dim: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.sketch_dim = sketch_dim
        # 固定随机哈希（注册为 buffer，确保可复现与不会被优化器更新）
        self.register_buffer("h_p", torch.randint(low=0, high=sketch_dim, size=(d_p,)))
        self.register_buffer("s_p", torch.randint(low=0, high=2, size=(d_p,)) * 2 - 1)
        self.register_buffer("h_d", torch.randint(low=0, high=sketch_dim, size=(d_d,)))
        self.register_buffer("s_d", torch.randint(low=0, high=2, size=(d_d,)) * 2 - 1)
        self.out = nn.Linear(sketch_dim, d_out)
        self.drop = nn.Dropout(dropout)

    def _count_sketch(self, x, h, s):
        B, D = x.shape
        sk = x.new_zeros(B, self.sketch_dim)
        idx = h.view(1, -1).expand(B, D)
        val = x * s.view(1, -1).expand(B, D)
        sk.scatter_add_(dim=1, index=idx, src=val)
        return sk

    def forward(self, hp, hd):
        sp = self._count_sketch(hp, self.h_p, self.s_p)
        sd = self._count_sketch(hd, self.h_d, self.s_d)
        fp = torch.fft.rfft(sp, n=self.sketch_dim, dim=1)
        fd = torch.fft.rfft(sd, n=self.sketch_dim, dim=1)
        fused = torch.fft.irfft(fp * fd, n=self.sketch_dim, dim=1)
        fused = torch.sign(fused) * torch.sqrt(torch.clamp(torch.abs(fused), min=1e-8))
        fused = F.normalize(fused, p=2, dim=1)
        return self.out(self.drop(fused))

def build_fusion_head(name: str, d_p: int, d_d: int, d_out: int, **kw):
    name = (name or "mutan").lower()
    if name == "mutan":
        return Mutan(dx=d_p, dy=d_d, dz=d_out, rank=kw.get("rank", 10), dropout=kw.get("dropout", 0.1), act=kw.get("act", "tanh"))
    if name == "concat":
        return ConcatMLPHead(d_p, d_d, d_out, hidden=kw.get("hidden", 512), dropout=kw.get("dropout", 0.1), act=kw.get("act", "silu"))
    if name == "block":
        return BlockFusionHead(d_p, d_d, d_out, k_blocks=kw.get("k_blocks", 8), block_dim=kw.get("block_dim", 64), dropout=kw.get("dropout", 0.1), act=kw.get("act", "silu"))
    if name == "mcb":
        return MCBFusionHead(d_p, d_d, d_out, sketch_dim=kw.get("sketch_dim", 8192), dropout=kw.get("dropout", 0.1))
    raise ValueError(f"Unknown fusion head: {name}")

# =========================
# UGCA (vector-level, V1)
# =========================
class UGCAUnit(nn.Module):
    """Gated fusion of two global vectors -> d-model"""
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

@dataclass
class ModelCfg:
    d_protein: int = 1280   # ESM2
    d_molclr: int = 300
    d_chem: int = 384
    d_model: int = 512
    dropout: float = 0.1
    act: str = "silu"
    fusion: str = "mutan"
    mutan_rank: int = 10
    mutan_dim: int = 512
    head_hidden: int = 512

class UGCAModel(nn.Module):
    """V1 (vector-level)"""
    def __init__(self, cfg: ModelCfg | Dict):
        super().__init__()
        self.cfg = cfg = ModelCfg(**cfg) if isinstance(cfg, dict) else cfg
        d = cfg.d_model
        self.dropout = nn.Dropout(cfg.dropout)
        self.ugca_drug = UGCAUnit(da=cfg.d_molclr, db=cfg.d_chem, d=d, dropout=cfg.dropout, act=cfg.act)
        self.proj_prot = nn.Linear(cfg.d_protein, d, bias=False)
        self.fusion_name = cfg.fusion
        self.fusion = build_fusion_head(cfg.fusion, d_p=d, d_d=d, d_out=cfg.mutan_dim, rank=cfg.mutan_rank, dropout=cfg.dropout)
        self.head = nn.Sequential(
            LayerNorm1d(cfg.mutan_dim),
            nn.Linear(cfg.mutan_dim, cfg.head_hidden),
            _act(cfg.act),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )
        self.apply(self._init_weights)
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
        joint = self.fusion(prot, drug)                       # [B, mutan_dim or alike]
        logit = self.head(joint).squeeze(-1)                  # [B]
        self._last_gd, self._last_gp = None, None
        return logit

    def last_gates(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._last_gd, self._last_gp

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

def compute_gate(mu: torch.Tensor, var: torch.Tensor, mode: str = "mu_times_evi", lam: float = 1.0, eps: float = 1e-6):
    mode = (mode or "mu_times_evi").lower()
    if mode == "evi_only":
        g = torch.exp(-lam * var.clamp_min(0.0))
    elif mode == "mu_only":
        g = torch.sigmoid(mu)
    else:  # mu_times_evi
        g = torch.sigmoid(mu) * torch.exp(-lam * var.clamp_min(0.0))
    return g.clamp(min=eps, max=1.0)

class EvidentialTokenGate(nn.Module):
    """
    Normal-Inverse-Gamma (NIG) gate:
      (mu, v, alpha, beta) -> epistemic var = beta / (v * (alpha - 1))
      gate = compute_gate(mu, var, mode, lam)
    """
    def __init__(self, d: int, lam: float = 2.0, mode: str = "mu_times_evi", g_min: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(d, 4, bias=True)
        self.lam = float(lam)
        self.eps = float(eps)
        self.mode = (mode or "mu_times_evi").lower()
        self.g_min = float(g_min)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)                    # (B,T,4)
        mu, v_raw, a_raw, b_raw = torch.chunk(h, 4, dim=-1)
        v = F.softplus(v_raw) + self.eps    # >0
        alpha = F.softplus(a_raw) + 1.0 + self.eps
        beta  = F.softplus(b_raw) + self.eps
        var_ep = beta / (v * (alpha - 1.0))
        g = compute_gate(mu, var_ep, mode=self.mode, lam=self.lam, eps=self.g_min)
        return g

def _smooth_gate(g: torch.Tensor) -> torch.Tensor:
    """Lightweight 1D smoothing on token axis with kernel [0.25, 0.5, 0.25]. g:(B,T,1)"""
    if g.dim() != 3 or g.size(-1) != 1:
        return g
    B, T, _ = g.shape
    if T < 3:
        return g
    weight = torch.tensor([0.25, 0.5, 0.25], dtype=g.dtype, device=g.device).view(1,1,3)
    x = g.transpose(1,2)  # (B,1,T)
    x = F.pad(x, (1,1), mode="replicate")
    y = F.conv1d(x, weight, padding=0)
    return y.transpose(1,2)

class GatedCrossAttn(nn.Module):
    """
    Bidirectional cross-attn with pre-softmax log-gate bias.
    score_xy = (Qx Ky^T)/sqrt(dh) + log(gx) + log(gy)^T
    """
    def __init__(self, d: int, nhead: int = 4, dropout: float = 0.1,
                 gate_type: str = "evidential", gate_lambda: float = 2.0,
                 gate_mode: str = "mu_times_evi", g_min: float = 0.05,
                 smooth_g: bool = False, attn_temp: float = 1.0):
        super().__init__()
        assert d % nhead == 0
        self.d, self.h = d, nhead
        self.dk = d // nhead
        self.attn_temp = float(attn_temp)
        # QKV
        self.qx = nn.Linear(d, d, bias=False)
        self.kx = nn.Linear(d, d, bias=False)
        self.vx = nn.Linear(d, d, bias=False)
        self.qy = nn.Linear(d, d, bias=False)
        self.ky = nn.Linear(d, d, bias=False)
        self.vy = nn.Linear(d, d, bias=False)
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
        self.out_x = nn.Linear(d, d, bias=False)
        self.out_y = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln_x = nn.LayerNorm(d)
        self.ln_y = nn.LayerNorm(d)
        self.ff_x = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Dropout(dropout))
        self.ff_y = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Dropout(dropout))
        # Buffers to expose attention entropy (optional)
        self.last_entropy_x = None
        self.last_entropy_y = None

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = t.shape
        return t.view(B, T, self.h, self.dk).transpose(1, 2)  # (B,h,T,dk)

    def _merge(self, t: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, self.d)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor,
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
        if y_mask is not None:
            mask_y = (~y_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,Ny)
            scores_xy = scores_xy.masked_fill(mask_y, float("-inf"))
        attn_xy = torch.softmax(scores_xy, dim=-1)
        self.last_entropy_x = (-attn_xy * (torch.log(attn_xy + 1e-8))).sum(dim=-1).mean()  # mean over (B,h,Nx)
        attn_xy = self.dropout(attn_xy)
        out_x = torch.matmul(attn_xy, vy)
        out_x = self._merge(out_x)
        out_x = self.out_x(out_x)
        x2 = self.ln_x(x + out_x)
        x2 = self.ln_x(x2 + self.ff_x(x2))

        # y -> x
        scores_yx = torch.matmul(qy, kx.transpose(-2, -1)) / (math.sqrt(self.dk) * max(self.attn_temp, 1e-6))
        if gate_enabled:
            scores_yx = scores_yx + log_gy.unsqueeze(1) + log_gx.transpose(1, 2).unsqueeze(1)
        if x_mask is not None:
            mask_x = (~x_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,Nx)
            scores_yx = scores_yx.masked_fill(mask_x, float("-inf"))
        attn_yx = torch.softmax(scores_yx, dim=-1)
        self.last_entropy_y = (-attn_yx * (torch.log(attn_yx + 1e-8))).sum(dim=-1).mean()
        attn_yx = self.dropout(attn_yx)
        out_y = torch.matmul(attn_yx, vx)
        out_y = self._merge(out_y)
        out_y = self.out_y(out_y)
        y2 = self.ln_y(y + out_y)
        y2 = self.ln_y(y2 + self.ff_y(y2))

        if gate_enabled:
            gd = gx.squeeze(-1); gp = gy.squeeze(-1)
        else:
            gd = gp = None
        return x2, y2, gd, gp

def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.float()
    s = (x * m.unsqueeze(-1)).sum(dim=1)
    d = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    return s / d

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

# =========================
# Drug composer (pool->concat)
# =========================
class DrugGlobalComposer(nn.Module):
    """
    将池化后的药物原子级表示 h_d_atom 与 化学向量 h_chem 做融合：
      - mode='pool_concat' : h_d = LN(W([h_d_atom; h_chem]))
      - mode='inject'      : 回退到“化学向量注入 token”的旧路径
    """
    def __init__(self, d_in_atom: int, d_in_chem: int, d_out: int,
                 mode: str = "pool_concat", strict_ln: bool = True):
        super().__init__()
        assert mode in ["pool_concat", "inject"]
        self.mode = mode
        self.strict_ln = strict_ln
        if mode == "pool_concat":
            self.proj = nn.Linear(d_in_atom + d_in_chem, d_out, bias=True)
            self.ln = nn.LayerNorm(d_out) if strict_ln else nn.Identity()
        else:
            self.proj = None
            self.ln = nn.Identity()

    def forward(self, h_d_atom: torch.Tensor, h_chem_raw: torch.Tensor,
                inject_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        h_d_atom: [B, D]  池化后的药物序列向量
        h_chem_raw: [B, C]  化学全局原始向量（非投影）
        inject_tokens: [B, Ld, D]（当 mode='inject' 时必传）
        """
        if self.mode == "pool_concat":
            x = torch.cat([h_d_atom, h_chem_raw], dim=-1)
            return self.ln(self.proj(x))   # 严格对齐 concat+Linear(+LN)
        else:
            assert inject_tokens is not None, "inject mode requires inject_tokens!"
            # 简单线性映射到 token 维度并逐 token 相加（随机权重会在外层初始化中重设）
            B, Ld, D = inject_tokens.shape
            W = torch.empty(D, h_chem_raw.shape[-1], device=inject_tokens.device)
            nn.init.normal_(W, mean=0.0, std=0.02)
            proj_chem = F.linear(h_chem_raw, weight=W).unsqueeze(1).expand(B, Ld, D)
            return inject_tokens + proj_chem

@dataclass
class SeqCfg:
    d_protein: int = 1280
    d_molclr: int = 300
    d_chem: int = 384
    d_model: int = 512
    nhead: int = 4
    nlayers: int = 2
    dropout: float = 0.1
    # fusion
    fusion: str = "mutan"
    mutan_dim: int = 512
    mutan_rank: int = 10
    head_hidden: int = 512
    # Gate
    gate_type: str = "evidential"      # evidential | mlp
    gate_mode: str = "mu_times_evi"    # mu_times_evi | evi_only | mu_only
    gate_lambda: float = 2.0           # uncertainty suppression λ
    g_min: float = 0.05                # clamp min
    smooth_g: bool = False
    topk_ratio: float = 0.0            # 可用训练循环来 warm-up
    # Cross-attn
    attn_temp: float = 1.0
    # Pooling
    pool_type: str = "attn"            # mean | attn
    # Drug fuse
    drug_fuse: str = "pool_concat"     # pool_concat | inject
    strict_concat_ln: bool = True
    chem_dim: int = 384
    # Regularization
    entropy_reg: float = 0.0           # 可选：注意力熵正则

class UGCASeqModel(nn.Module):
    """
    Sequence-level UGCA:
      - Project tokens to shared dim d
      - L layers of gated bidirectional cross-attn
      - Pool to global vectors (mean or cross-attn pooling)
      - Drug: pool_then_concat(+LN) or inject-to-token
      - Fusion head (mutan/concat/block/mcb) -> classifier
    """
    def __init__(self, cfg: SeqCfg | Dict):
        super().__init__()
        self.cfg = cfg = SeqCfg(**cfg) if isinstance(cfg, dict) else cfg
        d = cfg.d_model
        # Projections
        self.proj_p = nn.Linear(cfg.d_protein, d, bias=False)
        self.proj_d = nn.Linear(cfg.d_molclr,  d, bias=False)
        self.proj_c = nn.Linear(cfg.d_chem,    d, bias=False)  # 仅在 inject 模式下使用

        # Gated cross-attn layers
        self.layers = nn.ModuleList([
            GatedCrossAttn(d=d, nhead=cfg.nhead, dropout=cfg.dropout,
                           gate_type=cfg.gate_type, gate_lambda=cfg.gate_lambda,
                           gate_mode=cfg.gate_mode, g_min=cfg.g_min,
                           smooth_g=cfg.smooth_g, attn_temp=cfg.attn_temp)
            for _ in range(cfg.nlayers)
        ])
        self.topk_ratio = float(getattr(cfg, "topk_ratio", 0.0))
        self.gate_enabled = True
        self.entropy_reg = float(cfg.entropy_reg)

        # Pooling
        self.pool_type = str(cfg.pool_type or "attn").lower()
        if self.pool_type == "attn":
            self.pool_d = AttnPool(d, dropout=cfg.dropout)
            self.pool_p = AttnPool(d, dropout=cfg.dropout)

        # Drug composer (池化后再 concat+LN)
        self.drug_fuse = str(cfg.drug_fuse or "pool_concat")
        self.chem_dim = int(cfg.chem_dim)
        self.drug_global = DrugGlobalComposer(
            d_in_atom=d, d_in_chem=self.chem_dim, d_out=d,
            mode=self.drug_fuse, strict_ln=bool(cfg.strict_concat_ln)
        )

        # Fusion + head
        self.fusion_name = cfg.fusion
        self.fusion = build_fusion_head(cfg.fusion, d_p=d, d_d=d, d_out=cfg.mutan_dim,
                                        rank=cfg.mutan_rank, dropout=cfg.dropout)
        self.head = nn.Sequential(
            LayerNorm1d(cfg.mutan_dim),
            nn.Linear(cfg.mutan_dim, cfg.head_hidden),
            nn.SiLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1)
        )

        # For training loop: expose latest gates and attn entropies
        self._last_gd = None
        self._last_gp = None
        self._last_entropy = 0.0
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

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
        # 1) project tokens to d; 若选择 inject，则把化学向量投影后注入到 token
        P = self.proj_p(v_prot)  # (B,M,d)
        D = self.proj_d(v_mol)   # (B,N,d)

        if self.drug_fuse == "inject":
            D = D + self.proj_c(v_chem).unsqueeze(1)  # 旧路径：token 注入

        # 2) L 层 gated 双向 cross-attn（可选 top-k mask 稀疏）
        gd = gp = None
        curr_ratio = self.topk_ratio if topk_ratio is None else float(topk_ratio)
        mP, mD = m_prot, m_mol
        for layer in self.layers:
            D, P, gd, gp = layer(D, mD, P, mP, gate_enabled=self.gate_enabled)
            if curr_ratio > 0.0 and self.gate_enabled:
                with torch.no_grad():
                    if gd is not None: mD = self._apply_topk(gd, mD, curr_ratio)
                    if gp is not None: mP = self._apply_topk(gp, mP, curr_ratio)

        self._last_gd, self._last_gp = gd, gp

        # 3) 池化（先池化，再与化学全局向量 concat+LN）
        if self.pool_type == "attn":
            q_p = _masked_mean(P, mP)    # (B,d)
            q_d = _masked_mean(D, mD)
            hd_atom = self.pool_d(keys=D, values=D, mask=mD, query_vec=q_p)  # (B,d)
            hp = self.pool_p(keys=P, values=P, mask=mP, query_vec=q_d)
        else:
            hd_atom = _masked_mean(D, mD)
            hp = _masked_mean(P, mP)

        if self.drug_fuse == "pool_concat":
            hd = self.drug_global(hd_atom, v_chem)           # 严格 concat+LN
        else:
            # inject 路径在前面已把 chem 注入 token，这里再池化一次得到全局
            hd = hd_atom

        # 4) 融合头 + 分类头
        z = self.fusion(hp, hd)                              # (B, mutan_dim / etc.)
        logit = self.head(z).squeeze(-1)                     # (B,)
        return logit

    def last_gates(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._last_gd, self._last_gp

# =========================
# Factory
# =========================
def build_model(cfg: Dict) -> nn.Module:
    """
    train.py 会调用 build_model(cfg)。
    支持：
      - sequence=False: UGCAModel (向量级)
      - sequence=True : UGCASeqModel (序列级)
    """
    if bool(cfg.get("sequence", False)):
        mcfg = SeqCfg(
            d_protein=int(cfg.get("d_protein", 1280)),
            d_molclr=int(cfg.get("d_molclr", 300)),
            d_chem=int(cfg.get("d_chem", 384)),
            d_model=int(cfg.get("d_model", 512)),
            nhead=int(cfg.get("nhead", 4)),
            nlayers=int(cfg.get("nlayers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            # fusion
            fusion=str(cfg.get("fusion", "mutan")),
            mutan_rank=int(cfg.get("mutan_rank", 10)),
            mutan_dim=int(cfg.get("mutan_dim", 512)),
            head_hidden=int(cfg.get("head_hidden", 512)),
            # gate
            gate_type=str(cfg.get("gate_type", "evidential")),
            gate_mode=str(cfg.get("gate_mode", "mu_times_evi")),
            gate_lambda=float(cfg.get("gate_lambda", 2.0)),
            g_min=float(cfg.get("g_min", 0.05)),
            smooth_g=bool(cfg.get("smooth_g", False)),
            topk_ratio=float(cfg.get("topk_ratio", 0.0)),
            # cross-attn
            attn_temp=float(cfg.get("attn_temp", 1.0)),
            # pooling
            pool_type=str(cfg.get("pool_type", "attn")),
            # drug fuse
            drug_fuse=str(cfg.get("drug_fuse", "pool_concat")),
            strict_concat_ln=bool(cfg.get("strict_concat_ln", True)),
            chem_dim=int(cfg.get("chem_dim", cfg.get("d_chem", 384))),
            # regularization
            entropy_reg=float(cfg.get("entropy_reg", 0.0))
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
            fusion=str(cfg.get("fusion", "mutan")),
            mutan_rank=int(cfg.get("mutan_rank", 10)),
            mutan_dim=int(cfg.get("mutan_dim", 512)),
            head_hidden=int(cfg.get("head_hidden", 512)),
        )
        return UGCAModel(mcfg)
