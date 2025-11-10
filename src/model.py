# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import math, torch
import torch.nn as nn
import torch.nn.functional as F

# ================= 配置 =================
@dataclass
class ModelCfg:
    # 输入维
    d_protein: int = 1280
    d_molclr:  int = 300
    d_chem:    int = 384
    # 统一交互维
    d_model:   int = 256
    dropout:   float = 0.1
    act:       str = "silu"

    # UGCA（不确定性门控协同注意）
    ugca_layers: int = 2       # 层数
    ugca_heads:  int = 8       # 注意力头
    g_min:       float = 0.05  # 门控下界
    rho:         float = 0.6   # （保留接口，不在本文件使用）
    k_init:      float = 0.0   # warm-up 起始 k
    k_target:    float = 15.0  # 目标 k
    topk_ratio:  float = 0.0   # （保留接口，不在本文件使用）

    # MUTAN
    mutan_rank: int = 20
    mutan_dim:  int = 256
    mutan_drop: float = 0.2

    # 分类头
    head_hidden: int = 256
    head_drop:   float = 0.2

def _activation(name: str):
    return dict(relu=nn.ReLU(), gelu=nn.GELU(), silu=nn.SiLU()).get(name.lower(), nn.SiLU())

# ========== Evidential Head：NIG -> sigma^2_E -> g ==========
class EvidentialHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(d, 4)  # gamma, nu, alpha, beta
        )
    def forward(self, x: torch.Tensor, k: float, g_min: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, d)
        return: g  (B, T) in [g_min, 1], sigma2 (B, T)
        """
        p = self.mlp(x)                    # (B,T,4)
        gamma, nu, alpha, beta = torch.chunk(p, 4, dim=-1)
        nu    = F.softplus(nu)   + 1e-6
        alpha = F.softplus(alpha)+ 1.0     # >1
        beta  = F.softplus(beta) + 1e-6
        sigma2 = beta / (nu * (alpha - 1.0))         # 认知不确定性 σ_E^2
        g = torch.exp(-k * sigma2).squeeze(-1)       # (B,T)
        g = torch.clamp(g, min=g_min, max=1.0)
        return g, sigma2.squeeze(-1)

# ========== 线性投影 ==========
class Projector(nn.Module):
    def __init__(self, din: int, d: int, drop: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, d), nn.LayerNorm(d), nn.Dropout(drop))
    def forward(self, x): return self.net(x)

# ========== 多头 Cross-Attn（logit+log(g) 偏置） ==========
class CrossAttn(nn.Module):
    def __init__(self, d: int, heads: int, drop: float):
        super().__init__()
        assert d % heads == 0
        self.d = d; self.h = heads; self.dh = d // heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.do = nn.Dropout(drop)
        self.ln = nn.LayerNorm(d)

    def forward(self, Q: torch.Tensor, KV: torch.Tensor, log_gate_bias: Optional[torch.Tensor]=None):
        """
        Q:  (B, M, d)
        KV: (B, N, d)
        log_gate_bias: (B,N) or (B,1,N)，在 softmax 前加到 logits
        """
        B,M,_ = Q.shape; N = KV.size(1)
        q = self.q(Q).view(B,M,self.h,self.dh).transpose(1,2)   # (B,h,M,dh)
        k = self.k(KV).view(B,N,self.h,self.dh).transpose(1,2)  # (B,h,N,dh)
        v = self.v(KV).view(B,N,self.h,self.dh).transpose(1,2)  # (B,h,N,dh)

        scores = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.dh)  # (B,h,M,N)
        if log_gate_bias is not None:
            if log_gate_bias.dim() == 2:        # (B,N)
                log_gate_bias = log_gate_bias.unsqueeze(1).unsqueeze(2)  # -> (B,1,1,N)
            scores = scores + log_gate_bias

        attn = torch.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v)                         # (B,h,M,dh)
        out  = out.transpose(1,2).contiguous().view(B,M,self.d)
        out  = self.o(out)
        out  = self.do(out)
        return self.ln(Q + out)                              # 残差+LN

# ========== UGCA 层 ==========
class UGCALayer(nn.Module):
    def __init__(self, d: int, heads: int, drop: float, g_min: float):
        super().__init__()
        self.evi_drug = EvidentialHead(d)
        self.evi_prot = EvidentialHead(d)
        self.ca_p2d   = CrossAttn(d, heads, drop)  # protein->drug
        self.ca_d2p   = CrossAttn(d, heads, drop)  # drug->protein
        self.ffn_d    = nn.Sequential(
            nn.Linear(d, d*4), _activation("silu"), nn.Dropout(drop),
            nn.Linear(d*4, d), nn.Dropout(drop), nn.LayerNorm(d)
        )
        self.ffn_p    = nn.Sequential(
            nn.Linear(d, d*4), _activation("silu"), nn.Dropout(drop),
            nn.Linear(d*4, d), nn.Dropout(drop), nn.LayerNorm(d)
        )
        self.g_min    = g_min

    def forward(self, Hd, Hp, k: float):
        # Evidential -> 门控
        gd, _ = self.evi_drug(Hd, k, self.g_min)  # (B,N)
        gp, _ = self.evi_prot(Hp, k, self.g_min)  # (B,M)
        # 在 logits 前加 log(g)
        Hp2 = self.ca_p2d(Hp, Hd, torch.log(gd+1e-12))
        Hd2 = self.ca_d2p(Hd, Hp, torch.log(gp+1e-12))
        # FFN
        Hd = self.ffn_d(Hd2)
        Hp = self.ffn_p(Hp2)
        return Hd, Hp, gd, gp

# ========== MUTAN ==========
class MUTAN(nn.Module):
    def __init__(self, d: int, rank: int, z_dim: int, drop: float):
        super().__init__()
        self.U = nn.Parameter(torch.randn(rank, d, z_dim) * 0.02)
        self.V = nn.Parameter(torch.randn(rank, d, z_dim) * 0.02)
        self.drop = nn.Dropout(drop)
        self.ln   = nn.LayerNorm(z_dim)
    def forward(self, hd: torch.Tensor, hp: torch.Tensor):
        outs = []
        for r in range(self.U.size(0)):
            u = torch.matmul(hd, self.U[r])   # (B, z_dim)
            v = torch.matmul(hp, self.V[r])   # (B, z_dim)
            outs.append(u * v)
        z = torch.relu(torch.stack(outs, dim=0).sum(0))  # (B, z_dim)
        return self.ln(self.drop(z))

# ========== 池化 ==========
class AttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1,1,d)*0.02)
    def forward(self, x: torch.Tensor):
        q = self.q.expand(x.size(0), -1, -1)     # (B,1,d)
        attn = torch.softmax(torch.matmul(q, x.transpose(1,2))/math.sqrt(x.size(-1)), dim=-1)
        return torch.matmul(attn, x).squeeze(1)  # (B,d)

# ========== 总模型 ==========
class UGCAModel(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        d = cfg.d_model
        # 投影
        self.proj_p = Projector(cfg.d_protein, d, cfg.dropout)
        self.proj_d = Projector(cfg.d_molclr,  d, cfg.dropout)
        self.proj_c = Projector(cfg.d_chem,    d, cfg.dropout)
        # UGCA 堆叠
        self.layers = nn.ModuleList([UGCALayer(d, cfg.ugca_heads, cfg.dropout, cfg.g_min) for _ in range(cfg.ugca_layers)])
        # 池化
        self.pool_p = AttnPool(d)
        self.pool_d = AttnPool(d)
        # MUTAN + 头
        self.mutan  = MUTAN(d, cfg.mutan_rank, cfg.mutan_dim, cfg.mutan_drop)
        self.head   = nn.Sequential(
            nn.Linear(cfg.mutan_dim, cfg.head_hidden), _activation("silu"), nn.Dropout(cfg.head_drop),
            nn.Linear(cfg.head_hidden, 1)
        )
        self.cfg = cfg
        # k：注册为 buffer，训练时由外部 set_k 调整
        self.register_buffer("k", torch.tensor(self.cfg.k_init, dtype=torch.float32))

        # 将 pool 后的拼接规整到 mutan_dim
        self._proj_to_mutan = nn.Linear(d + d, cfg.mutan_dim)

    def set_k(self, value: float):
        with torch.no_grad():
            self.k.fill_(float(value))

    def forward(self, vp, vd, vc):
        """
        vp: protein  (B, d_protein) -> (B,1,d)
        vd: molclr   (B, d_molclr)  -> (B,1,d)
        vc: chemberta(B, d_chem)    -> (B,d)  （与 mol/蛋白的序列池化后再融合）
        """
        Hp = self.proj_p(vp).unsqueeze(1)  # (B,1,d)
        Hd = self.proj_d(vd).unsqueeze(1)  # (B,1,d)
        # chem 特征用于与药物池化后的全局向量拼接
        for layer in self.layers:
            Hd, Hp, _, _ = layer(Hd, Hp, float(self.k))

        hd_atom = self.pool_d(Hd)          # (B,d)
        hp      = self.pool_p(Hp)          # (B,d)

        # 药物(原子池化)与蛋白一起送入 MUTAN；chem 先与 hd 拼接后映射到 mutan 维
        hd_plus = torch.cat([hd_atom, self.proj_c(vc)], dim=-1)    # (B, 2d)
        hd = self._proj_to_mutan(hd_plus)                          # (B, mutan_dim)
        hp = nn.Linear(hp.size(-1), self.cfg.mutan_dim, device=hp.device)(hp)

        z  = self.mutan(hd, hp)            # (B, mutan_dim)
        logit = self.head(z).squeeze(-1)   # (B,)
        return logit

# 统一入口
def build_model(cfg_dict: dict) -> nn.Module:
    base = ModelCfg()
    for k,v in cfg_dict.items():
        if hasattr(base, k): setattr(base, k, v)
    return UGCAModel(base)
