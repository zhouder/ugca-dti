from __future__ import annotations
import torch
import torch.nn as nn
from .evidential import NIGHead, GateFromUncertainty, BudgetRegularizer, EDLRegularizer

class Projector(nn.Module):
    def __init__(self, in_dim: int|None, d: int, p: float):
        super().__init__()
        if in_dim is None:
            in_dim = d
        self.net = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.Dropout(p), nn.GELU())
    def forward(self, x): return self.net(x)

class MHCA(nn.Module):
    def __init__(self, d: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.h = heads
        self.dh = d // heads
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d)
    def forward(self, qx, kx, mask_q, mask_k, log_g_bias=None):
        B,Nq,D = qx.shape
        Nk = kx.shape[1]
        q = self.q(qx).view(B,Nq,self.h,self.dh).transpose(1,2)
        k = self.k(kx).view(B,Nk,self.h,self.dh).transpose(1,2)
        v = self.v(kx).view(B,Nk,self.h,self.dh).transpose(1,2)
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) / (self.dh ** 0.5)
        mask = mask_k.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(~mask, float('-inf'))
        if log_g_bias is not None:
            attn = attn + log_g_bias.unsqueeze(1).unsqueeze(2)
        w = torch.softmax(attn, dim=-1)
        ctx = torch.einsum('bhqk,bhkd->bhqd', w, v).transpose(1,2).contiguous().view(B,Nq,D)
        x = self.out(ctx)
        x = self.drop(x)
        return self.ln(x + qx), w

class UGCA(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float,
                 k_target: float, g_min: float, rho: float, budget_lambda: float,
                 topk_enable: bool, topk_ratio: float, layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([MHCA(d, heads, dropout) for _ in range(layers)])
        self.nig = NIGHead(d)
        self.gate = GateFromUncertainty(k=k_target, g_min=g_min)
        self.reg_budget = BudgetRegularizer(rho=rho, lam=budget_lambda)
        self.reg_edl = EDLRegularizer(lam=1e-3)
        self.topk_enable = topk_enable
        self.topk_ratio = topk_ratio

    def _apply_topk(self, g: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.topk_enable: return g
        B,N = g.shape
        out = torch.zeros_like(g)
        k = (mask.sum(dim=1).float() * self.topk_ratio).clamp(min=1).long()
        for b in range(B):
            nb = k[b].item()
            vals, idx = torch.topk(g[b], nb)
            out[b, idx] = g[b, idx]
        return out * mask.float()

    def forward(self, H_D, mask_D, H_P, mask_P, stage: str = 'B', k_now: float|None=None):
        _, _, alpha_D, beta_D, s2_D = self.nig(H_D)
        _, _, alpha_P, beta_P, s2_P = self.nig(H_P)
        if stage == 'A':
            log_g_D = log_g_P = None
            g_D = mask_D.float()
            g_P = mask_P.float()
            loss_reg = H_D.new_tensor(0.)
        else:
            if k_now is not None:
                g_tmp_D = torch.exp(- k_now * s2_D)
                g_tmp_P = torch.exp(- k_now * s2_P)
                g_D = torch.clamp(g_tmp_D, min=1e-6, max=1.0) * mask_D.float()
                g_P = torch.clamp(g_tmp_P, min=1e-6, max=1.0) * mask_P.float()
            else:
                g_D = self.gate(s2_D, mask_D)
                g_P = self.gate(s2_P, mask_P)
            g_D = self._apply_topk(g_D, mask_D)
            g_P = self._apply_topk(g_P, mask_P)
            log_g_D = torch.log(g_D + 1e-8)
            log_g_P = torch.log(g_P + 1e-8)
            loss_reg = self.reg_budget(g_D, mask_D) + self.reg_budget(g_P, mask_P) +                        self.reg_edl(alpha_D, beta_D) + self.reg_edl(alpha_P, beta_P)

        Xd, Xp = H_D, H_P
        for layer in self.layers:
            Xd, _ = layer(Xd, Xp, mask_D, mask_P, log_g_bias=log_g_P)
            Xp, _ = layer(Xp, Xd, mask_P, mask_D, log_g_bias=log_g_D)
        return Xd, Xp, g_D, g_P, loss_reg
