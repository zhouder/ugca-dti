# model.py
import math
from typing import Tuple, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_fill(logits: torch.Tensor, key_padding_mask: torch.Tensor, value: float):
    """
    logits: [B, H, Lq, Lk]
    key_padding_mask: [B, Lk]  (1 = valid, 0 = pad)
    """
    if key_padding_mask is None:
        return logits
    mask = (key_padding_mask == 0).unsqueeze(1).unsqueeze(2)  # [B,1,1,Lk]
    return logits.masked_fill(mask, value)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or (4 * d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class UncertaintyGate(nn.Module):
    """
    NIG 证据头 -> gate g_i ∈ (0,1]，支持三种模式
    """
    def __init__(
        self,
        d_model: int,
        hidden: Optional[int] = None,
        dropout: float = 0.1,
        gate_mode: Literal["mu_times_evi", "mu_only", "evi_only"] = "mu_times_evi",
        lamb: float = 1.0,
        g_min: float = 1e-3,
    ):
        super().__init__()
        hidden = hidden or d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4),  # mu, nu, alpha, beta
        )
        self.gate_mode = gate_mode
        self.lamb = lamb
        self.g_min = g_min

    def forward(self, h: torch.Tensor):
        out = self.mlp(h)  # [B,L,4]
        mu, nu, alpha, beta = out.unbind(dim=-1)
        nu = F.softplus(nu) + 1e-4
        alpha = F.softplus(alpha) + 1.0 + 1e-4
        beta = F.softplus(beta) + 1e-4

        sigma_e2 = beta / (nu * (alpha - 1.0))
        if self.gate_mode == "mu_only":
            g = torch.sigmoid(mu)
        elif self.gate_mode == "evi_only":
            g = torch.exp(-self.lamb * sigma_e2)
        else:
            g = torch.sigmoid(mu) * torch.exp(-self.lamb * sigma_e2)

        return torch.clamp(g, min=self.g_min, max=1.0)


class GatedCrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, temp: float = 0.8):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.temp = temp

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q_src, KV_src, key_gate, key_padding_mask=None):
        B, Lq, _ = Q_src.shape
        Lk = KV_src.shape[1]

        q = self.w_q(Q_src).view(B, Lq, self.nhead, self.d_head).transpose(1, 2)  # [B,H,Lq,Dh]
        k = self.w_k(KV_src).view(B, Lk, self.nhead, self.d_head).transpose(1, 2)
        v = self.w_v(KV_src).view(B, Lk, self.nhead, self.d_head).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,Lq,Lk]
        logits = logits + torch.log(key_gate).unsqueeze(1).unsqueeze(2)          # add log(gate)
        logits = _masked_fill(logits, key_padding_mask, value=-1e9)
        logits = logits / self.temp

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.w_o(ctx)


class UGCABlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, gate_mode="mu_times_evi", lamb=1.0, g_min=1e-3, temp=0.8):
        super().__init__()
        self.gate_d = UncertaintyGate(d_model, dropout=dropout, gate_mode=gate_mode, lamb=lamb, g_min=g_min)
        self.gate_p = UncertaintyGate(d_model, dropout=dropout, gate_mode=gate_mode, lamb=lamb, g_min=g_min)

        self.ca_d_from_p = GatedCrossAttention(d_model, nhead, dropout=dropout, temp=temp)
        self.ca_p_from_d = GatedCrossAttention(d_model, nhead, dropout=dropout, temp=temp)

        self.dropout = nn.Dropout(dropout)
        self.ln_d_1 = nn.LayerNorm(d_model)
        self.ln_p_1 = nn.LayerNorm(d_model)
        self.ffn_d = FeedForward(d_model, dropout=dropout)
        self.ffn_p = FeedForward(d_model, dropout=dropout)
        self.ln_d_2 = nn.LayerNorm(d_model)
        self.ln_p_2 = nn.LayerNorm(d_model)

    def forward(self, H_D, H_P, mask_d, mask_p):
        g_d = self.gate_d(H_D)
        g_p = self.gate_p(H_P)

        d_from_p = self.ca_d_from_p(H_D, H_P, key_gate=g_p, key_padding_mask=mask_p)
        p_from_d = self.ca_p_from_d(H_P, H_D, key_gate=g_d, key_padding_mask=mask_d)

        H_D = self.ln_d_1(H_D + self.dropout(d_from_p))
        H_P = self.ln_p_1(H_P + self.dropout(p_from_d))
        H_D = self.ln_d_2(H_D + self.ffn_d(H_D))
        H_P = self.ln_p_2(H_P + self.ffn_p(H_P))
        return H_D, H_P


class MeanMaxPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x, mask):
        mask_f = mask.float()
        sum_x = torch.einsum("bld,bl->bd", x, mask_f)
        denom = mask_f.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        mean = sum_x / denom

        neg_inf = torch.finfo(x.dtype).min
        max_val = x.masked_fill(mask.unsqueeze(-1) == 0, neg_inf).max(dim=1).values
        return self.proj(torch.cat([mean, max_val], dim=-1))


class AttnPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=True)
        self.w = nn.Linear(d_model, 1, bias=False)

    def forward(self, x, mask):
        s = self.w(torch.tanh(self.W(x))).squeeze(-1)  # [B,L]
        s = s.masked_fill(mask == 0, -1e9)
        a = F.softmax(s, dim=-1)
        return torch.einsum("bl,bld->bd", a, x)


class MHAttnPooling(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.mha = nn.MultiheadAttention(d_model, num_heads=nhead, batch_first=True)

    def forward(self, x, mask):
        B, _, d = x.shape
        q = self.query.expand(B, 1, d)
        kpm = (mask == 0)
        z, _ = self.mha(q, x, x, key_padding_mask=kpm, need_weights=False)
        return z.squeeze(1)


class BranchMLP(nn.Module):
    def __init__(self, d_in, d_out, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )
    def forward(self, x): return self.net(x)


class FusionHeadMatchMLP(nn.Module):
    def __init__(self, d_fuse, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * d_fuse, 2 * d_fuse),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_fuse, d_fuse),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fuse, 1),
        )

    def forward(self, f_d, f_p):
        u = torch.cat([f_d, f_p, f_d * f_p, torch.abs(f_d - f_p)], dim=-1)
        return self.net(u).squeeze(-1)


class FusionHeadConcatMLP(nn.Module):
    def __init__(self, d_fuse, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_fuse, 2 * d_fuse),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_fuse, 1),
        )
    def forward(self, f_d, f_p):
        return self.net(torch.cat([f_d, f_p], dim=-1)).squeeze(-1)


class UGCADTI(nn.Module):
    def __init__(
        self,
        d_mol_in: int,
        d_prot_in: int,
        d_model: int = 128,
        nlayers: int = 2,
        nhead: int = 4,
        d_fuse: int = 256,
        pooling: Literal["meanmax", "attn", "mh-attn"] = "attn",
        fusion_head: Literal["match-mlp", "concat-mlp"] = "match-mlp",
        gate_mode: Literal["mu_times_evi", "mu_only", "evi_only"] = "mu_times_evi",
        lamb: float = 1.0,
        temp: float = 0.8,
        g_min: float = 1e-3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.proj_mol = nn.Linear(d_mol_in, d_model)
        self.proj_prot = nn.Linear(d_prot_in, d_model)

        self.layers = nn.ModuleList([
            UGCABlock(d_model, nhead, dropout=dropout, gate_mode=gate_mode, lamb=lamb, g_min=g_min, temp=temp)
            for _ in range(nlayers)
        ])

        if pooling == "meanmax":
            self.pool = MeanMaxPooling(d_model)
        elif pooling == "attn":
            self.pool = AttnPooling(d_model)
        elif pooling == "mh-attn":
            self.pool = MHAttnPooling(d_model, nhead=nhead)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.branch_d = BranchMLP(d_model, d_fuse, dropout)
        self.branch_p = BranchMLP(d_model, d_fuse, dropout)
        self.fusion = FusionHeadMatchMLP(d_fuse, dropout) if fusion_head == "match-mlp" else FusionHeadConcatMLP(d_fuse, dropout)

    def forward(self, mol, prot, mask_d, mask_p):
        H_D = self.proj_mol(mol)
        H_P = self.proj_prot(prot)
        for layer in self.layers:
            H_D, H_P = layer(H_D, H_P, mask_d, mask_p)
        z_d = self.pool(H_D, mask_d)
        z_p = self.pool(H_P, mask_p)
        f_d = self.branch_d(z_d)
        f_p = self.branch_p(z_p)
        return self.fusion(f_d, f_p)  # [B]
