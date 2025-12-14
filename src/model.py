# src/model.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utils
# ---------------------------
def masked_mean(x: torch.Tensor, m: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    """
    x: (B, T, D), m: (B, T) bool (True=valid)
    return: (B, D)
    """
    m = m.to(dtype=x.dtype)
    num = (x * m.unsqueeze(-1)).sum(dim=dim)
    den = m.sum(dim=dim).clamp_min(eps).unsqueeze(-1)
    return num / den


def attention_entropy(attn: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    attn: (B, H, Lq, Lk) attention prob
    returns: (B, Lq, 1) normalized entropy in [0, 1]
    """
    p = attn.clamp_min(eps)
    ent = -(p * p.log()).sum(dim=-1)  # (B,H,Lq)
    lk = attn.size(-1)
    ent = ent / (torch.log(torch.tensor(float(lk), device=attn.device)).clamp_min(eps))
    ent = ent.mean(dim=1, keepdim=False)  # (B, Lq)
    return ent.unsqueeze(-1)              # (B, Lq, 1)


# ---------------------------
# PocketGraph Encoder (simple, no PyG dependency)
# ---------------------------
class PocketGraphEncoder(nn.Module):
    """
    输入：残基图
      - node_scalar_feat: (N, Fin)  (你的 npz 里目前是 21 维 AA one-hot)
      - edge_index: (2, E) long
    输出：全局向量 (d_model,)
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fc_self = nn.Linear(hidden_dim, hidden_dim)
        self.fc_nei  = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_scalar: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        node_scalar: (N, Fin) float
        edge_index:  (2, E) long
        """
        if node_scalar is None or node_scalar.numel() == 0:
            return torch.zeros((self.fc_out.out_features,), device=edge_index.device, dtype=torch.float32)

        x = F.relu(self.fc_in(node_scalar))   # (N, hidden)
        x = self.dropout(x)
        N = x.size(0)

        if edge_index is None or edge_index.numel() == 0:
            h = F.relu(self.fc_self(x))
            g = h.mean(dim=0)
            return self.fc_out(g)

        src, dst = edge_index[0], edge_index[1]          # (E,), (E,)
        msg = x[src]                                     # (E, hidden)
        agg = torch.zeros_like(x)                        # (N, hidden)
        agg.index_add_(0, dst, msg)

        deg = torch.zeros((N, 1), device=x.device, dtype=x.dtype)
        one = torch.ones((dst.size(0), 1), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, one)
        agg = agg / deg.clamp_min(1.0)

        h = F.relu(self.fc_self(x) + self.fc_nei(agg))   # (N, hidden)
        h = self.dropout(h)

        g = h.mean(dim=0)                                # (hidden,)
        return self.fc_out(g)                            # (out_dim,)


# ---------------------------
# Cross-Attention with uncertainty-aware residual gating
# ---------------------------
class CrossAttnGate(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        # gate: concat(x, attn_out) -> sigmoid -> scalar gate per token
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor, q_mask: torch.Tensor,   # (B, Lq, D), (B, Lq) bool True=valid
        kv: torch.Tensor, kv_mask: torch.Tensor  # (B, Lk, D), (B, Lk) bool True=valid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
          - updated q (B, Lq, D)
          - attn_entropy_norm (B, Lq, 1)  (可用于正则)
        """
        # MultiheadAttention 的 key_padding_mask: True 表示需要 mask 掉
        key_padding_mask = ~kv_mask
        # q 侧 padding 不需要传给 MHA（它会输出所有位置），但我们后面会用 q_mask 控制 pooling
        attn_out, attn_w = self.mha(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,   # -> (B, H, Lq, Lk)
        )

        ent = attention_entropy(attn_w)            # (B, Lq, 1) in [0,1]
        conf = (1.0 - ent).clamp(0.0, 1.0)         # 越小 entropy => 越“确定”

        g = self.gate(torch.cat([q, attn_out], dim=-1))  # (B, Lq, 1)
        g = g * conf

        # gated residual
        q2 = q + self.dropout(attn_out) * g
        q2 = self.ln(q2)
        q2 = q2 + self.ffn(q2)
        q2 = self.ln(q2)
        return q2, ent


class UGCABlock(nn.Module):
    """
    一层双向 cross-attention + gate:
      P <- D
      D <- P
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.p_from_d = CrossAttnGate(d_model, n_heads, dropout)
        self.d_from_p = CrossAttnGate(d_model, n_heads, dropout)

    def forward(
        self,
        P: torch.Tensor, Pm: torch.Tensor,
        D: torch.Tensor, Dm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        P2, ent_p = self.p_from_d(P, Pm, D, Dm)
        D2, ent_d = self.d_from_p(D, Dm, P, Pm)
        # 返回两个方向的 entropy 方便做正则（可选）
        return P2, D2, (ent_p.mean() + ent_d.mean()) * 0.5


# ---------------------------
# Model configs
# ---------------------------
@dataclass
class SeqCfg:
    # input dims
    d_protein: int = 1280
    d_molclr: int = 300
    d_chem: int = 384

    # model dims
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 2
    dropout: float = 0.1

    # pocket
    use_pocket: bool = False
    pocket_in_dim: int = 21
    pocket_hidden: int = 128

    # regularization
    entropy_reg: float = 0.0   # >0 时对 attention entropy 做正则（越小越“确定”）


@dataclass
class VecCfg:
    d_protein: int = 1280
    d_molclr: int = 300
    d_chem: int = 384
    d_model: int = 512
    dropout: float = 0.1


# ---------------------------
# Sequence model (UGCA v2 + global tokens)
# ---------------------------
class UGCASeqModel(nn.Module):
    """
    forward 输入对齐 datamodule 的输出：
      - 无 pocket: (P, Pm, D, Dm, C)
      - 有 pocket: (P, Pm, D, Dm, C, pocket_list)
    """
    def __init__(self, cfg: SeqCfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # projections to d_model
        self.proj_p = nn.Linear(cfg.d_protein, d, bias=False)
        self.proj_d = nn.Linear(cfg.d_molclr,  d, bias=False)
        self.proj_c = nn.Linear(cfg.d_chem,    d, bias=False)

        # optional pocket graph encoder
        self.use_pocket = bool(cfg.use_pocket)
        self.pocket_enc = None
        if self.use_pocket:
            self.pocket_enc = PocketGraphEncoder(
                in_dim=cfg.pocket_in_dim,
                hidden_dim=cfg.pocket_hidden,
                out_dim=d,
                dropout=cfg.dropout,
            )

        # UGCA blocks
        self.blocks = nn.ModuleList([
            UGCABlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)
        ])

        # head
        self.head = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, 1),
        )

    def _build_graph_token(self, pocket_batch: Optional[List[dict]], device: torch.device) -> Optional[torch.Tensor]:
        if (not self.use_pocket) or (self.pocket_enc is None) or (pocket_batch is None):
            return None
        g_list: List[torch.Tensor] = []
        for pg in pocket_batch:
            if pg is None:
                g_list.append(torch.zeros((self.cfg.d_model,), device=device, dtype=torch.float32))
                continue
            if "node_scalar_feat" not in pg or "edge_index" not in pg:
                g_list.append(torch.zeros((self.cfg.d_model,), device=device, dtype=torch.float32))
                continue
            node_scalar = torch.as_tensor(pg["node_scalar_feat"], dtype=torch.float32, device=device)
            edge_index  = torch.as_tensor(pg["edge_index"], dtype=torch.long, device=device)
            # 防御：edge_index 可能是 (2,E) 也可能被保存成 (E,2)
            if edge_index.ndim == 2 and edge_index.size(0) != 2 and edge_index.size(1) == 2:
                edge_index = edge_index.t().contiguous()
            g = self.pocket_enc(node_scalar, edge_index)  # (d,)
            g_list.append(g)
        return torch.stack(g_list, dim=0)  # (B, d)

    def forward(
        self,
        v_prot: torch.Tensor, m_prot: torch.Tensor,
        v_mol: torch.Tensor,  m_mol: torch.Tensor,
        v_chem: torch.Tensor,
        pocket_batch: Optional[List[dict]] = None,
        topk_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        v_prot: (B, Mp, d_protein), m_prot: (B, Mp) bool
        v_mol:  (B, Md, d_molclr),  m_mol:  (B, Md) bool
        v_chem: (B, d_chem)
        pocket_batch: list[dict] len=B or None
        """
        device = v_prot.device
        B = v_prot.size(0)

        # Project base sequences
        P = self.proj_p(v_prot)  # (B, Mp, d)
        D = self.proj_d(v_mol)   # (B, Md, d)
        Pm = m_prot
        Dm = m_mol

        # [CHEM] token on drug side (always)
        chem_tok = self.proj_c(v_chem).unsqueeze(1)  # (B,1,d)
        D = torch.cat([chem_tok, D], dim=1)          # (B, Md+1, d)
        chem_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
        Dm = torch.cat([chem_mask, Dm], dim=1)       # (B, Md+1)

        # [GRAPH] token on protein side (optional)
        graph_tok = self._build_graph_token(pocket_batch, device=device)
        if graph_tok is not None:
            graph_tok = graph_tok.unsqueeze(1)       # (B,1,d)
            P = torch.cat([graph_tok, P], dim=1)     # (B, Mp+1, d)
            graph_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
            Pm = torch.cat([graph_mask, Pm], dim=1)  # (B, Mp+1)

        # Optional: token pruning (topk_ratio) — 这里先不做剪枝，保留接口
        _ = topk_ratio

        # UGCA layers
        ent_acc = 0.0
        for blk in self.blocks:
            P, D, ent = blk(P, Pm, D, Dm)
            ent_acc = ent_acc + ent

        ent_acc = ent_acc / max(1, len(self.blocks))

        # Pool to fixed vectors
        hP = masked_mean(P, Pm)  # (B,d)
        hD = masked_mean(D, Dm)  # (B,d)

        y = self.head(torch.cat([hP, hD], dim=-1)).squeeze(-1)  # (B,)

        # Optional entropy regularization hook:
        # 你可以在训练脚本里读取 model.last_entropy 做额外 loss
        self.last_entropy = ent_acc

        return y


# ---------------------------
# Vector model (compatible with non-sequence mode)
# ---------------------------
class UGCAVecModel(nn.Module):
    def __init__(self, cfg: VecCfg):
        super().__init__()
        d = cfg.d_model
        self.proj_p = nn.Linear(cfg.d_protein, d, bias=False)
        self.proj_d = nn.Linear(cfg.d_molclr,  d, bias=False)
        self.proj_c = nn.Linear(cfg.d_chem,    d, bias=False)

        self.head = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, 1),
        )

    def forward(self, v_prot: torch.Tensor, v_mol: torch.Tensor, v_chem: torch.Tensor) -> torch.Tensor:
        """
        v_prot: (B, d_protein)
        v_mol:  (B, d_molclr)
        v_chem: (B, d_chem)
        """
        p = self.proj_p(v_prot)
        d = self.proj_d(v_mol)
        c = self.proj_c(v_chem)
        y = self.head(torch.cat([p, d, c], dim=-1)).squeeze(-1)
        return y


# ---------------------------
# Factory
# ---------------------------
def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """
    cfg 来自 train.py 中 build_model_from_src 的 dict
    必须包含：
      - sequence: bool
      - d_protein, d_molclr, d_chem
    以及可选：
      - d_model, n_heads, n_layers, dropout
      - use_pocket, pocket_in_dim, pocket_hidden
      - entropy_reg
    """
    sequence = bool(cfg.get("sequence", False))

    if sequence:
        mcfg = SeqCfg(
            d_protein=int(cfg.get("d_protein", 1280)),
            d_molclr=int(cfg.get("d_molclr", 300)),
            d_chem=int(cfg.get("d_chem", 384)),
            d_model=int(cfg.get("d_model", 512)),
            n_heads=int(cfg.get("n_heads", 8)),
            n_layers=int(cfg.get("n_layers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            use_pocket=bool(cfg.get("use_pocket", False)),
            pocket_in_dim=int(cfg.get("pocket_in_dim", 21)),
            pocket_hidden=int(cfg.get("pocket_hidden", 128)),
            entropy_reg=float(cfg.get("entropy_reg", 0.0)),
        )
        return UGCASeqModel(mcfg)

    vcfg = VecCfg(
        d_protein=int(cfg.get("d_protein", 1280)),
        d_molclr=int(cfg.get("d_molclr", 300)),
        d_chem=int(cfg.get("d_chem", 384)),
        d_model=int(cfg.get("d_model", 512)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    return UGCAVecModel(vcfg)
