# src/model/simple_dti.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanPool(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D], mask: [B, L] -> [B, D]
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

class SimpleDTI(nn.Module):
    """
    三路输入：
      H_D: [B, n_atoms, 300]
      H_P: [B, seq_len, 1280]
      h_C: [B, 384]
    """
    def __init__(self, d: int = 256, dropout: float = 0.2):
        super().__init__()
        self.pool = MeanPool()
        self.proj_d = nn.Linear(300, d)
        self.proj_p = nn.Linear(1280, d)
        self.proj_c = nn.Linear(384, d)
        fusion_in = d * 6   # [d, d, d, d⊙d, d⊙d, d⊙d]
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_in, d),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d, 1)
        )

    def forward(self, H_D, mask_D, H_P, mask_P, h_C):
        d_emb = self.pool(H_D, mask_D)               # [B,300]->[B,d]
        p_emb = self.pool(H_P, mask_P)               # [B,1280]->[B,d]
        d_emb = self.proj_d(d_emb)
        p_emb = self.proj_p(p_emb)
        c_emb = self.proj_c(h_C)                     # [B,384]->[B,d]

        z = torch.cat(
            [d_emb, p_emb, c_emb,
             d_emb * p_emb, d_emb * c_emb, p_emb * c_emb],
            dim=-1
        )                                            # [B, 6d]
        logit = self.mlp(z).squeeze(-1)              # [B]
        return logit
