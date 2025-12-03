import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.fc1(x)
        x2 = F.gelu(x2)
        x2 = self.dropout(x2)
        x2 = self.fc2(x2)
        x2 = self.dropout(x2)
        return x2


class EvidentialGate(nn.Module):
    """
    简化版 Evidential NIG 门控，用于为每个 token 产生 gate \in (0,1]，只用于注意力中。
    """
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        gate_mode: str = "mu_times_evi",
        gate_lambda: float = 1.0,
        gate_min: float = 1e-3,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 4)  # mu, log_nu, log_alpha, log_beta

        self.gate_mode = gate_mode
        self.gate_lambda = gate_lambda
        self.gate_min = gate_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d)
        return: gate (B, L) in (gate_min, 1]
        """
        h = F.gelu(self.fc1(x))
        out = self.fc2(h)

        mu = out[..., 0:1]
        log_nu = out[..., 1:2]
        log_alpha = out[..., 2:3]
        log_beta = out[..., 3:4]

        nu = F.softplus(log_nu) + 1e-6
        alpha = F.softplus(log_alpha) + 1.0 + 1e-6  # > 1
        beta = F.softplus(log_beta) + 1e-6

        sigma2 = beta / (nu * (alpha - 1.0) + 1e-6)  # 认知不确定性

        if self.gate_mode == "none":
            g = x.new_ones(x.size(0), x.size(1))
        elif self.gate_mode == "evi_only":
            g = torch.exp(-self.gate_lambda * sigma2).squeeze(-1)
        elif self.gate_mode == "mu_only":
            g = torch.sigmoid(mu).squeeze(-1)
        elif self.gate_mode == "mu_times_evi":
            g_mu = torch.sigmoid(mu)
            g_evi = torch.exp(-self.gate_lambda * sigma2)
            g = (g_mu * g_evi).squeeze(-1)
        else:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

        g = torch.clamp(g, min=self.gate_min, max=1.0)
        return g


class MultiHeadGatedCrossAttention(nn.Module):
    """
    带 gate 的多头 cross-attention 模块。
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dropout: float = 0.1,
        attn_temp: float = 1.0,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.attn_temp = attn_temp

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_gate: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        q:  (B, Lq, d)
        kv: (B, Lk, d)
        key_gate: (B, Lk) in (0,1]
        key_mask: (B, Lk) 1=valid, 0=pad
        return: (B, Lq, d)
        """
        B, Lq, _ = q.size()
        _, Lk, _ = kv.size()

        q_proj = self.q_proj(q).view(B, Lq, self.nhead, self.d_head).transpose(1, 2)  # (B, h, Lq, d_h)
        k_proj = self.k_proj(kv).view(B, Lk, self.nhead, self.d_head).transpose(1, 2)  # (B, h, Lk, d_h)
        v_proj = self.v_proj(kv).view(B, Lk, self.nhead, self.d_head).transpose(1, 2)  # (B, h, Lk, d_h)

        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1))  # (B, h, Lq, Lk)
        scores = scores / (math.sqrt(self.d_head) * self.attn_temp)

        if key_gate is not None:
            # key_gate: (B, Lk) -> (B, 1, 1, Lk)
            log_gate = torch.log(key_gate.clamp_min(1e-6))
            scores = scores + log_gate.unsqueeze(1).unsqueeze(2)

        if key_mask is not None:
            # key_mask: 1 valid, 0 pad
            mask = (key_mask == 0).unsqueeze(1).unsqueeze(2)  # (B,1,1,Lk)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v_proj)  # (B, h, Lq, d_h)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.out_proj(out)
        return out


class UGCALayer(nn.Module):
    """
    一层双向 UGCA：D←P 和 P←D。
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        gate_mode: str = "mu_times_evi",
        gate_lambda: float = 1.0,
        gate_min: float = 1e-3,
        attn_temp: float = 1.0,
    ):
        super().__init__()
        self.gate = EvidentialGate(
            d_model=d_model,
            hidden_dim=128,
            gate_mode=gate_mode,
            gate_lambda=gate_lambda,
            gate_min=gate_min,
        )

        self.attn_d_from_p = MultiHeadGatedCrossAttention(
            d_model=d_model, nhead=nhead, dropout=dropout, attn_temp=attn_temp
        )
        self.attn_p_from_d = MultiHeadGatedCrossAttention(
            d_model=d_model, nhead=nhead, dropout=dropout, attn_temp=attn_temp
        )

        self.ln_d1 = nn.LayerNorm(d_model)
        self.ln_p1 = nn.LayerNorm(d_model)
        self.ffn_d = PositionwiseFFN(d_model, dim_feedforward, dropout)
        self.ffn_p = PositionwiseFFN(d_model, dim_feedforward, dropout)
        self.ln_d2 = nn.LayerNorm(d_model)
        self.ln_p2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H_d: torch.Tensor,
        H_p: torch.Tensor,
        mask_d: Optional[torch.Tensor],
        mask_p: Optional[torch.Tensor],
    ):
        """
        H_d: (B, Ld, d)
        H_p: (B, Lp, d)
        mask_d: (B, Ld), mask_p: (B, Lp)
        """
        g_d = self.gate(H_d)  # (B, Ld)
        g_p = self.gate(H_p)  # (B, Lp)

        # D <- P
        H_d_delta = self.attn_d_from_p(H_d, H_p, key_gate=g_p, key_mask=mask_p)
        H_d = self.ln_d1(H_d + self.dropout(H_d_delta))
        H_d_ffn = self.ffn_d(H_d)
        H_d = self.ln_d2(H_d + self.dropout(H_d_ffn))

        # P <- D
        H_p_delta = self.attn_p_from_d(H_p, H_d, key_gate=g_d, key_mask=mask_d)
        H_p = self.ln_p1(H_p + self.dropout(H_p_delta))
        H_p_ffn = self.ffn_p(H_p)
        H_p = self.ln_p2(H_p + self.dropout(H_p_ffn))

        return H_d, H_p


class UGCAEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        gate_mode: str = "mu_times_evi",
        gate_lambda: float = 1.0,
        gate_min: float = 1e-3,
        attn_temp: float = 1.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                UGCALayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    gate_mode=gate_mode,
                    gate_lambda=gate_lambda,
                    gate_min=gate_min,
                    attn_temp=attn_temp,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        H_d: torch.Tensor,
        H_p: torch.Tensor,
        mask_d: Optional[torch.Tensor],
        mask_p: Optional[torch.Tensor],
    ):
        for layer in self.layers:
            H_d, H_p = layer(H_d, H_p, mask_d, mask_p)
        return H_d, H_p


class MeanMaxPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, L, d), mask: (B, L) 1 valid / 0 pad
        """
        if mask is None:
            mean = x.mean(dim=1)
            max_val, _ = x.max(dim=1)
        else:
            mask = mask.unsqueeze(-1).float()  # (B, L, 1)
            length = mask.sum(dim=1).clamp(min=1.0)
            mean = (x * mask).sum(dim=1) / length
            # 对 pad 位置设置极小值以做 max
            x_masked = x.masked_fill(mask == 0, float("-inf"))
            max_val, _ = x_masked.max(dim=1)
            max_val[torch.isinf(max_val)] = 0.0

        out = torch.cat([mean, max_val], dim=-1)
        out = self.proj(out)
        return out


class AttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, L, d), mask: (B, L)
        """
        # 简化实现：线性打分
        scores = self.linear(x).squeeze(-1)  # (B, L)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # (B, L)
        attn = attn.unsqueeze(-1)
        out = (x * attn).sum(dim=1)
        return out


class MultiHeadAttentionPool(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.queries = nn.Parameter(torch.randn(num_heads, d_model))
        self.linear = nn.Linear(d_model, num_heads)
        self.proj = nn.Linear(num_heads * d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, L, d), mask: (B, L)
        """
        scores = self.linear(x)  # (B, L, H)
        if mask is not None:
            mask_exp = mask.unsqueeze(-1)  # (B, L, 1)
            scores = scores.masked_fill(mask_exp == 0, float("-inf"))

        attn = torch.softmax(scores, dim=1)  # (B, L, H)
        attn = attn.permute(0, 2, 1)  # (B, H, L)
        x_exp = x.unsqueeze(1)  # (B, 1, L, d)
        attn_exp = attn.unsqueeze(-1)  # (B, H, L, 1)
        out = (x_exp * attn_exp).sum(dim=2)  # (B, H, d)
        out = out.reshape(x.size(0), -1)  # (B, H*d)
        out = self.proj(out)  # (B, d)
        return out


class BranchMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(out_dim, in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MatchMLPHead(nn.Module):
    """
    默认融合头： [f_d, f_p, f_d ⊙ f_p, |f_d - f_p|] -> MLP -> logit
    """
    def __init__(self, d_fuse: int, hidden_factor: float = 2.0, dropout: float = 0.1):
        super().__init__()
        in_dim = 4 * d_fuse
        hidden = int(d_fuse * hidden_factor)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, f_d: torch.Tensor, f_p: torch.Tensor) -> torch.Tensor:
        prod = f_d * f_p
        diff = torch.abs(f_d - f_p)
        u = torch.cat([f_d, f_p, prod, diff], dim=-1)
        logit = self.net(u)
        return logit


class ConcatMLPHead(nn.Module):
    def __init__(self, d_fuse: int, hidden_factor: float = 2.0, dropout: float = 0.1):
        super().__init__()
        in_dim = 2 * d_fuse
        hidden = int(d_fuse * hidden_factor)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, f_d: torch.Tensor, f_p: torch.Tensor) -> torch.Tensor:
        u = torch.cat([f_d, f_p], dim=-1)
        logit = self.net(u)
        return logit


class MutanHead(nn.Module):
    """
    简化版 MUTAN 融合头。
    """
    def __init__(
        self,
        d_fuse: int,
        out_dim: int = 512,
        rank: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.out_dim = out_dim
        self.d_fuse = d_fuse

        self.proj_d = nn.Linear(d_fuse, rank * out_dim)
        self.proj_p = nn.Linear(d_fuse, rank * out_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(out_dim, 1)

    def forward(self, f_d: torch.Tensor, f_p: torch.Tensor) -> torch.Tensor:
        B = f_d.size(0)
        d_proj = self.proj_d(f_d).view(B, self.rank, self.out_dim)
        p_proj = self.proj_p(f_p).view(B, self.rank, self.out_dim)
        z = (d_proj * p_proj).sum(dim=1)  # (B, out_dim)
        z = F.gelu(z)
        z = self.dropout(z)
        logit = self.out(z)
        return logit


class UGCADTIModel(nn.Module):
    """
    UGCA-DTI v1.0 主模型：序列 UGCA + 全局晚融合 + Match-MLP / MUTAN。
    """
    def __init__(
        self,
        d_mol: int = 300,
        d_prot: int = 1280,
        d_chem: int = 384,
        d_graph: int = 256,
        d_model: int = 256,
        d_fuse: int = 512,
        nlayers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pool_type: str = "meanmax",  # meanmax / attn / mh-attn
        fusion_head: str = "match-mlp",  # match-mlp / mutan / concat-mlp
        gate_mode: str = "mu_times_evi",
        gate_lambda: float = 1.0,
        gate_min: float = 1e-3,
        attn_temp: float = 1.0,
        pool_heads: int = 4,
        mutan_rank: int = 8,
    ):
        super().__init__()

        # 序列特征投影
        self.drug_seq_proj = nn.Linear(d_mol, d_model)
        self.prot_seq_proj = nn.Linear(d_prot, d_model)

        # 全局特征投影
        self.chem_proj = nn.Linear(d_chem, d_model)
        self.graph_proj = nn.Linear(d_graph, d_model)

        # UGCA
        self.ugca = UGCAEncoder(
            num_layers=nlayers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            gate_mode=gate_mode,
            gate_lambda=gate_lambda,
            gate_min=gate_min,
            attn_temp=attn_temp,
        )

        # 池化
        pool_type = pool_type.lower()
        if pool_type == "meanmax":
            self.drug_pool = MeanMaxPool(d_model)
            self.prot_pool = MeanMaxPool(d_model)
        elif pool_type == "attn":
            self.drug_pool = AttentionPool(d_model)
            self.prot_pool = AttentionPool(d_model)
        elif pool_type == "mh-attn":
            self.drug_pool = MultiHeadAttentionPool(d_model, num_heads=pool_heads)
            self.prot_pool = MultiHeadAttentionPool(d_model, num_heads=pool_heads)
        else:
            raise ValueError(f"Unknown pooling type: {pool_type}")
        self.pool_type = pool_type

        # 分支 MLP
        self.drug_branch = BranchMLP(d_model * 2, d_fuse, dropout=dropout)
        self.prot_branch = BranchMLP(d_model * 2, d_fuse, dropout=dropout)

        # 融合头
        fusion_head = fusion_head.lower()
        self.fusion_head_type = fusion_head
        if fusion_head == "match-mlp":
            self.head = MatchMLPHead(d_fuse=d_fuse, dropout=dropout)
        elif fusion_head == "concat-mlp":
            self.head = ConcatMLPHead(d_fuse=d_fuse, dropout=dropout)
        elif fusion_head == "mutan":
            self.head = MutanHead(d_fuse=d_fuse, out_dim=d_fuse, rank=mutan_rank, dropout=dropout)
        else:
            raise ValueError(f"Unknown fusion head: {fusion_head}")

    def forward(
        self,
        drug_seq: torch.Tensor,
        prot_seq: torch.Tensor,
        chem: torch.Tensor,
        graph: torch.Tensor,
        drug_mask: Optional[torch.Tensor] = None,
        prot_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        drug_seq: (B, Ld, d_mol)
        prot_seq: (B, Lp, d_prot)
        chem:     (B, d_chem)
        graph:    (B, d_graph)
        drug_mask:(B, Ld) 1 valid
        prot_mask:(B, Lp) 1 valid
        return: logits (B,)
        """
        # 投影到统一维度
        drug_seq = self.drug_seq_proj(drug_seq)
        prot_seq = self.prot_seq_proj(prot_seq)

        # 序列级 UGCA
        drug_seq, prot_seq = self.ugca(drug_seq, prot_seq, drug_mask, prot_mask)

        # 池化
        z_d = self.drug_pool(drug_seq, drug_mask)
        z_p = self.prot_pool(prot_seq, prot_mask)

        # 全局晚融合
        c = self.chem_proj(chem)
        g = self.graph_proj(graph)
        h_d = torch.cat([z_d, c], dim=-1)
        h_p = torch.cat([z_p, g], dim=-1)

        f_d = self.drug_branch(h_d)
        f_p = self.prot_branch(h_p)

        logit = self.head(f_d, f_p).squeeze(-1)
        return logit
