import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UncertaintyGate(nn.Module):
    def __init__(self, d_model, gate_mode='mu_times_evi', lamb=1.0, g_min=1e-3):
        super().__init__()
        self.d_model = d_model
        self.gate_mode = gate_mode
        self.lamb = lamb
        self.g_min = g_min
        # Output: [mu, v, alpha, beta]
        self.nig_predictor = nn.Linear(d_model, 4)

    def forward(self, x):
        """
        x: [B, L, D]
        return g: [B, L] in (0, 1]
        """
        nig_params = self.nig_predictor(x)  # [B, L, 4]
        mu = nig_params[..., 0]
        v = F.softplus(nig_params[..., 1]) + 1e-6
        alpha = F.softplus(nig_params[..., 2]) + 1.0 + 1e-6
        beta = F.softplus(nig_params[..., 3]) + 1e-6

        sigma_e2 = beta / (v * (alpha - 1))
        mu_sigmoid = torch.sigmoid(mu)

        if self.gate_mode == 'evi_only':
            g = torch.exp(-self.lamb * sigma_e2)
        elif self.gate_mode == 'mu_only':
            g = mu_sigmoid
        else:  # mu_times_evi
            g = mu_sigmoid * torch.exp(-self.lamb * sigma_e2)

        g = torch.clamp(g, min=self.g_min, max=1.0)
        return g


class UGCALayer(nn.Module):
    """
    双向不确定性感知 cross-attention 层
    """
    def __init__(self, d_model, nhead, gate_mode='mu_times_evi',
                 lamb=1.0, temp=1.0, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.temp = temp

        self.gate_d = UncertaintyGate(d_model, gate_mode, lamb)
        self.gate_p = UncertaintyGate(d_model, gate_mode, lamb)

        # D <- P
        self.w_q_d = nn.Linear(d_model, d_model)
        self.w_k_p = nn.Linear(d_model, d_model)
        self.w_v_p = nn.Linear(d_model, d_model)

        # P <- D
        self.w_q_p = nn.Linear(d_model, d_model)
        self.w_k_d = nn.Linear(d_model, d_model)
        self.w_v_d = nn.Linear(d_model, d_model)

        self.out_proj_d = nn.Linear(d_model, d_model)
        self.out_proj_p = nn.Linear(d_model, d_model)

        self.ffn_d = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_p = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm_d1 = nn.LayerNorm(d_model)
        self.norm_d2 = nn.LayerNorm(d_model)
        self.norm_p1 = nn.LayerNorm(d_model)
        self.norm_p2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def cross_attn(self, q, k, v, gate, mask=None):
        """
        q: [B, Lq, D]
        k,v: [B, Lk, D]
        gate: [B, Lk]
        mask: [B, Lk] (1 = valid, 0 = pad)
        """
        B, Lq, D = q.shape
        Bk, Lk, Dk = k.shape
        head_dim = D // self.nhead

        q = q.view(B, Lq, self.nhead, head_dim).transpose(1, 2)  # [B, H, Lq, Dh]
        k = k.view(B, Lk, self.nhead, head_dim).transpose(1, 2)  # [B, H, Lk, Dh]
        v = v.view(B, Lk, self.nhead, head_dim).transpose(1, 2)  # [B, H, Lk, Dh]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # [B, H, Lq, Lk]

        # gate 融入 attention logits
        log_gate = torch.log(gate + 1e-9).view(B, 1, 1, Lk)
        scores = scores + log_gate

        if mask is not None:
            mask = mask.view(B, 1, 1, Lk)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores / self.temp, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, Lq, Dh]
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        return out

    def forward(self, h_d, h_p, mask_d=None, mask_p=None):
        """
        h_d: [B, Ld, D], h_p: [B, Lp, D]
        """
        g_d = self.gate_d(h_d)  # [B, Ld]
        g_p = self.gate_p(h_p)  # [B, Lp]

        # D <- P
        q_d = self.w_q_d(h_d)
        k_p = self.w_k_p(h_p)
        v_p = self.w_v_p(h_p)
        h_d = self.norm_d1(
            h_d + self.dropout(
                self.out_proj_d(
                    self.cross_attn(q_d, k_p, v_p, g_p, mask_p)
                )
            )
        )

        # P <- D
        q_p = self.w_q_p(h_p)
        k_d = self.w_k_d(h_d)
        v_d = self.w_v_d(h_d)
        h_p = self.norm_p1(
            h_p + self.dropout(
                self.out_proj_p(
                    self.cross_attn(q_p, k_d, v_d, g_d, mask_d)
                )
            )
        )

        # FFN
        h_d = self.norm_d2(h_d + self.ffn_d(h_d))
        h_p = self.norm_p2(h_p + self.ffn_p(h_p))
        return h_d, h_p


class SequencePooling(nn.Module):
    """
    支持 mean+max / 单头 attn / 多头 attn 的序列池化
    """
    def __init__(self, d_model, mode='meanmax'):
        super().__init__()
        self.mode = mode
        self.d_model = d_model

        if mode == 'meanmax':
            self.proj = nn.Linear(d_model * 2, d_model)

        elif mode == 'attn':
            self.attn_query = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1)
            )

        elif mode == 'mh-attn':
            self.num_heads = 4
            self.query_emb = nn.Parameter(torch.randn(1, self.num_heads, self.d_model))
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=self.num_heads, batch_first=True
            )
            self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        x: [B, L, D]
        mask: [B, L] (1 = valid, 0 = pad)
        """
        if self.mode == 'meanmax':
            if mask is not None:
                mask_float = mask.unsqueeze(-1).float()
                x_masked = x * mask_float
                sum_x = torch.sum(x_masked, dim=1)
                lens = torch.sum(mask_float, dim=1).clamp(min=1e-9)
                mean_x = sum_x / lens

                x_for_max = x.clone()
                x_for_max[mask == 0] = -1e9
                max_x = torch.max(x_for_max, dim=1)[0]
            else:
                mean_x = torch.mean(x, dim=1)
                max_x = torch.max(x, dim=1)[0]
            return self.proj(torch.cat([mean_x, max_x], dim=-1))

        elif self.mode == 'attn':
            scores = self.attn_query(x)  # [B, L, 1]
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            weights = F.softmax(scores, dim=1)
            return torch.sum(weights * x, dim=1)  # [B, D]

        elif self.mode == 'mh-attn':
            B = x.shape[0]
            q = self.query_emb[:, 0:1, :].expand(B, 1, -1)  # [B, 1, D]
            key_mask = (mask == 0) if mask is not None else None
            attn_out, _ = self.attn(
                query=q, key=x, value=x, key_padding_mask=key_mask
            )  # [B, 1, D]
            return self.proj(attn_out.squeeze(1))

        # fallback: simple mean
        return torch.mean(x, dim=1)


class MatchMLP(nn.Module):
    """
    f([fd, fp, fd * fp, |fd - fp|]) → logit
    """
    def __init__(self, d_in, hidden_dims=[1024, 512, 1], dropout=0.1):
        super().__init__()
        input_dim = 4 * d_in
        layers = []
        curr_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(curr_dim, h_dim))
            if i < len(hidden_dims) - 1:
                layers.append(nn.LayerNorm(h_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, fd, fp):
        diff = torch.abs(fd - fp)
        prod = fd * fp
        cat = torch.cat([fd, fp, prod, diff], dim=-1)
        return self.mlp(cat)


class UGCA_DTI(nn.Module):
    """
    只用 MolCLR(药物) + ESM2(蛋白) 的轻量版 UGCA-DTI
    - 药物序列: MolCLR (L_d, 300)
    - 蛋白序列: ESM2   (L_p, 1280)
    - 不再使用 chemberta / GVP
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        # 投影到统一 d_model 维度
        self.proj_drug_seq = nn.Linear(300, args.d_model)
        self.proj_prot_seq = nn.Linear(1280, args.d_model)

        # 多层 UGCA
        self.layers = nn.ModuleList([
            UGCALayer(
                args.d_model, args.nhead,
                gate_mode=args.gate_mode,
                lamb=args.lamb,
                temp=args.temp,
                dropout=args.dropout
            )
            for _ in range(args.nlayers)
        ])

        # 序列池化（分别对药物 / 蛋白）
        self.pool = SequencePooling(args.d_model, mode=args.pooling)

        # 对 pooled 表示做一层侧内融合 MLP
        self.fusion_mlp_d = nn.Sequential(
            nn.Linear(args.d_model, args.d_fuse),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_fuse, args.d_fuse),
        )
        self.fusion_mlp_p = nn.Sequential(
            nn.Linear(args.d_model, args.d_fuse),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_fuse, args.d_fuse),
        )

        # 交互 head
        if args.fusion_head == 'match-mlp':
            self.head = MatchMLP(args.d_fuse, dropout=args.dropout)
        else:
            self.head = nn.Sequential(
                nn.Linear(2 * args.d_fuse, 512),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(512, 1),
            )

    def forward(self, batch):
        """
        batch:
          molclr: [B, Ld, 300]
          esm2:   [B, Lp, 1280]
          mask_d: [B, Ld]
          mask_p: [B, Lp]
        """
        molclr = batch['molclr']
        esm2 = batch['esm2']
        mask_d = batch['mask_d']
        mask_p = batch['mask_p']

        # 投影
        H_d = self.proj_drug_seq(molclr)
        H_p = self.proj_prot_seq(esm2)

        # 多层 UGCA 交互
        for layer in self.layers:
            H_d, H_p = layer(H_d, H_p, mask_d, mask_p)

        # 池化得到全局表征
        z_d_seq = self.pool(H_d, mask_d)  # [B, D]
        z_p_seq = self.pool(H_p, mask_p)  # [B, D]

        # 各侧内融合
        f_d = self.fusion_mlp_d(z_d_seq)  # [B, d_fuse]
        f_p = self.fusion_mlp_p(z_p_seq)  # [B, d_fuse]

        # 匹配 head
        if isinstance(self.head, MatchMLP):
            logits = self.head(f_d, f_p)  # [B, 1]
        else:
            logits = self.head(torch.cat([f_d, f_p], dim=-1))  # [B, 1]

        return logits
