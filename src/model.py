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
        nig_params = self.nig_predictor(x)
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
    def __init__(self, d_model, nhead, gate_mode='mu_times_evi', lamb=1.0, temp=1.0, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.temp = temp
        self.gate_d = UncertaintyGate(d_model, gate_mode, lamb)
        self.gate_p = UncertaintyGate(d_model, gate_mode, lamb)

        self.w_q_d = nn.Linear(d_model, d_model)
        self.w_k_p = nn.Linear(d_model, d_model)
        self.w_v_p = nn.Linear(d_model, d_model)

        self.w_q_p = nn.Linear(d_model, d_model)
        self.w_k_d = nn.Linear(d_model, d_model)
        self.w_v_d = nn.Linear(d_model, d_model)

        self.out_proj_d = nn.Linear(d_model, d_model)
        self.out_proj_p = nn.Linear(d_model, d_model)

        self.ffn_d = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(d_model * 4, d_model))
        self.ffn_p = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(d_model * 4, d_model))

        self.norm_d1 = nn.LayerNorm(d_model)
        self.norm_d2 = nn.LayerNorm(d_model)
        self.norm_p1 = nn.LayerNorm(d_model)
        self.norm_p2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def cross_attn(self, q, k, v, gate, mask=None):
        B, Lq, D = q.shape
        Bk, Lk, Dk = k.shape
        head_dim = D // self.nhead

        q = q.view(B, Lq, self.nhead, head_dim).transpose(1, 2)
        k = k.view(B, Lk, self.nhead, head_dim).transpose(1, 2)
        v = v.view(B, Lk, self.nhead, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        log_gate = torch.log(gate + 1e-9).view(B, 1, 1, Lk)
        scores = scores + log_gate

        if mask is not None:
            mask = mask.view(B, 1, 1, Lk)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores / self.temp, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Lq, D)
        return out

    def forward(self, h_d, h_p, mask_d=None, mask_p=None):
        g_d = self.gate_d(h_d)
        g_p = self.gate_p(h_p)

        # D <- P
        q_d = self.w_q_d(h_d)
        k_p = self.w_k_p(h_p)
        v_p = self.w_v_p(h_p)
        h_d = self.norm_d1(h_d + self.dropout(self.out_proj_d(self.cross_attn(q_d, k_p, v_p, g_p, mask_p))))

        # P <- D
        q_p = self.w_q_p(h_p)
        k_d = self.w_k_d(h_d)
        v_d = self.w_v_d(h_d)
        h_p = self.norm_p1(h_p + self.dropout(self.out_proj_p(self.cross_attn(q_p, k_d, v_d, g_d, mask_d))))

        h_d = self.norm_d2(h_d + self.ffn_d(h_d))
        h_p = self.norm_p2(h_p + self.ffn_p(h_p))
        return h_d, h_p


class SequencePooling(nn.Module):
    def __init__(self, d_model, mode='meanmax'):
        super().__init__()
        self.mode = mode
        self.d_model = d_model

        if mode == 'meanmax':
            self.proj = nn.Linear(d_model * 2, d_model)

        elif mode == 'attn':
            # 学习一个 Query 向量
            self.attn_query = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1)
            )

        elif mode == 'mh-attn':
            self.num_heads = 4
            self.query_emb = nn.Parameter(torch.randn(1, self.num_heads, self.d_model))
            self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=self.num_heads, batch_first=True)
            self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: [B, L, D]
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
            scores = self.attn_query(x)
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            weights = F.softmax(scores, dim=1)
            return torch.sum(weights * x, dim=1)

        elif self.mode == 'mh-attn':
            B = x.shape[0]
            q = self.query_emb[:, 0:1, :].expand(B, 1, -1)
            key_mask = (mask == 0) if mask is not None else None
            attn_out, _ = self.attn(query=q, key=x, value=x, key_padding_mask=key_mask)
            return self.proj(attn_out.squeeze(1))

        return torch.mean(x, dim=1)


class MatchMLP(nn.Module):
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
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.proj_drug_seq = nn.Linear(300, args.d_model)
        self.proj_prot_seq = nn.Linear(1280, args.d_model)
        self.proj_chem = nn.Linear(384, args.d_model)
        self.proj_graph = nn.Linear(256, args.d_model)

        self.layers = nn.ModuleList([
            UGCALayer(args.d_model, args.nhead, args.gate_mode, args.lamb, args.temp, args.dropout)
            for _ in range(args.nlayers)
        ])

        self.pool = SequencePooling(args.d_model, mode=args.pooling)

        self.fusion_mlp_d = nn.Sequential(nn.Linear(2 * args.d_model, args.d_fuse), nn.GELU(), nn.Dropout(args.dropout),
                                          nn.Linear(args.d_fuse, args.d_fuse))
        self.fusion_mlp_p = nn.Sequential(nn.Linear(2 * args.d_model, args.d_fuse), nn.GELU(), nn.Dropout(args.dropout),
                                          nn.Linear(args.d_fuse, args.d_fuse))

        if args.fusion_head == 'match-mlp':
            self.head = MatchMLP(args.d_fuse, dropout=args.dropout)
        else:
            self.head = nn.Sequential(nn.Linear(2 * args.d_fuse, 512), nn.GELU(), nn.Dropout(args.dropout),
                                      nn.Linear(512, 1))

    def forward(self, batch):
        molclr = batch['molclr']
        esm2 = batch['esm2']
        chemberta = batch['chemberta']
        gvp = batch['gvp']
        mask_d = batch['mask_d']
        mask_p = batch['mask_p']

        H_d = self.proj_drug_seq(molclr)
        H_p = self.proj_prot_seq(esm2)
        c_prime = self.proj_chem(chemberta)
        g_prime = self.proj_graph(gvp)

        for layer in self.layers:
            H_d, H_p = layer(H_d, H_p, mask_d, mask_p)

        z_d_seq = self.pool(H_d, mask_d)
        z_p_seq = self.pool(H_p, mask_p)

        h_d_all = torch.cat([z_d_seq, c_prime], dim=-1)
        h_p_all = torch.cat([z_p_seq, g_prime], dim=-1)

        f_d = self.fusion_mlp_d(h_d_all)
        f_p = self.fusion_mlp_p(h_p_all)

        return self.head(f_d, f_p)