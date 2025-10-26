import torch
import torch.nn as nn

class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim))
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum('bnd,d->bn', x, self.q)
        logits = logits.masked_fill(~mask, float('-inf'))
        w = torch.softmax(logits, dim=-1)
        return torch.einsum('bn,bnd->bd', w, x)

class TokenPooler(nn.Module):
    def __init__(self, dim: int, mode: str = 'attn'):
        super().__init__()
        self.mode = mode
        if mode == 'attn':
            self.attn = AttentionPool(dim)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.mode == 'mean':
            m = mask.float().unsqueeze(-1)
            return (x*m).sum(1) / (m.sum(1) + 1e-6)
        elif self.mode == 'max':
            x_ = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            return x_.max(1).values
        else:
            return self.attn(x, mask)
