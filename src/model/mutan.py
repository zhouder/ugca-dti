import torch
import torch.nn as nn

class MUTAN(nn.Module):
    def __init__(self, d_in1: int, d_in2: int, d_out: int = 256, rank: int = 20, dropout: float = 0.1):
        super().__init__()
        self.U = nn.Linear(d_in1, rank, bias=False)
        self.V = nn.Linear(d_in2, rank, bias=False)
        self.core = nn.Parameter(torch.randn(rank, rank, d_out) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.out = nn.LayerNorm(d_out)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xr = self.U(x)
        yr = self.V(y)
        z = torch.einsum('bi,ijc,bj->bc', xr, self.core, yr)
        z = self.drop(z)
        return self.out(z)
