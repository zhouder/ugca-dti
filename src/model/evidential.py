from __future__ import annotations
import torch
import torch.nn as nn

class NIGHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 4)
        )
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        raw = self.net(x)
        gamma = raw[..., 0]
        nu    = self.softplus(raw[..., 1]) + 1e-6
        alpha = self.softplus(raw[..., 2]) + 1.0 + 1e-6
        beta  = self.softplus(raw[..., 3]) + 1e-6
        sigma2 = beta / (nu * (alpha - 1.0))
        return gamma, nu, alpha, beta, sigma2

class GateFromUncertainty(nn.Module):
    def __init__(self, k: float = 2.0, g_min: float = 0.05):
        super().__init__()
        self.k = k
        self.g_min = g_min
    def forward(self, sigma2: torch.Tensor, mask: torch.Tensor):
        g = torch.exp(- self.k * sigma2)
        g = torch.clamp(g, min=self.g_min, max=1.0)
        return g * mask.float()

class BudgetRegularizer(nn.Module):
    def __init__(self, rho: float = 0.6, lam: float = 1e-2):
        super().__init__()
        self.rho = rho
        self.lam = lam
    def forward(self, g: torch.Tensor, mask: torch.Tensor):
        denom = mask.float().sum() + 1e-6
        mean_g = (g * mask.float()).sum() / denom
        return self.lam * (mean_g - self.rho) ** 2

class EDLRegularizer(nn.Module):
    def __init__(self, lam: float = 1e-3):
        super().__init__()
        self.lam = lam
    def forward(self, alpha: torch.Tensor, beta: torch.Tensor):
        evid = torch.relu(alpha - 1.0)
        return self.lam * (evid.mean() + (beta.mean()))
