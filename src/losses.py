from __future__ import annotations
import torch
import torch.nn as nn

class BCEWithLogitsLossWrapper(nn.Module):
    def __init__(self, pos_weight: float|None=None):
        super().__init__()
        if pos_weight is not None:
            self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.loss = nn.BCEWithLogitsLoss()
    def forward(self, logits, labels):
        return self.loss(logits.view(-1), labels.view(-1))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, labels):
        bce = nn.functional.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1), reduction='none')
        p = torch.sigmoid(logits.view(-1))
        pt = labels.view(-1) * p + (1-labels.view(-1)) * (1-p)
        loss = (self.alpha * (1-pt) ** self.gamma * bce).mean()
        return loss
