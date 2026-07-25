import torch
import torch.nn as nn


class HRMSNorm(nn.Module):
    """norms amps - not phases"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, z1, z2):
        # 1. Berechne die quaternionische Quadrate-Summe (|q|^2 = |z1|^2 + |z2|^2)
        ampsq = (z1 * z1.conj()).real + (z2 * z2.conj()).real
        ampsq = ampsq / 2.0

        # RMS berechnen
        rms = torch.rsqrt(ampsq.mean(dim=-1, keepdim=True) + self.eps)

        return (z1 * rms) * self.weight, (z2 * rms) * self.weight
