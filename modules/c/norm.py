import torch
import torch.nn as nn


class CRMSNorm(nn.Module):
    """norms amps - not phases"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, z):
        ampsq = (z * z.conj()).real

        # RMS berechnen
        rms = torch.rsqrt(ampsq.mean(dim=-1, keepdim=True) + self.eps)

        return (z * rms) * self.weight
