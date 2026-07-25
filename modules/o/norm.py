import torch
import torch.nn as nn


class ORMSNorm(nn.Module):
    """norms amps - not phases"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, z1, z2, z3, z4):
        ampsq = (
            (z1 * z1.conj()).real
            + (z2 * z2.conj()).real
            + (z3 * z3.conj()).real
            + (z4 * z4.conj()).real
        )

        ampsq = ampsq / 4.0

        rms = torch.rsqrt(ampsq.mean(dim=-1, keepdim=True) + self.eps)

        return (
            (z1 * rms) * self.weight,
            (z2 * rms) * self.weight,
            (z3 * rms) * self.weight,
            (z4 * rms) * self.weight,
        )
