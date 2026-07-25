import torch
import torch.nn as nn


class SRMSNorm(nn.Module):
    """norms amps - not phases"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, z1, z2, z3, z4, z5, z6, z7, z8):
        ampsq = (
            (z1 * z1.conj()).real
            + (z2 * z2.conj()).real
            + (z3 * z3.conj()).real
            + (z4 * z4.conj()).real
            + (z5 * z5.conj()).real
            + (z6 * z6.conj()).real
            + (z7 * z7.conj()).real
            + (z8 * z8.conj()).real
        )

        ampsq = ampsq / 8.0

        rms = torch.rsqrt(ampsq.mean(dim=-1, keepdim=True) + self.eps)

        return (
            (z1 * rms) * self.weight,
            (z2 * rms) * self.weight,
            (z3 * rms) * self.weight,
            (z4 * rms) * self.weight,
            (z5 * rms) * self.weight,
            (z6 * rms) * self.weight,
            (z7 * rms) * self.weight,
            (z8 * rms) * self.weight,
        )
