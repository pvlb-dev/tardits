import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class HyperLinear(nn.Module):
    """
    quaternion Linear-Layer based on CD-constructs
    """

    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_z1 = nn.Parameter(torch.empty(out_feat, in_feat, dtype=torch.complex64))
        self.w_z2 = nn.Parameter(torch.empty(out_feat, in_feat, dtype=torch.complex64))

        self._reset_parameters()

    def _reset_parameters(self):
        """
        unit-sphere init with xavier scaling
        """
        with torch.no_grad():
            r1 = torch.randn_like(self.w_z1.real)
            r2 = torch.randn_like(self.w_z1.real)
            r3 = torch.randn_like(self.w_z1.real)
            r4 = torch.randn_like(self.w_z1.real)

            norm = torch.sqrt(
                r1**2 + r2**2 + r3**2 + r4**2 + 1e-8
            )  # 1e-8 to not divide by 0
            r1, r2, r3, r4 = r1 / norm, r2 / norm, r3 / norm, r4 / norm

            scale = math.sqrt(0.5 / (self.in_feat + self.out_feat))

            self.w_z1.copy_(torch.complex(r1 * scale, r2 * scale))
            self.w_z2.copy_(torch.complex(r3 * scale, r4 * scale))

    def forward(self, x_z1, x_z2):

        out_z1 = F.linear(x_z1, self.w_z1) - F.linear(x_z2, self.w_z2.conj())
        out_z2 = F.linear(x_z1, self.w_z2) + F.linear(x_z2, self.w_z1.conj())

        return out_z1, out_z2
