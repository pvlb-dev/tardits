import torch
import torch.nn as nn
from torch.nn import functional as F
import math


def quaternion_mul_helper(z1, z2, w1, w2):
    res_z1 = z1 * w1 - z2 * torch.conj(w2)

    res_z2 = z1 * w2 + z2 * torch.conj(w1)

    return res_z1, res_z2


def quaternion_conj_helper(z1, z2):
    return torch.conj(z1), -z2


def octonion_mul(z1, z2, z3, z4, w1, w2, w3, w4):
    p1p2_z1, p1p2_z2 = quaternion_mul_helper(z1, z2, w1, w2)

    q2_conj_z3, q2_conj_z4 = quaternion_conj_helper(w3, w4)

    q2q1_z1, q2q1_z2 = quaternion_mul_helper(q2_conj_z3, q2_conj_z4, z3, z4)

    res_z1 = p1p2_z1 - q2q1_z1
    res_z2 = p1p2_z2 - q2q1_z2

    q2p1_z3, q2p1_z4 = quaternion_mul_helper(w3, w4, z1, z2)

    p2_conj_z1, p2_conj_z2 = quaternion_conj_helper(w1, w2)

    q1p2_z3, q1p2_z4 = quaternion_mul_helper(z3, z4, p2_conj_z1, p2_conj_z2)

    res_z3 = q2p1_z3 + q1p2_z3
    res_z4 = q2p1_z4 + q1p2_z4

    return res_z1, res_z2, res_z3, res_z4


class OctoLinear(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_z1 = nn.Parameter(torch.empty(out_feat, in_feat, dtype=torch.complex64))
        self.w_z2 = nn.Parameter(torch.empty(out_feat, in_feat, dtype=torch.complex64))
        self.w_z3 = nn.Parameter(torch.empty(out_feat, in_feat, dtype=torch.complex64))
        self.w_z4 = nn.Parameter(torch.empty(out_feat, in_feat, dtype=torch.complex64))

        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            r1 = torch.randn_like(self.w_z1.real)
            r2 = torch.randn_like(self.w_z1.real)
            r3 = torch.randn_like(self.w_z1.real)
            r4 = torch.randn_like(self.w_z1.real)
            r5 = torch.randn_like(self.w_z1.real)
            r6 = torch.randn_like(self.w_z1.real)
            r7 = torch.randn_like(self.w_z1.real)
            r8 = torch.randn_like(self.w_z1.real)

            norm = torch.sqrt(
                r1**2 + r2**2 + r3**2 + r4**2 + r5**2 + r6**2 + r7**2 + r8**2 + 1e-8
            )  # 1e-8 to not divide by 0
            r1, r2, r3, r4, r5, r6, r7, r8 = (
                r1 / norm,
                r2 / norm,
                r3 / norm,
                r4 / norm,
                r5 / norm,
                r6 / norm,
                r7 / norm,
                r8 / norm,
            )

            scale = math.sqrt(0.25 / (self.in_feat + self.out_feat))

            self.w_z1.copy_(torch.complex(r1 * scale, r2 * scale))
            self.w_z2.copy_(torch.complex(r3 * scale, r4 * scale))
            self.w_z3.copy_(torch.complex(r5 * scale, r6 * scale))
            self.w_z4.copy_(torch.complex(r7 * scale, r8 * scale))

    def forward(self, x_z1, x_z2, x_z3, x_z4):
        # Term 1a: X_p * W_p
        p1p2_z1 = F.linear(x_z1, self.w_z1) - F.linear(x_z2, self.w_z2.conj())
        p1p2_z2 = F.linear(x_z1, self.w_z2) + F.linear(x_z2, self.w_z1.conj())

        # Term 1b: W_q_conj * X_q
        w_z3_conj = self.w_z3.conj()
        w_z4_neg = -self.w_z4

        q2q1_z1 = F.linear(x_z3, w_z3_conj) - F.linear(x_z4, w_z4_neg.conj())
        q2q1_z2 = F.linear(x_z3, w_z4_neg) + F.linear(x_z4, w_z3_conj.conj())

        out_z1 = p1p2_z1 - q2q1_z1
        out_z2 = p1p2_z2 - q2q1_z2

        # Term 2a:
        q2p1_z3 = F.linear(x_z1, self.w_z3) - F.linear(x_z2, self.w_z4.conj())
        q2p1_z4 = F.linear(x_z1, self.w_z4) + F.linear(x_z2, self.w_z3.conj())

        # Term 2b: X_q * W_p_conj
        w_z1_conj = self.w_z1.conj()
        w_z2_neg = -self.w_z2

        q1p2_z3 = F.linear(x_z3, w_z1_conj) - F.linear(x_z4, w_z2_neg.conj())
        q1p2_z4 = F.linear(x_z3, w_z2_neg) + F.linear(x_z4, w_z1_conj.conj())

        out_z3 = q2p1_z3 + q1p2_z3
        out_z4 = q2p1_z4 + q1p2_z4

        return out_z1, out_z2, out_z3, out_z4
