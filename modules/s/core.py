import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from modules.o.core import octonion_mul


def sedenion_mul(z1, z2, z3, z4, z5, z6, z7, z8, w1, w2, w3, w4, w5, w6, w7, w8):
    a1, a2, a3, a4 = octonion_mul(z1, z2, z3, z4, w1, w2, w3, w4)

    b1, b2, b3, b4 = octonion_mul(torch.conj(w5), -w6, -w7, -w8, z5, z6, z7, z8)

    res_z1 = a1 - b1
    res_z2 = a2 - b2
    res_z3 = a3 - b3
    res_z4 = a4 - b4

    c1, c2, c3, c4 = octonion_mul(w5, w6, w7, w8, z1, z2, z3, z4)

    d1, d2, d3, d4 = octonion_mul(z5, z6, z7, z8, torch.conj(w1), -w2, -w3, -w4)

    res_z5 = c1 + d1
    res_z6 = c2 + d2
    res_z7 = c3 + d3
    res_z8 = c4 + d4

    return (res_z1, res_z2, res_z3, res_z4, res_z5, res_z6, res_z7, res_z8)


class SedeLinear(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        # 2 komplexe matrizen repräsentieren 1 quaternion weight
        self.w_z1 = nn.Parameter(torch.empty(out_feat, in_feat, dtype=torch.complex64))
        self.w_z2 = nn.Parameter(torch.empty(out_feat, in_feat, dtype=torch.complex64))
        self.w_z3 = nn.Parameter(torch.empty(out_feat, in_feat, dtype=torch.complex64))
        self.w_z4 = nn.Parameter(torch.empty(out_feat, in_feat, dtype=torch.complex64))
        self.w_z5 = nn.Parameter(torch.empty(out_feat, in_feat, dtype=torch.complex64))
        self.w_z6 = nn.Parameter(torch.empty(out_feat, in_feat, dtype=torch.complex64))
        self.w_z7 = nn.Parameter(torch.empty(out_feat, in_feat, dtype=torch.complex64))
        self.w_z8 = nn.Parameter(torch.empty(out_feat, in_feat, dtype=torch.complex64))

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
            r9 = torch.randn_like(self.w_z1.real)
            r10 = torch.randn_like(self.w_z1.real)
            r11 = torch.randn_like(self.w_z1.real)
            r12 = torch.randn_like(self.w_z1.real)
            r13 = torch.randn_like(self.w_z1.real)
            r14 = torch.randn_like(self.w_z1.real)
            r15 = torch.randn_like(self.w_z1.real)
            r16 = torch.randn_like(self.w_z1.real)

            norm = torch.sqrt(
                r1**2
                + r2**2
                + r3**2
                + r4**2
                + r5**2
                + r6**2
                + r7**2
                + r8**2
                + r9**2
                + r10**2
                + r11**2
                + r12**2
                + r13**2
                + r14**2
                + r15**2
                + r16**2
                + 1e-8
            )  # 1e-8 to not divide by 0
            r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16 = (
                r1 / norm,
                r2 / norm,
                r3 / norm,
                r4 / norm,
                r5 / norm,
                r6 / norm,
                r7 / norm,
                r8 / norm,
                r9 / norm,
                r10 / norm,
                r11 / norm,
                r12 / norm,
                r13 / norm,
                r14 / norm,
                r15 / norm,
                r16 / norm,
            )

            scale = math.sqrt(0.125 / (self.in_feat + self.out_feat))

            self.w_z1.copy_(torch.complex(r1 * scale, r2 * scale))
            self.w_z2.copy_(torch.complex(r3 * scale, r4 * scale))
            self.w_z3.copy_(torch.complex(r5 * scale, r6 * scale))
            self.w_z4.copy_(torch.complex(r7 * scale, r8 * scale))

            self.w_z5.copy_(torch.complex(r9 * scale, r10 * scale))
            self.w_z6.copy_(torch.complex(r11 * scale, r12 * scale))
            self.w_z7.copy_(torch.complex(r13 * scale, r14 * scale))
            self.w_z8.copy_(torch.complex(r15 * scale, r16 * scale))

    def forward(self, x_z1, x_z2, x_z3, x_z4, x_z5, x_z6, x_z7, x_z8):
        p1p2_z1 = (
            F.linear(x_z1, self.w_z1)
            - F.linear(x_z2, self.w_z2.conj())
            - F.linear(x_z3, self.w_z3.conj())
            - F.linear(x_z4, self.w_z4.conj())
        )
        p1p2_z2 = (
            F.linear(x_z1, self.w_z2)
            + F.linear(x_z2, self.w_z1.conj())
            + F.linear(x_z3, self.w_z4.conj())
            - F.linear(x_z4, self.w_z3.conj())
        )
        p1p2_z3 = (
            F.linear(x_z1, self.w_z3)
            - F.linear(x_z2, self.w_z4.conj())
            + F.linear(x_z3, self.w_z1.conj())
            + F.linear(x_z4, self.w_z2.conj())
        )
        p1p2_z4 = (
            F.linear(x_z1, self.w_z4)
            + F.linear(x_z2, self.w_z3.conj())
            - F.linear(x_z3, self.w_z2.conj())
            + F.linear(x_z4, self.w_z1.conj())
        )

        w5_c, w6_n, w7_n, w8_n = self.w_z5.conj(), -self.w_z6, -self.w_z7, -self.w_z8

        q2q1_z1 = (
            F.linear(x_z5, w5_c)
            - F.linear(x_z6, w6_n.conj())
            - F.linear(x_z7, w7_n.conj())
            - F.linear(x_z8, w8_n.conj())
        )
        q2q1_z2 = (
            F.linear(x_z5, w6_n)
            + F.linear(x_z6, w5_c.conj())
            + F.linear(x_z7, w8_n.conj())
            - F.linear(x_z8, w7_n.conj())
        )
        q2q1_z3 = (
            F.linear(x_z5, w7_n)
            - F.linear(x_z6, w8_n.conj())
            + F.linear(x_z7, w5_c.conj())
            + F.linear(x_z8, w6_n.conj())
        )
        q2q1_z4 = (
            F.linear(x_z5, w8_n)
            + F.linear(x_z6, w7_n.conj())
            - F.linear(x_z7, w6_n.conj())
            + F.linear(x_z8, w5_c.conj())
        )

        out_z1 = p1p2_z1 - q2q1_z1
        out_z2 = p1p2_z2 - q2q1_z2
        out_z3 = p1p2_z3 - q2q1_z3
        out_z4 = p1p2_z4 - q2q1_z4

        q2p1_z5 = (
            F.linear(x_z1, self.w_z5)
            - F.linear(x_z2, self.w_z6.conj())
            - F.linear(x_z3, self.w_z7.conj())
            - F.linear(x_z4, self.w_z8.conj())
        )
        q2p1_z6 = (
            F.linear(x_z1, self.w_z6)
            + F.linear(x_z2, self.w_z5.conj())
            + F.linear(x_z3, self.w_z8.conj())
            - F.linear(x_z4, self.w_z7.conj())
        )
        q2p1_z7 = (
            F.linear(x_z1, self.w_z7)
            - F.linear(x_z2, self.w_z6.conj())
            + F.linear(x_z3, self.w_z5.conj())
            + F.linear(x_z4, self.w_z6.conj())
        )
        q2p1_z8 = (
            F.linear(x_z1, self.w_z8)
            + F.linear(x_z2, self.w_z7.conj())
            - F.linear(x_z3, self.w_z6.conj())
            + F.linear(x_z4, self.w_z5.conj())
        )

        w1_c, w2_n, w3_n, w4_n = self.w_z1.conj(), -self.w_z2, -self.w_z3, -self.w_z4

        q1p2_z5 = (
            F.linear(x_z5, w1_c)
            - F.linear(x_z6, w2_n.conj())
            - F.linear(x_z7, w3_n.conj())
            - F.linear(x_z8, w4_n.conj())
        )
        q1p2_z6 = (
            F.linear(x_z5, w2_n)
            + F.linear(x_z6, w1_c.conj())
            + F.linear(x_z7, w4_n.conj())
            - F.linear(x_z8, w3_n.conj())
        )
        q1p2_z7 = (
            F.linear(x_z5, w3_n)
            - F.linear(x_z6, w4_n.conj())
            + F.linear(x_z7, w1_c.conj())
            + F.linear(x_z8, w2_n.conj())
        )
        q1p2_z8 = (
            F.linear(x_z5, w4_n)
            + F.linear(x_z6, w3_n.conj())
            - F.linear(x_z7, w2_n.conj())
            + F.linear(x_z8, w1_c.conj())
        )

        out_z5 = q2p1_z5 + q1p2_z5
        out_z6 = q2p1_z6 + q1p2_z6
        out_z7 = q2p1_z7 + q1p2_z7
        out_z8 = q2p1_z8 + q1p2_z8

        return out_z1, out_z2, out_z3, out_z4, out_z5, out_z6, out_z7, out_z8
