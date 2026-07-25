import torch
import torch.nn as nn
import math

from .core import octonion_mul


class OctoRoPE(nn.Module):
    def __init__(self, head_size, block_size, n_head):
        super().__init__()
        self.head_size = head_size
        self.n_head = n_head

        # rotation speed:
        self.theta = nn.Parameter(torch.empty(n_head, head_size))
        self.axis_params = nn.Parameter(torch.empty(n_head, head_size, 7))
        self._reset_parameters()

        t = torch.arange(block_size, dtype=torch.float32)
        self.register_buffer("t", t)
        self.t: torch.Tensor

    def _reset_parameters(self):
        """unit circle init"""
        with torch.no_grad():
            self.theta.uniform_(-2 * math.pi, 2 * math.pi)

            axis = torch.randn(self.n_head, self.head_size, 7)
            axis_norm = torch.norm(axis, dim=-1, keepdim=True) + 1e-8
            self.axis_params.copy_(axis / axis_norm)

    def forward(self, z1, z2, z3, z4):
        # Input-Shapes: (B, nh, T, hs)
        B, nh, T, hs = z1.shape
        t_current = self.t[:T]  # (T,)

        axis = self.axis_params  # (nh, hs, 7)
        axis_norm = torch.norm(axis, dim=-1, keepdim=True) + 1e-8
        u = axis / axis_norm  # (nh, hs, 7)

        u_1 = u[..., 0].unsqueeze(0).unsqueeze(2)  # (1, nh, 1, hs)
        u_2 = u[..., 1].unsqueeze(0).unsqueeze(2)
        u_3 = u[..., 2].unsqueeze(0).unsqueeze(2)
        u_4 = u[..., 3].unsqueeze(0).unsqueeze(2)
        u_5 = u[..., 4].unsqueeze(0).unsqueeze(2)
        u_6 = u[..., 5].unsqueeze(0).unsqueeze(2)
        u_7 = u[..., 6].unsqueeze(0).unsqueeze(2)

        phi = torch.einsum("hs,t->hts", self.theta, t_current)  # (nh, T, hs)
        r_phi = phi.unsqueeze(0)  # (1, nh, T, hs)

        cos_p = torch.cos(r_phi)
        sin_p = torch.sin(r_phi)

        r_z1 = torch.complex(cos_p, sin_p * u_1)
        r_z2 = torch.complex(sin_p * u_2, sin_p * u_3)
        r_z3 = torch.complex(sin_p * u_4, sin_p * u_5)
        r_z4 = torch.complex(sin_p * u_6, sin_p * u_7)

        out_z1, out_z2, out_z3, out_z4 = octonion_mul(
            z1, z2, z3, z4, r_z1, r_z2, r_z3, r_z4
        )

        return out_z1, out_z2, out_z3, out_z4
