import torch
import torch.nn as nn
import math


class HyperRoPE(nn.Module):
    """trainable 3D-RoPE (quaternion)"""

    def __init__(self, head_size, block_size, n_head):
        super().__init__()
        self.head_size = head_size
        self.n_head = n_head

        # rotation speed:
        self.theta = nn.Parameter(torch.empty(n_head, head_size))
        self.axis_params = nn.Parameter(torch.empty(n_head, head_size, 3))
        self._reset_parameters()

        t = torch.arange(block_size, dtype=torch.float32)
        self.register_buffer("t", t)
        self.t: torch.Tensor

    def _reset_parameters(self):
        """unit circle init"""
        with torch.no_grad():
            self.theta.uniform_(-2 * math.pi, 2 * math.pi)

            axis = torch.randn(self.n_head, self.head_size, 3)
            axis_norm = torch.norm(axis, dim=-1, keepdim=True) + 1e-8
            self.axis_params.copy_(axis / axis_norm)

    def forward(self, z1, z2):
        # Input-Shapes: (B, nh, T, hs)
        B, nh, T, hs = z1.shape
        t_current = self.t[:T]  # (T,)

        # 1. Rotationsachsen normieren
        axis = self.axis_params  # (nh, hs, 3)
        axis_norm = torch.norm(axis, dim=-1, keepdim=True) + 1e-8
        u = axis / axis_norm  # (nh, hs, 3)

        # Einzelne Achsen-Hälften holen -> jeweils (nh, hs)
        u_x, u_y, u_z = u[..., 0], u[..., 1], u[..., 2]

        # 2. Rotationswinkel pro Zeitschritt berechnen
        # self.theta ist (nh, hs). t_current ist (T,)
        # Wir wollen am Ende (nh, T, hs) herausbekommen!
        # h: n_head, s: head_size, t: timestep
        phi = torch.einsum("hs,t->hts", self.theta, t_current)  # -> Shape: (nh, T, hs)

        # 3. Dimensionen für Broad-Casting auf (1, nh, T, hs) vorbereiten:
        # Achsen-Vektoren haben (nh, hs) -> wir fügen Dimensionen für B (vorne) und T (Mitte) hinzu
        u_x = u_x.unsqueeze(0).unsqueeze(2)  # (1, nh, 1, hs)
        u_y = u_y.unsqueeze(0).unsqueeze(2)  # (1, nh, 1, hs)
        u_z = u_z.unsqueeze(0).unsqueeze(2)  # (1, nh, 1, hs)

        # phi hat (nh, T, hs) -> wir brauchen nur eine 1 ganz vorne für Batch-Size
        r_phi = phi.unsqueeze(0)  # (1, nh, T, hs)

        # 4. Quaternionen-Rotator R = cos(phi) + sin(phi)*(u_x*i + u_y*j + u_z*k)
        # R = (r_z1, r_z2)
        r_z1 = torch.complex(torch.cos(r_phi), torch.sin(r_phi) * u_x)
        r_z2 = torch.complex(torch.sin(r_phi) * u_y, torch.sin(r_phi) * u_z)

        # 5. Hamilton-Produkt ausführen: out = input * R
        out_z1 = z1 * r_z1 - z2 * r_z2.conj()
        out_z2 = z1 * r_z2 + z2 * r_z1.conj()

        return out_z1, out_z2
