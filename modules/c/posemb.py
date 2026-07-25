import torch
import torch.nn as nn
import math


class GyRoPE(nn.Module):
    """trainable RoPE - range: 4pi"""

    def __init__(self, head_size, block_size, n_head):
        super().__init__()
        self.head_size = head_size
        self.n_head = n_head
        self.roparam = nn.Parameter(torch.empty(n_head, head_size, 2))
        self._reset_parameters()

        t = torch.arange(block_size, dtype=torch.float32)
        self.register_buffer("t", t)
        self.t: torch.Tensor

    def _reset_parameters(self):
        """unit circle init"""
        with torch.no_grad():
            # random phases between -2 pi and 2 pi
            phases = torch.empty(self.n_head, self.head_size).uniform_(
                -2 * math.pi, 2 * math.pi
            )

            self.roparam[..., 0].copy_(torch.cos(phases))
            self.roparam[..., 1].copy_(torch.sin(phases))

    def forward(self, z):
        # z.shape == (B, nh, T, hs)
        T = z.shape[2]

        freq = torch.view_as_complex(self.roparam)
        angles = torch.angle(freq)

        t_current = self.t[:T]
        rot_phases = torch.einsum("nh,t->nth", angles, t_current)
        rotator = torch.complex(torch.cos(rot_phases), torch.sin(rot_phases))

        return z * rotator.unsqueeze(0)
