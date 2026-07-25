import torch
import torch.nn as nn
import math


class SAIL(nn.Module):
    """⛵ Self-Activating Interference Layer"""

    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w_signal = nn.Linear(d_model, d_ff, bias=False, dtype=torch.complex64)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False, dtype=torch.complex64)

        self.w_out = nn.Linear(d_ff, d_model, bias=False, dtype=torch.complex64)

        self._reset_parameters()

    def _reset_parameters(self):
        """unit circle init"""
        with torch.no_grad():
            for w in [self.w_signal, self.w_gate, self.w_out]:
                # random phases between -2 and 2 pi
                phases = torch.empty_like(w.weight.real).uniform_(
                    -2 * math.pi, 2 * math.pi
                )

                scale = math.sqrt(1.0 / (w.in_features + w.out_features))

                w_comp = torch.complex(
                    torch.cos(phases) * scale, torch.sin(phases) * scale
                )

                w.weight.copy_(w_comp)

    def forward(self, z):
        z_signal = self.w_signal(z)
        z_gate = self.w_gate(z)
        # the naming is arbitrary. the layers are equally acting as signal and gate

        # 🌊 interference gating:
        gated = z_signal * z_gate

        return self.w_out(gated)
