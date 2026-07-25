import torch.nn as nn
from .core import OctoLinear, octonion_mul


class SAIL(nn.Module):
    """
    Self-Activating Interference Layer (octonian).
    """

    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w_signal = OctoLinear(d_model, d_ff)
        self.w_gate = OctoLinear(d_model, d_ff)

        self.w_out = OctoLinear(d_ff, d_model)

    def forward(self, z1, z2, z3, z4):
        sig_z1, sig_z2, sig_z3, sig_z4 = self.w_signal(z1, z2, z3, z4)
        gate_z1, gate_z2, gate_z3, gate_z4 = self.w_gate(z1, z2, z3, z4)

        gated_z1, gated_z2, gated_z3, gated_z4 = octonion_mul(
            sig_z1, sig_z2, sig_z3, sig_z4, gate_z1, gate_z2, gate_z3, gate_z4
        )

        return self.w_out(gated_z1, gated_z2, gated_z3, gated_z4)
