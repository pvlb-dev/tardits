import torch.nn as nn
from .core import SedeLinear, sedenion_mul


class SAIL(nn.Module):
    """
    Self-Activating Interference Layer (sedenion).
    """

    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w_signal = SedeLinear(d_model, d_ff)
        self.w_gate = SedeLinear(d_model, d_ff)

        self.w_out = SedeLinear(d_ff, d_model)

    def forward(self, z1, z2, z3, z4, z5, z6, z7, z8):
        sig_z1, sig_z2, sig_z3, sig_z4, sig_z5, sig_z6, sig_z7, sig_z8 = self.w_signal(
            z1, z2, z3, z4, z5, z6, z7, z8
        )
        gate_z1, gate_z2, gate_z3, gate_z4, gate_z5, gate_z6, gate_z7, gate_z8 = (
            self.w_gate(z1, z2, z3, z4, z5, z6, z7, z8)
        )

        (
            gated_z1,
            gated_z2,
            gated_z3,
            gated_z4,
            gated_z5,
            gated_z6,
            gated_z7,
            gated_z8,
        ) = sedenion_mul(
            sig_z1,
            sig_z2,
            sig_z3,
            sig_z4,
            sig_z5,
            sig_z6,
            sig_z7,
            sig_z8,
            gate_z1,
            gate_z2,
            gate_z3,
            gate_z4,
            gate_z5,
            gate_z6,
            gate_z7,
            gate_z8,
        )

        return self.w_out(
            gated_z1,
            gated_z2,
            gated_z3,
            gated_z4,
            gated_z5,
            gated_z6,
            gated_z7,
            gated_z8,
        )
