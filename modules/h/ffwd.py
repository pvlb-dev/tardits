import torch.nn as nn
from .core import HyperLinear


class SAIL(nn.Module):
    """
    Self Activating Interference Layer in 4D
    """

    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w_signal = HyperLinear(d_model, d_ff)
        self.w_gate = HyperLinear(d_model, d_ff)

        self.w_out = HyperLinear(d_ff, d_model)

    def forward(self, z1, z2):
        # projection for signal and gate
        sig_z1, sig_z2 = self.w_signal(z1, z2)
        gate_z1, gate_z2 = self.w_gate(z1, z2)

        # 🌊 interference gating:
        # gated = signal * gate
        gated_z1 = sig_z1 * gate_z1 - sig_z2 * gate_z2.conj()
        gated_z2 = sig_z1 * gate_z2 + sig_z2 * gate_z1.conj()

        return self.w_out(gated_z1, gated_z2)
