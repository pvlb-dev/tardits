import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """🌀 rotates query/key pairwise in the complex plane based on position"""

    def __init__(self, head_size, block_size):
        super().__init__()
        self.head_size = head_size

        # prepare rotation frequencies - the input parameters for the rotation
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_size, 2).float() / head_size)
        )
        t = torch.arange(block_size, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", self.emb.cos())  # (block_size, head_size)
        self.cos_cached: torch.Tensor
        self.register_buffer("sin_cached", self.emb.sin())  # (block_size, head_size)
        self.sin_cached: torch.Tensor

    def _rotate_half(self, x):
        x1 = x[..., : self.head_size // 2]
        x2 = x[..., self.head_size // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        # x.shape == (B, nh, T, hs)
        T = x.shape[2]

        # slice to sequence length T
        cos = self.cos_cached[:T, :].unsqueeze(0).unsqueeze(1)  # (1,1,T,hs)
        sin = self.sin_cached[:T, :].unsqueeze(0).unsqueeze(1)  # (1,1,T,hs)

        return (x * cos) + (self._rotate_half(x) * sin)


class TrainableRoPE(nn.Module):
    """🌀🧠 Trainable 'askew' RoPE with head-size trainable parameters."""

    def __init__(self, head_size, block_size, n_head):
        super().__init__()
        self.head_size = head_size
        self.n_head = n_head

        self.bias = nn.Parameter(torch.zeros(n_head, head_size))
        self._reset_parameters()

        t = torch.arange(block_size, dtype=torch.float32)
        self.register_buffer("t", t)
        self.t: torch.Tensor

    def _reset_parameters(self):
        with torch.no_grad():
            self.bias.uniform_(-2 * math.pi, 2 * math.pi)

    def _rotate_half(self, x):
        x1 = x[..., : self.head_size]
        x2 = x[..., self.head_size :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        # x.shape == (B, nh, T, hs)
        T = x.shape[2]

        t_reshaped = self.t[:T].view(1, T, 1)  # (1, T, 1)

        bias_reshaped = self.bias.view(self.n_head, 1, self.head_size)  # (nh, 1, hs)

        freqs = t_reshaped * bias_reshaped  # (nh,T,hs)

        cos = freqs.cos().unsqueeze(0)
        sin = freqs.sin().unsqueeze(0)  # (1, nh, T, hs)

        return (x * cos) + (self._rotate_half(x) * sin)
