import torch
import torch.nn as nn
from torch.nn import functional as F


from modules.h import SAIL, RoPE, Embedding, LMHead, RMSNorm, Linear
from modules import LMCoreMixin
from modules.conf import ModelConfig


class HyperSelfAttention(nn.Module):
    """quaternion (4D) multi-headed self-attention"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.q_proj = Linear(self.n_embd, self.n_embd)
        self.k_proj = Linear(self.n_embd, self.n_embd)
        self.v_proj = Linear(self.n_embd, self.n_embd)

        self.rope = RoPE(self.head_size, config.block_size, self.n_head)

        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.tril: torch.Tensor

        self.dropout = nn.Dropout(config.dropout)

        self.out_proj = Linear(config.n_embd, config.n_embd)

    def forward(self, z1, z2):
        B, T, C = z1.shape

        # project
        q1, q2 = self.q_proj(z1, z2)
        k1, k2 = self.k_proj(z1, z2)
        v1, v2 = self.v_proj(z1, z2)

        # split into heads
        # (B, T, C) -> (B, nh, T, hs)
        q1 = q1.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q2 = q2.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k1 = k1.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k2 = k2.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v1 = v1.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v2 = v2.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # HyperRoPE
        q1, q2 = self.rope(q1, q2)
        k1, k2 = self.rope(k1, k2)

        # quaternion scalar product Real(Q1 * conj(K1) + Q2 * conj(K2))
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        attn1 = q1 @ k1.transpose(-2, -1).conj()
        attn2 = q2 @ k2.transpose(-2, -1).conj()

        # extract real part
        wei = (attn1 + attn2).real * (4 * self.head_size) ** -0.5

        # causal mask & softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        wei = wei.to(v1.dtype)

        # @ v
        out_z1 = wei @ v1  # (B, nh, T, hs)
        out_z2 = wei @ v2  # (B, nh, T, hs)

        # recombine heads -> (B, T, C)
        out_z1 = out_z1.transpose(1, 2).reshape(B, T, C)
        out_z2 = out_z2.transpose(1, 2).reshape(B, T, C)

        # out project
        out_z1, out_z2 = self.out_proj(out_z1, out_z2)

        # dropout mask
        if self.training:
            mask = F.dropout(
                torch.ones_like(out_z1.real), p=self.config.dropout, training=True
            ) / (1.0 - self.config.dropout)
            out_z1 = torch.complex(out_z1.real * mask, out_z1.imag * mask)
            out_z2 = torch.complex(out_z2.real * mask, out_z2.imag * mask)

        return out_z1, out_z2


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config):
        # n_embd: embedding dimension, n_head: number of heads
        super().__init__()
        self.sa = HyperSelfAttention(config)
        self.ffwd = SAIL(d_model=config.n_embd, d_ff=config.n_embd * 1)
        self.ln1 = RMSNorm(config.n_embd)
        self.ln2 = RMSNorm(config.n_embd)

    def forward(self, z1, z2):
        # attention + residual
        sa_z1, sa_z2 = self.sa(z1, z2)
        norm_z1, norm_z2 = self.ln1(sa_z1, sa_z2)
        z1 = z1 + norm_z1
        z2 = z2 + norm_z2

        # ffwd + residual
        ff_z1, ff_z2 = self.ffwd(z1, z2)
        norm_ff_z1, norm_ff_z2 = self.ln2(ff_z1, ff_z2)
        z1 = z1 + norm_ff_z1
        z2 = z2 + norm_ff_z2

        return z1, z2


class HyperFormer(nn.Module, LMCoreMixin):
    """i**2 + j**2 + k**2 == i*k*j == -1"""

    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.config = config
        self.token_embedding = Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = LMHead(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        z1, z2 = self.token_embedding(idx)

        for block in self.blocks:
            z1, z2 = block(z1, z2)
        z1, z2 = self.ln_f(z1, z2)
        logits = self.lm_head(z1, z2)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
