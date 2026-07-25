import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.s import SAIL, RoPE, Embedding, LMHead, RMSNorm, Linear
from modules import LMCoreMixin
from modules.conf import ModelConfig


class SedenionSelfAttention(nn.Module):
    """sedenion  multi-headed self-attention"""

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

    def forward(self, z1, z2, z3, z4, z5, z6, z7, z8):
        B, T, C = z1.shape

        # okay... i know this is silly :P

        # project
        q1, q2, q3, q4, q5, q6, q7, q8 = self.q_proj(z1, z2, z3, z4, z5, z6, z7, z8)
        k1, k2, k3, k4, k5, k6, k7, k8 = self.k_proj(z1, z2, z3, z4, z5, z6, z7, z8)
        v1, v2, v3, v4, v5, v6, v7, v8 = self.v_proj(z1, z2, z3, z4, z5, z6, z7, z8)

        # split into heads
        # (B, T, C) -> (B, nh, T, hs)
        q1 = q1.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q2 = q2.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q3 = q3.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q4 = q4.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q5 = q5.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q6 = q6.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q7 = q7.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q8 = q8.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        k1 = k1.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k2 = k2.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k3 = k3.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k4 = k4.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k5 = k5.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k6 = k6.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k7 = k7.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k8 = k8.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        v1 = v1.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v2 = v2.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v3 = v3.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v4 = v4.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v5 = v5.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v6 = v6.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v7 = v7.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v8 = v8.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 3. HyperRoPE
        q1, q2, q3, q4, q5, q6, q7, q8 = self.rope(q1, q2, q3, q4, q5, q6, q7, q8)
        k1, k2, k3, k4, k5, k6, k7, k8 = self.rope(k1, k2, k3, k4, k5, k6, k7, k8)

        # 4. scalar product:
        attn1 = q1 @ k1.transpose(-2, -1).conj()
        attn2 = q2 @ k2.transpose(-2, -1).conj()
        attn3 = q3 @ k3.transpose(-2, -1).conj()
        attn4 = q4 @ k4.transpose(-2, -1).conj()
        attn5 = q5 @ k5.transpose(-2, -1).conj()
        attn6 = q6 @ k6.transpose(-2, -1).conj()
        attn7 = q7 @ k7.transpose(-2, -1).conj()
        attn8 = q8 @ k8.transpose(-2, -1).conj()

        # extract real part
        wei = (attn1 + attn2 + attn3 + attn4 + attn5 + attn6 + attn7 + attn8).real * (
            16 * self.head_size
        ) ** -0.5

        # causal Mask & softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        wei = wei.to(v1.dtype)

        # @ v
        out_z1 = wei @ v1  # (B, nh, T, hs)
        out_z2 = wei @ v2  # (B, nh, T, hs)
        out_z3 = wei @ v3  # (B, nh, T, hs)
        out_z4 = wei @ v4  # (B, nh, T, hs)
        out_z5 = wei @ v5  # (B, nh, T, hs)
        out_z6 = wei @ v6  # (B, nh, T, hs)
        out_z7 = wei @ v7  # (B, nh, T, hs)
        out_z8 = wei @ v8  # (B, nh, T, hs)

        # recombine heads -> (B, T, C)
        out_z1 = out_z1.transpose(1, 2).reshape(B, T, C)
        out_z2 = out_z2.transpose(1, 2).reshape(B, T, C)
        out_z3 = out_z3.transpose(1, 2).reshape(B, T, C)
        out_z4 = out_z4.transpose(1, 2).reshape(B, T, C)
        out_z5 = out_z5.transpose(1, 2).reshape(B, T, C)
        out_z6 = out_z6.transpose(1, 2).reshape(B, T, C)
        out_z7 = out_z7.transpose(1, 2).reshape(B, T, C)
        out_z8 = out_z8.transpose(1, 2).reshape(B, T, C)

        # 6. out projection
        (
            out_z1,
            out_z2,
            out_z3,
            out_z4,
            out_z5,
            out_z6,
            out_z7,
            out_z8,
        ) = self.out_proj(
            out_z1, out_z2, out_z3, out_z4, out_z5, out_z6, out_z7, out_z8
        )

        if self.training:
            mask = F.dropout(
                torch.ones_like(out_z1.real), p=self.config.dropout, training=True
            ) / (1.0 - self.config.dropout)
            out_z1 = torch.complex(out_z1.real * mask, out_z1.imag * mask)
            out_z2 = torch.complex(out_z2.real * mask, out_z2.imag * mask)
            out_z3 = torch.complex(out_z3.real * mask, out_z3.imag * mask)
            out_z4 = torch.complex(out_z4.real * mask, out_z4.imag * mask)
            out_z5 = torch.complex(out_z5.real * mask, out_z5.imag * mask)
            out_z6 = torch.complex(out_z6.real * mask, out_z6.imag * mask)
            out_z7 = torch.complex(out_z7.real * mask, out_z7.imag * mask)
            out_z8 = torch.complex(out_z8.real * mask, out_z8.imag * mask)

        return out_z1, out_z2, out_z3, out_z4, out_z5, out_z6, out_z7, out_z8


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config):
        # n_embd: embedding dimension, n_head: number of heads
        super().__init__()
        self.sa = SedenionSelfAttention(config)
        self.ffwd = SAIL(d_model=config.n_embd, d_ff=config.n_embd * 1)
        self.ln1 = RMSNorm(config.n_embd)
        self.ln2 = RMSNorm(config.n_embd)

    def forward(self, z1, z2, z3, z4, z5, z6, z7, z8):
        # attention + residual
        sa_z1, sa_z2, sa_z3, sa_z4, sa_z5, sa_z6, sa_z7, sa_z8 = self.sa(
            z1, z2, z3, z4, z5, z6, z7, z8
        )
        norm_z1, norm_z2, norm_z3, norm_z4, norm_z5, norm_z6, norm_z7, norm_z8 = (
            self.ln1(sa_z1, sa_z2, sa_z3, sa_z4, sa_z5, sa_z6, sa_z7, sa_z8)
        )
        z1 = z1 + norm_z1
        z2 = z2 + norm_z2
        z3 = z3 + norm_z3
        z4 = z4 + norm_z4
        z5 = z5 + norm_z5
        z6 = z6 + norm_z6
        z7 = z7 + norm_z7
        z8 = z8 + norm_z8

        # ffwd + residual
        ff_z1, ff_z2, ff_z3, ff_z4, ff_z5, ff_z6, ff_z7, ff_z8 = self.ffwd(
            z1, z2, z3, z4, z5, z6, z7, z8
        )

        (
            norm_ff_z1,
            norm_ff_z2,
            norm_ff_z3,
            norm_ff_z4,
            norm_ff_z5,
            norm_ff_z6,
            norm_ff_z7,
            norm_ff_z8,
        ) = self.ln2(ff_z1, ff_z2, ff_z3, ff_z4, ff_z5, ff_z6, ff_z7, ff_z8)
        z1 = z1 + norm_ff_z1
        z2 = z2 + norm_ff_z2
        z3 = z3 + norm_ff_z3
        z4 = z4 + norm_ff_z4
        z5 = z5 + norm_ff_z5
        z6 = z6 + norm_ff_z6
        z7 = z7 + norm_ff_z7
        z8 = z8 + norm_ff_z8

        return z1, z2, z3, z4, z5, z6, z7, z8


class HydraFormer(nn.Module, LMCoreMixin):
    """release the 16-dimensional sedenion hydra 🐍"""

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
        z1, z2, z3, z4, z5, z6, z7, z8 = self.token_embedding(idx)
        for block in self.blocks:
            z1, z2, z3, z4, z5, z6, z7, z8 = block(z1, z2, z3, z4, z5, z6, z7, z8)
        z1, z2, z3, z4, z5, z6, z7, z8 = self.ln_f(z1, z2, z3, z4, z5, z6, z7, z8)
        logits = self.lm_head(z1, z2, z3, z4, z5, z6, z7, z8)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
