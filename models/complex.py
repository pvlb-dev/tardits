import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from modules.c import SAIL, RoPE, Embedding, LMHead, RMSNorm
from modules import LMCoreMixin
from modules.conf import ModelConfig


class ComplexSelfAttention(nn.Module):
    """complex multi-headed self-attention"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(
            self.n_embd, 3 * self.n_embd, bias=False, dtype=torch.complex64
        )

        self.rope = RoPE(self.head_size, config.block_size, self.n_head)

        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.tril: torch.Tensor

        self.dropout = nn.Dropout(config.dropout)

        self.proj = nn.Linear(config.n_embd, config.n_embd, dtype=torch.complex64)
        self.proj_dropout = nn.Dropout(config.dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        """unit circle init"""
        with torch.no_grad():
            for w in [self.c_attn, self.proj]:
                # random phases between -2 pi and 2 pi
                phases = torch.empty_like(w.weight.real).uniform_(
                    -2 * math.pi, 2 * math.pi
                )

                scale = math.sqrt(1 / (w.in_features + w.out_features))

                w_comp = torch.complex(
                    torch.cos(phases) * scale, torch.sin(phases) * scale
                )

                w.weight.copy_(w_comp)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)  # (B,T,3*C)
        q, k, v = qkv.split(C, dim=2)  # (B,T,C)

        # view(B,T,nh,hs): (B,T,C) == (B,T,nh*hs) --> (B,T,nh,hs)
        # transpose(1, 2): (B,T,nh,hs) --> (B,nh,T,hs)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        wei = q @ k.adjoint() * (2 * self.head_size) ** -0.5
        wei = wei.real
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,nh,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,nh,T,T)

        wei = self.dropout(wei)  # (B,nh,T,T)

        wei = wei.to(torch.complex64)

        # weighted aggregation of value
        out = wei @ v  # (B,nh,T,T) @ (B,nh,T,hs) --> (B,nh,T,hs)

        out = out.transpose(1, 2).reshape(B, T, C)  # -> (B,T,nh,hs) -> (B,T,C)

        out = self.proj(out)
        if self.training:
            mask = F.dropout(
                torch.ones_like(out.real), p=self.config.dropout, training=True
            ) / (1.0 - self.config.dropout)
            out = torch.complex(out.real * mask, out.imag * mask)

        return out  # (B,T,C)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config):
        # n_embd: embedding dimension, n_head: number of heads
        super().__init__()
        self.sa = ComplexSelfAttention(config)
        self.ffwd = SAIL(d_model=config.n_embd, d_ff=config.n_embd)
        self.ln1 = RMSNorm(config.n_embd)
        self.ln2 = RMSNorm(config.n_embd)

    def forward(self, z):
        z = z + self.ln1(self.sa(z))  # residuals
        z = z + self.ln2(self.ffwd(z))

        return z


class InterFormer(nn.Module, LMCoreMixin):
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
        z = self.token_embedding(idx)
        z = self.blocks(z)
        z = self.ln_f(z)
        logits = self.lm_head(z)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
