import torch
import torch.nn as nn
from torch.nn import functional as F

from tardits.modules import LMCoreMixin
from tardits.modules.conf import ModelConfig


class RotaryPositionalEmbedding(nn.Module):
    """🌀 Rotates query/key pairwise in the complex plane based on position."""

    def __init__(self, head_size, block_size):
        super().__init__()
        self.head_size = head_size

        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_size, 2).float() / head_size)
        )
        t = torch.arange(block_size, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", self.emb.cos(), persistent=False)
        self.register_buffer("sin_cached", self.emb.sin(), persistent=False)
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor

    def _rotate_half(self, x):
        x1 = x[..., : self.head_size // 2]
        x2 = x[..., self.head_size // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        # x.shape: (B, nh, T, hs)
        T = x.shape[2]
        cos = self.cos_cached[:T, :].unsqueeze(0).unsqueeze(1)  # (1, 1, T, hs)
        sin = self.sin_cached[:T, :].unsqueeze(0).unsqueeze(1)  # (1, 1, T, hs)

        return (x * cos) + (self._rotate_half(x) * sin)


RoPE = RotaryPositionalEmbedding


class SwiGLU(nn.Module):
    """🚀 SwiGLU Feed-Forward Network."""

    def __init__(self, n_embd, hidden_dim, dropout):
        super().__init__()
        self.w = nn.Linear(n_embd, hidden_dim, bias=False)
        self.v = nn.Linear(n_embd, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w(x))
        activated = gate * self.v(x)
        out = self.out_proj(activated)

        return self.dropout(out)


class CasualSelfAttention(nn.Module):
    """casual (and causal) multi-headed self-attention"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)

        self.rope = RoPE(self.head_size, config.block_size)

        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.tril: torch.Tensor

        self.dropout = nn.Dropout(config.dropout)

        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.proj_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)  # (B,T,3*C)
        q, k, v = qkv.split(C, dim=2)  # (B,T,C)

        # view(B,T,nh,hs): (B,T,C) == (B,T,nh*hs) --> (B,T,nh,hs)
        # transpose(1, 2): (B,T,nh,hs) --> (B,nh,T,hs)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # fancy RotaryPositionalEmbedding
        q = self.rope(q)
        k = self.rope(k)

        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)  # (B,nh,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,nh,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,nh,T,T)
        wei = self.dropout(wei)  # (B,nh,T,T)

        # weighted aggregation of value
        out = wei @ v  # (B,nh,T,T) @ (B,nh,T,hs) --> (B,nh,T,hs)

        out = out.transpose(1, 2).reshape(B, T, C)  # -> (B,T,nh,hs) -> (B,T,C)

        out = self.proj(out)
        out = self.proj_dropout(out)

        return out  # (B,T,C)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config):
        # n_embd: embedding dimension, n_head: number of heads
        super().__init__()
        self.sa = CasualSelfAttention(config)
        self.ffwd = SwiGLU(config.n_embd, int(config.n_embd * (8 / 3)), config.dropout)
        self.ln1 = nn.RMSNorm(config.n_embd)
        self.ln2 = nn.RMSNorm(config.n_embd)

    def forward(self, x):
        x = x + self.ln1(self.sa(x))  # x + because "residual connections"...
        x = x + self.ln2(self.ffwd(x))
        return x


class RealGPT(nn.Module, LMCoreMixin):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.ln_f = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)  # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
