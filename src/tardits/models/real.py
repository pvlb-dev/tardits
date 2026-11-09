import torch
import torch.nn as nn
from torch.nn import functional as F

from tardits.modules import LMCoreMixin
from tardits.models.base import ModelConfig


class RotaryPositionalEmbedding(nn.Module):
    """🌀 Rotates query/key pairwise in the complex plane based on position."""

    def __init__(self, head_size, block_size):
        super().__init__()
        self.head_size = head_size

        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_size, 2).float() / head_size)
        )
        # Kept as a non-persistent buffer so the cache can be grown lazily for
        # long-context decoding without changing checkpoint keys.
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._extend_cache(block_size)

    def _extend_cache(self, size: int):
        """Grow the cos/sin cache to cover at least ``size`` positions."""
        current = self.cos_cached.shape[0]
        if size <= current:
            return
        # Double when possible so repeated decode steps do not reallocate.
        target = max(size, 2 * current)
        positions = torch.arange(
            current, target, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = torch.cat((self.cos_cached, emb.cos()), dim=0)
        self.sin_cached = torch.cat((self.sin_cached, emb.sin()), dim=0)

    def _rotate_half(self, x):
        x1 = x[..., : self.head_size // 2]
        x2 = x[..., self.head_size // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, start_pos: int = 0):
        # x.shape: (B, nh, T, hs)
        T = x.shape[2]
        # Long-context decoding can walk past the training window; grow the
        # cache on demand instead of erroring (positions > block_size are
        # extrapolation, which is what long-context benchmarks measure).
        self._extend_cache(start_pos + T)
        cos = (
            self.cos_cached[start_pos : start_pos + T, :]
            .unsqueeze(0)
            .unsqueeze(1)
        )  # (1, 1, T, hs)
        sin = (
            self.sin_cached[start_pos : start_pos + T, :]
            .unsqueeze(0)
            .unsqueeze(1)
        )  # (1, 1, T, hs)

        return (x * cos) + (self._rotate_half(x) * sin)


class KVCache:
    """Per-layer key/value cache for O(T) RealGPT decoding."""

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def append(self, k: torch.Tensor, v: torch.Tensor):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat((self.k, k), dim=2)
            self.v = torch.cat((self.v, v), dim=2)
        return self.k, self.v

    def __len__(self):
        return 0 if self.k is None else self.k.shape[2]


def _sample_next(logits, temp: float = 1.0, top_k: int = 10):
    logits = logits / temp
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float("Inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


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

        if self.head_size % 2 != 0:
            raise ValueError(
                f"RealGPT RoPE needs an even head_size, got {self.head_size} "
                f"(n_embd={self.n_embd}, n_head={self.n_head})"
            )

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)

        self.rope = RoPE(self.head_size, config.block_size)

        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.tril: torch.Tensor

        self.dropout = nn.Dropout(config.dropout)

        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.proj_dropout = nn.Dropout(config.dropout)

    def forward(self, x, kv_cache: KVCache | None = None, start_pos: int = 0):
        B, T, C = x.shape

        # The training window is block_size. Cached prefill may legitimately
        # exceed it (chunked prefill of a long prompt); keep the guard for the
        # uncached/training path so a misconfigured batch still fails loudly.
        if T > self.config.block_size and kv_cache is None:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.config.block_size}"
            )

        qkv = self.c_attn(x)  # (B,T,3*C)
        q, k, v = qkv.split(C, dim=2)  # (B,T,C)

        # view(B,T,nh,hs): (B,T,C) == (B,T,nh*hs) --> (B,T,nh,hs)
        # transpose(1, 2): (B,T,nh,hs) --> (B,nh,T,hs)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # fancy RotaryPositionalEmbedding, offset for cached decoding
        q = self.rope(q, start_pos=start_pos)
        k = self.rope(k, start_pos=start_pos)

        if kv_cache is not None:
            k, v = kv_cache.append(k, v)

        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)  # (B,nh,T,T_kv)
        if T > 1:
            # Causal mask for prefill/training; for a cached single-token step
            # every key in the cache is in the past, so no mask is needed.
            q_pos = torch.arange(start_pos, start_pos + T, device=x.device)
            k_pos = torch.arange(k.shape[-2], device=x.device)
            mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            wei = wei.masked_fill(~mask, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B,nh,T,T_kv)
        wei = self.dropout(wei)  # (B,nh,T,T_kv)

        # weighted aggregation of value
        out = wei @ v  # (B,nh,T,T_kv) @ (B,nh,T_kv,hs) --> (B,nh,T,hs)

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
        # getattr keeps checkpoints/configs pickled before the field existed loadable.
        self.norm = getattr(config, "norm", "branch")

    def forward(self, x, kv_cache: KVCache | None = None, start_pos: int = 0):
        if self.norm == "pre":
            # Canonical pre-norm: normalize the sublayer input.
            x = x + self.sa(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos)
            x = x + self.ffwd(self.ln2(x))
            return x
        # Deliberate branch/post-norm variant: the *branch output* is normalized
        # before it is added back to the residual stream.
        x = x + self.ln1(
            self.sa(x, kv_cache=kv_cache, start_pos=start_pos)
        )  # x + because "residual connections"...
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

    def forward(self, idx, targets=None, kv_cache=None, start_pos: int = 0):
        B, T = idx.shape
        x = self.token_embedding_table(idx)  # (B,T,C)
        caches = kv_cache if kv_cache is not None else [None] * len(self.blocks)
        for block, cache in zip(self.blocks, caches):
            x = block(x, kv_cache=cache, start_pos=start_pos)
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

    @torch.no_grad()
    def prefill(self, idx, caches):
        """Fill per-layer KV caches for ``idx`` in ``block_size`` chunks.

        Chunking keeps prefill memory bounded when the prompt is longer than the
        training window; the causal mask and RoPE extension make the chunked
        result identical to one long forward pass.
        """
        chunk = self.config.block_size
        logits = None
        for start in range(0, idx.shape[1], chunk):
            logits, _ = self.forward(
                idx[:, start : start + chunk], kv_cache=caches, start_pos=start
            )
        return logits

    @torch.no_grad()
    def generate_cached(
        self, idx, max_new_tokens: int, temp: float = 1.0, top_k: int = 10
    ):
        """KV-cached generation: prefill once, then O(T) work per token.

        This is the fair baseline for RAT's O(1) recurrence. The result should
        match ``generate`` token-for-token for the same RNG seed within the
        training window; beyond it, ``generate`` crops to ``block_size`` while
        this path keeps the full cache.
        """
        device = next(self.parameters()).device
        idx = idx.to(device)
        if max_new_tokens <= 0:
            return idx

        caches = [KVCache() for _ in self.blocks]
        logits = self.prefill(idx, caches)
        next_token = _sample_next(logits[:, -1, :], temp, top_k)
        idx = torch.cat((idx, next_token), dim=1)

        for _ in range(max_new_tokens - 1):
            start_pos = idx.shape[1] - 1
            logits, _ = self.forward(
                next_token, kv_cache=caches, start_pos=start_pos
            )
            next_token = _sample_next(logits[:, -1, :], temp, top_k)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
