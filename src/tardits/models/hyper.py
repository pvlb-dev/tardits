import torch
import torch.nn as nn
from torch.nn import functional as F

from tardits.modules import (
    HyperEmbedding,
    HyperLMHead,
    HyperRMSNorm,
    SAIL,
    GyRoPE,
    HyperLinear,
    LMCoreMixin
)
from tardits.modules.conf import ModelConfig


class HyperSelfAttention(nn.Module):
    """
    Universal multi-headed self-attention for 2^level hypercomplex spaces.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.level = config.level
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.hyper_dim = 2**config.level

        # Projections
        self.q_proj = HyperLinear(self.n_embd, self.n_embd, level=self.level)
        self.k_proj = HyperLinear(self.n_embd, self.n_embd, level=self.level)
        self.v_proj = HyperLinear(self.n_embd, self.n_embd, level=self.level)
        self.out_proj = HyperLinear(self.n_embd, self.n_embd, level=self.level)

        self.rope = GyRoPE(
            self.head_size, config.block_size, self.n_head, level=self.level
        )

        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.tril: torch.Tensor

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x Shape: [B, T, C, hyper_dim]
        B, T, C, hyper_dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_head, self.head_size, hyper_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size, hyper_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size, hyper_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        # [B, nh, T, hs, hyper_dim] -> [B, nh, T, hs * hyper_dim]
        q_flat = q.flatten(start_dim=-2)
        k_flat = k.flatten(start_dim=-2)
        v_flat = v.flatten(start_dim=-2)

        wei = q_flat @ k_flat.adjoint() * (self.hyper_dim * self.head_size) ** -0.5
        wei = wei.real
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        wei_complex = wei.to(torch.complex64)

        out_flat = wei_complex @ v_flat  # -> [B, nh, T, hs * hyper_dim]

        # 6. Un-Flatten und Recombine: [B, nh, T, hs * hyper_dim] -> [B, nh, T, hs, hyper_dim]
        out = out_flat.view(B, self.n_head, T, self.head_size, hyper_dim)
        
        # -> [B, T, C, hyper_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C, hyper_dim)

        out = self.out_proj(out)

        return out


   
class HyperBlock(nn.Module):
    """Transformer block: Attention + SAIL with Residuals & RMSNorm"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sa = HyperSelfAttention(config)
        self.sail = SAIL(dim=config.n_embd, level=config.level)
        self.ln1 = HyperRMSNorm(config.n_embd, level=config.level)
        self.ln2 = HyperRMSNorm(config.n_embd, level=config.level)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Residual
        x = x + self.ln1(self.sa(x))
        # FeedForward (SAIL) + Residual
        x = x + self.ln2(self.sail(x))
        return x


class HyperFormer(nn.Module, LMCoreMixin):
    """
    Universal Activation-Free Hypercomplex Transformer (v0.2.0).
    Dynamically scaled via `config.level` (1=Complex, 2=Quaternion, 3=Octonion, 4=Sedenion...).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.level = config.level

        self.token_embedding = HyperEmbedding(
            config.vocab_size, config.n_embd, level=self.level
        )
        self.blocks = nn.ModuleList([HyperBlock(config) for _ in range(config.n_layer)])
        self.ln_f = HyperRMSNorm(config.n_embd, level=self.level)
        self.lm_head = HyperLMHead(config.n_embd, config.vocab_size, level=self.level)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.token_embedding(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
