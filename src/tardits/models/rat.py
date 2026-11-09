"""RAT - the Recurrent Attention Transformer.

A hypercomplex decoder that replaces softmax attention with a linear
fast-weight associative memory, so training is O(T) and generation runs the
recurrence instead of recomputing the full window:

    S_t   = S_{t-1} (x) gamma + v_t (x) k_bar_t      # decay + ingestion
    out_t = (S_t (x) q_t) . scale                    # recall

`forward` (parallel, used for training) and `step` (recurrent, used for
inference) must stay numerically identical - `tests/test_rat_equivalence.py`
guards that invariant for every supported `level`.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from tardits.modules.core import cd_mul, cd_conjugate, HyperDropout
from tardits.modules import (
    HyperEmbedding,
    HyperLMHead,
    HyperRMSNorm,
    SAIL,
    HyperLinear,
    LMCoreMixin,
)
from tardits.models.base import ModelConfig


def _sample_next(
    logits: torch.Tensor, temp: float = 1.0, top_k: int = 10
) -> torch.Tensor:
    """Sample one token per row from ``[B, vocab]`` logits with temperature + top-k."""
    logits = logits / temp
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float("Inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class Rattention(nn.Module):
    """
    RAT-tention: Hypercomplex Linear Fast-Weight Attention for 2^level spaces.
    Replaces quadratic Softmax-Attention with O(T) associative Fast-Weight Memory.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.level = config.level
        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.hyper_dim = 2 ** (
            config.level - 1
        )  # 1 for complex, 2 for quaternion, etc.
        self.scale = ((2**config.level) * self.head_size) ** -0.5

        # Projections
        self.q_proj = HyperLinear(self.n_embd, self.n_embd, level=self.level)
        self.k_proj = HyperLinear(self.n_embd, self.n_embd, level=self.level)
        self.v_proj = HyperLinear(self.n_embd, self.n_embd, level=self.level)
        self.out_proj = HyperLinear(self.n_embd, self.n_embd, level=self.level)

        # Attention dropout on the sublayer output (complex-safe)
        self.dropout = HyperDropout(config.dropout)

        # Gamma Parameter über die Hyper-Dimension: [1, n_head, 1, head_size, 1, hyper_dim]
        self.gamma = nn.Parameter(
            torch.zeros(
                # (1, self.n_head, 1, self.head_size, 1, self.hyper_dim),
                (1, self.n_head, 1, self.head_size, self.head_size, self.hyper_dim),
                dtype=torch.complex64,
            )
        )
        self.reset_gamma_parameters()

        # self.rope = GyRoPE(
        #     self.head_size, config.block_size, self.n_head, level=self.level
        # )

        # Normierung für Stabilität im Phasenraum
        # self.norm = HyperRMSNorm(self.head_size, level=self.level)

    def reset_gamma_parameters(self):
        with torch.no_grad():
            raw_real = torch.randn_like(self.gamma.real)
            raw_imag = torch.randn_like(self.gamma.imag)
            raw = torch.complex(raw_real, raw_imag)

            # Euklidische Norm über die Hyper-Dimension (dim=-1):
            # Level 1 (2 reelle Dims) -> S^1 (Einheitskreis)
            # Level 2 (4 reelle Dims) -> S^3 (3D-Hypersphäre)
            # Level 3 (8 reelle Dims) -> S^7 (7D-Hypersphäre)
            # Level L (2^L reelle Dims) -> S^(2^L - 1)
            norm = torch.linalg.vector_norm(raw, ord=2, dim=-1, keepdim=True).clamp(
                min=1e-8
            )
            unit_sphere_gamma = raw / norm
            self.gamma.copy_(unit_sphere_gamma)

    def _compute_gamma_powers(self, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.gamma.device
        hyper_dim = self.gamma.shape[-1]

        # 1. Magnitude r:
        r = torch.linalg.vector_norm(self.gamma, dim=-1, keepdim=True).clamp(min=1e-8)
        unit_gamma = self.gamma / r

        # 2. Angle theta from scalar/real part of e_0:
        cos_theta = unit_gamma[..., 0].real.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta).unsqueeze(-1)

        # 3. Unit imaginary part u:
        u_imag0 = unit_gamma[..., 0].imag
        u0 = torch.complex(torch.zeros_like(u_imag0), u_imag0)
        if hyper_dim > 1:
            u_raw = torch.cat([u0.unsqueeze(-1), unit_gamma[..., 1:]], dim=-1)
        else:
            u_raw = u0.unsqueeze(-1)

        u_norm = torch.linalg.vector_norm(u_raw, dim=-1, keepdim=True).clamp(min=1e-8)
        u_unit = u_raw / u_norm

        # 4. Timesteps:
        t_steps = torch.arange(T, device=device, dtype=torch.float32).view(
            1, 1, T, 1, 1, 1
        )
        t_theta = t_steps * theta
        cos_t = torch.cos(t_theta)
        sin_t = torch.sin(t_theta)

        r_t = r**t_steps
        r_t_inv = (1.0 / r.clamp(min=1e-4)) ** t_steps

        g_pow0_real = r_t.squeeze(-1) * cos_t.squeeze(-1)
        g_pow0_imag = (r_t * sin_t).squeeze(-1) * u_unit[..., 0].imag
        g_pow0 = torch.complex(g_pow0_real, g_pow0_imag).unsqueeze(-1)

        g_inv0_real = r_t_inv.squeeze(-1) * cos_t.squeeze(-1)
        g_inv0_imag = -(r_t_inv * sin_t).squeeze(-1) * u_unit[..., 0].imag
        g_inv0 = torch.complex(g_inv0_real, g_inv0_imag).unsqueeze(-1)

        if hyper_dim > 1:
            g_pow_rest = u_unit[..., 1:] * (r_t * sin_t)
            gamma_pow_t = torch.cat([g_pow0, g_pow_rest], dim=-1)

            g_inv_rest = -u_unit[..., 1:] * (r_t_inv * sin_t)
            gamma_inv_t = torch.cat([g_inv0, g_inv_rest], dim=-1)
        else:
            gamma_pow_t = g_pow0
            gamma_inv_t = g_inv0

        return gamma_pow_t, gamma_inv_t

    def _decay_add(self, state: torch.Tensor, kv_outer: torch.Tensor) -> torch.Tensor:
        """Single recurrent fast-weight update: S_t = S_{t-1} (x) gamma + kv_t.

        Shared by ``step`` (inference) and the exact sequential branch of
        ``forward`` (training) so the two paths cannot drift apart.
        """
        gamma_step = self.gamma.squeeze(2)  # [1, nh, hs, hs, hyper_dim]
        return cd_mul(state, gamma_step, level=self.level) + kv_outer

    def _scan_chunk_size(self) -> int:
        """Largest chunk whose local scan stays inside float32.

        The scan multiplies by ``gamma**-t``; trained ``|gamma|`` decays below 1,
        so a long prompt (or a long training window) overflows in float32. Chunk
        at a bound derived from the trained decay and carry the state forward.
        """
        gamma_norm = float(
            torch.linalg.vector_norm(self.gamma.detach(), dim=-1).min()
        )
        gamma_norm = max(gamma_norm, 1e-6)
        max_chunk = int(25.0 / max(math.log10(1.0 / gamma_norm), 1e-6))
        return max(8, min(self.config.block_size, 128, max_chunk))

    def _causal_scan(
        self,
        kv_outer: torch.Tensor,
        force_scan: bool = False,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Causal fast-weight scan: S_t = sum_{s<=t} gamma^{t-s} (x) A_s."""
        B, _, T, _, _, hyper_dim = kv_outer.shape

        if self.level <= 3:
            # Stable de Moivre/prefix-scan path for L1-L3. The old L1 shortcut
            # used ``1/(gamma_t + 1e-8)``, which saturates for trained
            # |gamma|<1 and makes training disagree with ``step`` (#23).
            # ``force_scan`` is kept for API compatibility; all causal levels
            # now use the same numerically stable path.
            gamma_pow_t, gamma_inv_t = self._compute_gamma_powers(T)
            X_scaled = cd_mul(kv_outer, gamma_inv_t, level=self.level)
            cumsum_X = torch.cumsum(X_scaled, dim=2)
            S = cd_mul(cumsum_X, gamma_pow_t, level=self.level)

        else:
            # level >= 4: non-alternative algebra, exact sequential recurrence.
            if initial_state is None:
                S = torch.zeros(
                    B,
                    self.n_head,
                    self.head_size,
                    self.head_size,
                    hyper_dim,
                    dtype=torch.complex64,
                    device=kv_outer.device,
                )
            else:
                S = initial_state
            S_list = []
            for t in range(T):
                S = self._decay_add(S, kv_outer[:, :, t])
                S_list.append(S)
            S = torch.stack(S_list, dim=2)
            gamma_pow_t = None

        if initial_state is not None and gamma_pow_t is not None:
            # Chunked prefill carry: S_t = gamma^{t+1} (x) S_prev + S_local_t.
            gamma_next = cd_mul(gamma_pow_t, self.gamma, level=self.level)
            S = S + cd_mul(
                initial_state.unsqueeze(2), gamma_next, level=self.level
            )
        return S

    def forward(
        self,
        x: torch.Tensor,
        return_state: bool = False,
        initial_state: torch.Tensor | None = None,
        force_scan: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, C, hyper_dim = x.shape

        chunk = self._scan_chunk_size()
        if self.config.bidirectional:
            if initial_state is not None or return_state:
                raise ValueError(
                    "bidirectional RAT does not expose a recurrent state"
                )
            if T > chunk:
                raise ValueError(
                    f"bidirectional RAT supports windows up to {chunk} tokens; "
                    f"got {T}"
                )
        elif T > chunk and self.level <= 3:
            # Long causal sequences: run the scan per chunk and carry the
            # fast-weight state (numerical bound, issue #23).
            outs = []
            state = initial_state
            for start in range(0, T, chunk):
                out, state = self.forward(
                    x[:, start : start + chunk],
                    return_state=True,
                    initial_state=state,
                    force_scan=True,
                )
                outs.append(out)
            out = torch.cat(outs, dim=1)
            return (out, state) if return_state else out

        # 1. Normale Base Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into Heads -> [B, n_head, T, head_size, hyper_dim]
        q = q.view(B, T, self.n_head, self.head_size, hyper_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size, hyper_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size, hyper_dim).transpose(1, 2)

        # Ingestion: Outer Product V x K_conj (associative memory S = V @ K^H)
        k_conj = cd_conjugate(k, level=self.level)
        v_exp = v.unsqueeze(-2)  # [B, nh, T, hs, 1, hyper_dim] -> Value is dim 3 (hs)
        k_exp = k_conj.unsqueeze(
            -3
        )  # [B, nh, T, 1, hs, hyper_dim] -> Key is dim 4 (hs)

        kv_outer = cd_mul(
            v_exp, k_exp, level=self.level
        )  # [B, nh, T, hs, hs, hyper_dim]

        if self.config.bidirectional:
            S = self._causal_scan(kv_outer, force_scan=True)
            S = S + self._causal_scan(kv_outer.flip(2), force_scan=True).flip(2) - kv_outer
        else:
            S = self._causal_scan(
                kv_outer, force_scan=force_scan, initial_state=initial_state
            )

        # Read / Recall: Fast-Weight Query trifft auf Speicher
        # S has shape [B, nh, T, hs_v, hs_k, hyper_dim]
        # Query matches keys at dim 4 (hs_k)
        q_exp = q.unsqueeze(-3)  # [B, nh, T, 1, hs, hyper_dim]

        # S * q_exp: [B, nh, T, hs_v, hs_k, hyper_dim]
        # Sum over key dimension (dim -2, which is index 4) to get recalled values
        out_raw = (
            cd_mul(S, q_exp, level=self.level).sum(dim=-2) * self.scale
        )  # [B, nh, T, hs, hyper_dim]

        # Squeeze is removed to prevent permutation bugs when T=1
        out_head = out_raw

        out = out_head.transpose(1, 2).contiguous().view(B, T, C, hyper_dim)
        out = self.dropout(self.out_proj(out))

        if return_state:
            # S_t is the memory after ingesting position t (causal scan), so the
            # last position is exactly the state the recurrent path would hold.
            return out, S[:, :, -1].clone()
        return out

    def step(
        self, x: torch.Tensor, state: torch.Tensor, pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C, hyper_dim = x.shape
        assert T == 1, "step expects sequence length of 1"

        # 1. Base Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into Heads -> [B, n_head, 1, head_size, hyper_dim]
        q = q.view(B, 1, self.n_head, self.head_size, hyper_dim).transpose(1, 2)
        k = k.view(B, 1, self.n_head, self.head_size, hyper_dim).transpose(1, 2)
        v = v.view(B, 1, self.n_head, self.head_size, hyper_dim).transpose(1, 2)

        # Apply GyRoPE positions at current position index
        # q = self.rope(q, start_pos=pos)
        # k = self.rope(k, start_pos=pos)

        # Squeeze the time dimension for the recurrent update
        q_step = q.squeeze(2)
        k_step = k.squeeze(2)
        v_step = v.squeeze(2)

        k_conj = cd_conjugate(k_step, level=self.level)

        v_exp = v_step.unsqueeze(-2)
        k_exp = k_conj.unsqueeze(-3)
        kv_outer = cd_mul(v_exp, k_exp, level=self.level)

        # Decay previous state and add current outer product
        new_state = self._decay_add(state, kv_outer)

        # Recall: S_t * Q
        q_exp = q_step.unsqueeze(-3)
        out_raw = cd_mul(new_state, q_exp, level=self.level).sum(dim=-2) * self.scale

        # Restore sequence dimension of size 1 and project out
        out_head = out_raw.unsqueeze(2)
        out = out_head.transpose(1, 2).contiguous().view(B, 1, C, hyper_dim)
        out = self.dropout(self.out_proj(out))

        return out, new_state


class RATBlock(nn.Module):
    """Transformer block: Attention + SAIL with Residuals & RMSNorm"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sa = Rattention(config)
        self.sail = SAIL(dim=config.n_embd, level=config.level)
        self.ln1 = HyperRMSNorm(config.n_embd, level=config.level)
        self.ln2 = HyperRMSNorm(config.n_embd, level=config.level)

    def forward(
        self, x: torch.Tensor, return_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_state:
            attn_out, state = self.sa(x, return_state=True)
            x = x + self.ln1(attn_out)
            x = x + self.ln2(self.sail(x))
            return x, state
        # Attention + Residual
        x = x + self.ln1(self.sa(x))
        # FeedForward (SAIL) + Residual
        x = x + self.ln2(self.sail(x))
        return x

    def step(
        self, x: torch.Tensor, state: torch.Tensor, pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Attention + Residual (Post-LN)
        attn_out, new_state = self.sa.step(x, state, pos)
        x = x + self.ln1(attn_out)
        # FeedForward (SAIL) + Residual (Post-LN)
        x = x + self.ln2(self.sail(x))
        return x, new_state


class RAT(nn.Module, LMCoreMixin):
    """
    Recurrent Attention Transformer (RAT).

    Universal activation-free hypercomplex decoder with linear fast-weight
    attention. Dynamically scaled via `config.level`
    (1=Complex, 2=Quaternion, 3=Octonion, 4=Sedenion, ...).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.level = config.level

        self.token_embedding = HyperEmbedding(
            config.vocab_size, config.n_embd, level=self.level
        )
        self.blocks = nn.ModuleList([RATBlock(config) for _ in range(config.n_layer)])
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

    def init_states(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        states = []
        for block in self.blocks:
            state = torch.zeros(
                batch_size,
                block.sa.n_head,
                block.sa.head_size,
                block.sa.head_size,
                block.sa.hyper_dim,
                dtype=torch.complex64,
                device=device,
            )
            states.append(state)
        return states

    def _prefill_chunk_size(self) -> int:
        """Largest chunk whose scan stays numerically safe for this checkpoint.

        The scan multiplies by ``gamma**-t``; trained ``|gamma|`` can decay well
        below 1, so a single scan over a very long prompt overflows float32
        (observed NaNs at 256 tokens for RAT L2). Chunking keeps the exponent
        near the trained window and carries the state between chunks.
        """
        gamma_norm = min(
            float(torch.linalg.vector_norm(block.sa.gamma.detach(), dim=-1).min())
            for block in self.blocks
        )
        gamma_norm = max(gamma_norm, 1e-6)
        # Keep r**-chunk below ~1e25 (float32 max is ~3e38).
        max_chunk = int(25.0 / max(math.log10(1.0 / gamma_norm), 1e-6))
        return max(1, min(self.config.block_size, 128, max_chunk))

    @torch.no_grad()
    def prefill(
        self, idx: torch.Tensor
    ) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
        if self.config.bidirectional:
            raise ValueError("bidirectional RAT is a denoiser; use forward()")

        """Vectorised prompt ingestion through the chunked scan.

        Returns ``(logits, states)``: logits for every prompt position (the last
        one samples the next token) and each block's fast-weight memory after
        reading the whole prompt. Each chunk runs the parallel scan; the memory
        is carried to the next chunk, so the result matches ``_warm_states``
        while avoiding one Python step per token.
        """
        device = next(self.parameters()).device
        if idx.shape[1] == 0:
            return None, self.init_states(idx.shape[0], device)

        chunk = self._prefill_chunk_size()
        x = self.token_embedding(idx)
        states = []
        for block in self.blocks:
            sa = block.sa
            state = None
            outs = []
            for start in range(0, idx.shape[1], chunk):
                out, state = sa(
                    x[:, start : start + chunk],
                    return_state=True,
                    initial_state=state,
                    force_scan=True,
                )
                outs.append(out)
            attn = torch.cat(outs, dim=1)
            x = x + block.ln1(attn)
            x = x + block.ln2(block.sail(x))
            states.append(state)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, states

    def step(
        self, idx: torch.Tensor, states: list[torch.Tensor], pos: int
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.config.bidirectional:
            raise ValueError("bidirectional RAT is a denoiser; use forward()")
        x = self.token_embedding(idx)
        new_states = []
        for block, state in zip(self.blocks, states):
            x, new_state = block.step(x, state, pos)
            new_states.append(new_state)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_states

    def _warm_states(
        self, idx: torch.Tensor, states: list[torch.Tensor]
    ) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
        """Ingest the prompt token by token, returning logits after the last one.

        ``logits`` is ``None`` only when ``idx`` is empty (streaming then needs
        at least one token, because the next token has to come from somewhere).
        """
        logits = None
        for pos in range(idx.shape[1]):
            logits, states = self.step(idx[:, pos : pos + 1], states, pos)
        return logits, states

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temp: float = 1.0, top_k: int = 10
    ) -> torch.Tensor:
        if self.config.bidirectional:
            raise ValueError("bidirectional RAT is a denoiser; use forward()")

        device = next(self.parameters()).device
        idx = idx.to(device)
        B, T = idx.shape

        # Vectorised prefill through the scan; the final prompt token is then
        # stepped so sampling follows the same path as streamed generation.
        _, states = self.prefill(idx[:, : T - 1])

        curr_token = idx[:, -1:]

        for i in range(max_new_tokens):
            pos = T - 1 + i
            logits, states = self.step(curr_token, states, pos)
            next_token = _sample_next(logits[:, -1, :], temp, top_k)

            idx = torch.cat((idx, next_token), dim=1)
            curr_token = next_token

        return idx

    @torch.no_grad()
    def generate_stream(self, idx: torch.Tensor, temp: float = 1.0, top_k: int = 10):
        if self.config.bidirectional:
            raise ValueError("bidirectional RAT is a denoiser; use forward()")
        """Endless token stream on the recurrent path (O(1) work per token)."""
        device = next(self.parameters()).device
        idx = idx.to(device)
        B = idx.shape[0]

        logits, states = self.prefill(idx)
        if logits is None:
            raise ValueError("RAT generation needs a non-empty prompt")

        pos = idx.shape[1]
        while True:
            next_token = _sample_next(logits[:, -1, :], temp, top_k)
            yield next_token.item()
            logits, states = self.step(next_token, states, pos)
            pos += 1
