import functools
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


def cd_conjugate(x: torch.Tensor, level: int) -> torch.Tensor:
    """
    Recursive Cayley-Dickson conjugation operating on complex tensors.
    Level 1 = Complex (hyper_dim = 1, torch.conj)
    Level 2 = Quaternion (hyper_dim = 2)
    """
    if level < 1:
        raise ValueError(
            f"Invalid level={level}. Requires level >= 1 "
            f"(1=Complex, 2=Quaternion, 3=Octonion, 4=Sedenion, etc.)."
        )
    if level == 1:
        return torch.conj(x)

    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([cd_conjugate(x1, level - 1), -x2], dim=-1)


def cd_mul(a: torch.Tensor, b: torch.Tensor, level: int) -> torch.Tensor:
    """
    Universal recursive Cayley-Dickson multiplication over complex numbers.
    Formula: (a1, a2) * (b1, b2) = (a1*b1 - b2_conj*a2, b2*a1 + a2*b1_conj)
    """
    if level < 1:
        raise ValueError(f"Invalid level={level}. Requires level >= 1.")
    if level == 1:
        return a * b  # Native PyTorch complex matrix multiplication / elementwise prod

    a1, a2 = torch.chunk(a, 2, dim=-1)
    b1, b2 = torch.chunk(b, 2, dim=-1)

    b2_conj = cd_conjugate(b2, level - 1)
    b1_conj = cd_conjugate(b1, level - 1)

    res_left = cd_mul(a1, b1, level - 1) - cd_mul(b2_conj, a2, level - 1)
    res_right = cd_mul(b2, a1, level - 1) + cd_mul(a2, b1_conj, level - 1)

    return torch.cat([res_left, res_right], dim=-1)


@functools.lru_cache(maxsize=None)
def _cd_multiplication_table(level: int) -> torch.Tensor:
    """Dense real Cayley-Dickson multiplication table.

    Returns ``T`` of shape ``[n, n, n]`` (``n = 2**level`` real dims) such that
    the product ``c = a (x) b`` satisfies ``c_r = sum_{p,q} T[p, q, r] a_p b_q``.

    Real-dim ordering: dim ``2k`` is ``component[k].real`` and dim ``2k + 1`` is
    ``component[k].imag``. Built from the existing ``cd_mul`` on one-hot basis
    vectors so the table is always consistent with the recursive implementation.
    """
    n = 2 ** level
    hyper_dim = 2 ** (level - 1)
    table = torch.zeros(n, n, n)

    for i in range(n):
        a = torch.zeros(hyper_dim, dtype=torch.complex64)
        a[i // 2] = 1j if i % 2 else (1.0 + 0.0j)
        for j in range(n):
            b = torch.zeros(hyper_dim, dtype=torch.complex64)
            b[j // 2] = 1j if j % 2 else (1.0 + 0.0j)
            c = cd_mul(a, b, level)
            table[i, j, 0::2] = c.real
            table[i, j, 1::2] = c.imag

    return table


class HyperLinear(nn.Module):
    """Universal hypercomplex Linear-Layer for arbitrary dimensioning (complex-native)."""

    def __init__(
        self, in_features: int, out_features: int, level: int = 1, bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.level = level
        self.hyper_dim = 2 ** (level - 1)

        # FAST-PATH: Level 1 ist pure complex multiplication
        if self.level == 1:
            self.linear = nn.Linear(
                in_features, out_features, bias=bias, dtype=torch.complex64
            )
            self._reset_parameters_level1()
        else:
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(out_features, self.hyper_dim, dtype=torch.complex64)
                )
            else:
                self.register_parameter("bias", None)

            self.weight = nn.Parameter(
                torch.empty(
                    out_features, in_features, self.hyper_dim, dtype=torch.complex64
                )
            )
            self._reset_parameters()
            self._cd_table = _cd_multiplication_table(level)

    def _reset_parameters_level1(self):
        with torch.no_grad():
            phases = torch.empty_like(self.linear.weight.real).uniform_(
                -2 * math.pi, 2 * math.pi
            )
            scale = math.sqrt(1.0 / (self.in_features + self.out_features))
            w_comp = torch.complex(torch.cos(phases) * scale, torch.sin(phases) * scale)
            self.linear.weight.copy_(w_comp)

    def _reset_parameters(self):
        with torch.no_grad():
            r_real = torch.randn(
                self.out_features,
                self.in_features,
                self.hyper_dim,
                device=self.weight.device,
            )
            r_imag = torch.randn(
                self.out_features,
                self.in_features,
                self.hyper_dim,
                device=self.weight.device,
            )
            r = torch.complex(r_real, r_imag)

            norm = torch.linalg.vector_norm(r, ord=2, dim=-1, keepdim=True).clamp(
                min=1e-8
            )
            r_normalized = r / norm

            scale_numerator = 2.0 ** (1.0 - self.level)
            scale = math.sqrt(scale_numerator / (self.in_features + self.out_features))

            self.weight.copy_(r_normalized * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fast lane for Level 1:
        if self.level == 1:
            # x Shape: [..., in_features, 1] -> squeeze -> nn.Linear -> unsqueeze
            x_sq = x.squeeze(-1)
            out = self.linear(x_sq)
            return out.unsqueeze(-1)

        # Cayley-Dickson for Level >= 2 as a single real BLAS matmul.
        # The CD product is bilinear, so the hypercomplex weight can be spread
        # into a real [out * n, in * n] matrix once and applied via F.linear.
        n = self._cd_table.shape[0]
        if self._cd_table.device != self.weight.device:
            self._cd_table = self._cd_table.to(self.weight.device)

        # weight real components: [out, in, n]
        wr = torch.stack([self.weight.real, self.weight.imag], dim=-1).flatten(-2)

        # W_mat[out * r, in * p] = sum_q wr[out, in, q] * T[p, q, r]
        # (matches cd_mul(x_expanded, w_expanded): input x on the left, weight on
        # the right, summed over the input feature axis.)
        g = torch.einsum("oiq,pqr->oipr", wr, self._cd_table)  # [out, in, p, r]
        w_mat = g.permute(0, 3, 1, 2).reshape(
            self.out_features * n, self.in_features * n
        )

        # input real components: [..., in * n]
        xr = torch.stack([x.real, x.imag], dim=-1).flatten(-2).flatten(start_dim=-2)

        yr = F.linear(xr, w_mat)  # [..., out * n]
        yr = yr.view(*xr.shape[:-1], self.out_features, n)

        out = torch.complex(yr[..., 0::2], yr[..., 1::2])

        if self.bias is not None:
            out = out + self.bias

        return out


class HyperDropout(nn.Module):
    """Dropout that also supports complex inputs (``nn.Dropout`` does not).

    ``nn.Dropout`` raises ``NotImplementedError`` for complex tensors
    (``bernoulli_scalar_cpu_`` is not implemented for ``ComplexFloat``), so this
    module builds a single real mask that is shared across the real and imaginary
    parts. Whole hypercomplex entries are therefore dropped together, with the
    usual scaled-dropout semantics (survivors scaled by ``1 / (1 - p)``).
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return torch.zeros_like(x)
        # `x.real` is the tensor itself for real inputs, so this works for both.
        mask = (torch.rand_like(x.real) >= self.p).to(x.real.dtype) / (1.0 - self.p)
        return x * mask


class HyperRMSNorm(nn.Module):
    """Norms amplitudes of complex hyper-vectors."""

    def __init__(self, dim: int, level: int = 1, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.level = level

        self.weight = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x Shape: [..., dim, hyper_dim] (complex; real is also tolerated)
        # |z|^2 computed directly for complex input (avoids the sqrt-then-square
        # round trip of x.abs() ** 2; equivalent, slightly more accurate, ~2x
        # faster). `.imag` is not defined for real tensors, so branch on dtype.
        if torch.is_complex(x):
            ampsq = (x.real**2 + x.imag**2).sum(dim=-1)  # Shape: [..., dim]
        else:
            ampsq = (x**2).sum(dim=-1)

        pairs = 2 ** (self.level - 1)
        ampsq = ampsq / pairs

        rms = torch.rsqrt(ampsq.mean(dim=-1, keepdim=True) + self.eps)
        rms = rms.unsqueeze(-1)  # [..., 1, 1]

        return (x * rms) * self.weight


class SAIL(nn.Module):
    """Self-Activating Interference Layer"""

    def __init__(self, dim: int, level: int = 1, hidden_mult: int = 1):
        super().__init__()
        self.dim = dim
        self.level = level
        hidden_dim = dim * hidden_mult

        self.w1 = HyperLinear(dim, hidden_dim, level=level)
        self.w2 = HyperLinear(dim, hidden_dim, level=level)
        self.w3 = HyperLinear(hidden_dim, dim, level=level)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.w1(x)
        h2 = self.w2(x)
        interfered = cd_mul(h1, h2, self.level)
        out = self.w3(interfered)
        return out


class HyperEmbedding(nn.Module):
    """Embeds tokens into a 2^(level-1) complex hypercomplex space."""

    def __init__(self, vocab_size: int, d_model: int, level: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.level = level
        self.hyper_dim = 2 ** (level - 1)

        self.weight = nn.Parameter(
            torch.empty(vocab_size, d_model, self.hyper_dim, dtype=torch.complex64)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            std = 1.0 / math.sqrt(2**self.level)
            real = torch.randn_like(self.weight.real) * std
            imag = torch.randn_like(self.weight.imag) * std
            self.weight.copy_(torch.complex(real, imag))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class HyperLMHead(nn.Module):
    """Projects complex hypercomplex features back to real vocabulary logits."""

    def __init__(self, d_model: int, vocab_size: int, level: int = 1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.level = level
        self.hyper_dim = 2 ** (level - 1)

        # Nimmt Real- und Imaginärteile aller komplexen Komponenten
        self.lin = nn.Linear(d_model * self.hyper_dim * 2, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x Shape: [batch, seq_len, d_model, hyper_dim] (complex)
        # get real, touch gras
        x_real_imag = torch.cat([x.real, x.imag], dim=-1)
        flattened = x_real_imag.flatten(start_dim=-2)
        return self.lin(flattened)


class GyRoPE(nn.Module):
    """GyRoPE: Universal Hypercomplex Rotary Position Embedding."""

    def __init__(self, head_size: int, block_size: int, n_head: int, level: int = 1):
        super().__init__()
        self.head_size = head_size
        self.n_head = n_head
        self.level = level
        self.hyper_dim = 2 ** (level - 1)

        self.real_imag_dim = (2**level) - 1

        self.theta = nn.Parameter(torch.empty(n_head, head_size))
        self.axis_params = nn.Parameter(
            torch.empty(n_head, head_size, self.real_imag_dim)
        )
        self._reset_parameters()

        t = torch.arange(block_size, dtype=torch.float32)
        self.register_buffer("t", t)
        self.t: torch.Tensor

    def _reset_parameters(self):
        with torch.no_grad():
            self.theta.uniform_(-2 * math.pi, 2 * math.pi)

            axis = torch.randn(self.n_head, self.head_size, self.real_imag_dim)
            axis_norm = torch.linalg.vector_norm(
                axis, ord=2, dim=-1, keepdim=True
            ).clamp(min=1e-8)
            self.axis_params.copy_(axis / axis_norm)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # Shape: [Batch, n_head, T, head_size, hyper_dim] (complex)
        B, nh, T, hs, _ = x.shape
        t_current = torch.arange(
            start_pos, start_pos + T, dtype=torch.float32, device=x.device
        )

        axis = self.axis_params
        axis_norm = torch.linalg.vector_norm(axis, ord=2, dim=-1, keepdim=True).clamp(
            min=1e-8
        )
        u = axis / axis_norm  # [nh, hs, real_imag_dim]

        phi = torch.einsum("nh,t->nth", self.theta, t_current)
        r_phi = phi.unsqueeze(0).unsqueeze(-1)
        cos_p = torch.cos(r_phi)
        sin_p = torch.sin(r_phi)

        u_expanded = u.unsqueeze(0).unsqueeze(2)
        imag_part = sin_p * u_expanded

        real_components = torch.cat(
            [cos_p, imag_part], dim=-1
        )  # [1, nh, T, hs, 2^level]

        # Pack pairs to torch.complex64. The real-dimensional ordering is
        # [component0.real, component0.imag, component1.real, component1.imag,
        # ...], so the real/imag parts of each complex component interleave.
        # Slicing first-half/second-half here would permute the rotation axes
        # for level >= 2 (e.g. a quaternion i-rotation would land on j).
        r_real = real_components[..., 0::2]
        r_imag = real_components[..., 1::2]
        rotor = torch.complex(r_real, r_imag)  # [1, nh, T, hs, hyper_dim]

        return cd_mul(x, rotor, level=self.level)
