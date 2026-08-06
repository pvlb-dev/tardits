import math
import torch
import torch.nn as nn


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

        # Cayley-Dickson for Level >= 2
        x_expanded = x.unsqueeze(-3)
        w_expanded = self.weight

        prod = cd_mul(x_expanded, w_expanded, level=self.level)
        out = prod.sum(dim=-2)

        if self.bias is not None:
            out = out + self.bias

        return out


class HyperRMSNorm(nn.Module):
    """Norms amplitudes of complex hyper-vectors."""

    def __init__(self, dim: int, level: int = 1, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.level = level

        self.weight = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x Shape: [..., dim, hyper_dim] (complex)
        ampsq = (x.abs() ** 2).sum(dim=-1)  # Shape: [..., dim]

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: [Batch, n_head, T, head_size, hyper_dim] (complex)
        B, nh, T, hs, _ = x.shape
        t_current = self.t[:T]

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

        # pack pairs to torch.complex64:
        r_real = real_components[..., : self.hyper_dim]
        r_imag = real_components[..., self.hyper_dim :]
        rotor = torch.complex(r_real, r_imag)  # [1, nh, T, hs, hyper_dim]

        return cd_mul(x, rotor, level=self.level)
