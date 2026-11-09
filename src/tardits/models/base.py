from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Shared configuration for all models (RealGPT, HyperFormer, RAT)."""

    level: int = 1
    block_size: int = 8  # maximum context length
    vocab_size: int = 74
    n_head: int = 4
    n_embd: int = 32
    n_layer: int = 3
    dropout: float = 0.2
    # Residual/norm layout for the RealGPT block: "branch" is the project's
    # deliberate x + ln(sublayer(x)) variant; "pre" is canonical x + sublayer(ln(x)).
    norm: str = "branch"
    # RAT only: symmetric decay kernel for non-causal (diffusion/denoiser)
    # training. Generation and recurrent state are disabled in this mode.
    bidirectional: bool = False

    def __post_init__(self):
        """Fail early on shapes that would otherwise break deep in a view()."""
        if self.level < 0:
            raise ValueError(f"level must be >= 0, got {self.level}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.n_layer}")
        if self.n_embd <= 0 or self.n_head <= 0:
            raise ValueError(
                f"n_embd and n_head must be positive, got "
                f"n_embd={self.n_embd}, n_head={self.n_head}"
            )
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}); "
                "otherwise head_size is not an integer"
            )
        if self.norm not in {"branch", "pre"}:
            raise ValueError(
                f"norm must be 'branch' or 'pre', got {self.norm!r}"
            )
