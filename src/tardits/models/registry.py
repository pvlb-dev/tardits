"""Model registry - resolve a model class from a config ``arch`` string.

The YAML configs select a model family with an optional ``arch`` key in the
``model`` section:

    model:
      arch: rat      # real | hyper | rat  (default: level < 1 -> real, else hyper)
      level: 1
      ...

`arch` is *not* part of :class:`ModelConfig` (it selects the class that consumes
the config), so the CLIs pop it out of the model dict before building the
config. Keeping the mapping here - instead of hardcoded `if level < 1` branches
in every entry point - means new models only have to be registered once.
"""

import torch.nn as nn

from tardits.models.base import ModelConfig
from tardits.models.hyper import HyperFormer
from tardits.models.rat import RAT
from tardits.models.real import RealGPT

# Canonical name -> class. These are the values accepted by `arch:` / `--arch`.
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "real": RealGPT,
    "hyper": HyperFormer,
    "rat": RAT,
}

# Convenience spellings that map onto a canonical name.
ARCH_ALIASES: dict[str, str] = {
    "gpt": "real",
    "realgpt": "real",
    "hyperformer": "hyper",
    "lina": "rat",  # LinA: the pre-rename name of RAT
}


def available_archs() -> list[str]:
    """Canonical model names, e.g. for `--help` output."""
    return sorted(MODEL_REGISTRY)


def resolve_arch(arch: str | None, level: int = 1) -> str:
    """Normalize ``arch`` to a canonical name, applying the legacy default.

    Without an explicit ``arch`` the historical rule is kept so old configs
    train the same model they always did: ``level < 1`` selects the real
    baseline, anything else the hypercomplex softmax transformer.
    """
    if arch is None or arch == "":
        return "real" if level < 1 else "hyper"

    key = str(arch).strip().lower().replace("-", "").replace("_", "")
    name = ARCH_ALIASES.get(key, key)
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model arch {arch!r} - expected one of: "
            f"{', '.join(available_archs())}"
        )

    if name == "rat" and level < 1:
        raise ValueError(
            "RAT is a hypercomplex model - set model.level >= 1 "
            "(use arch: real for the real-valued baseline)"
        )
    return name


def get_model_class(arch: str | None = None, level: int = 1) -> type[nn.Module]:
    """Return the model class for ``arch`` (resolved with the legacy default)."""
    return MODEL_REGISTRY[resolve_arch(arch, level)]


def build_model(arch: str | None, config: ModelConfig) -> nn.Module:
    """Instantiate the model selected by ``arch`` from ``config``."""
    return get_model_class(arch, config.level)(config)
