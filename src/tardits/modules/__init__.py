from .trainer import Trainer, TrainerConfig
from .base import get_device, LMCoreMixin
from .logging import MLFlowConfig
from .core import (
    SAIL,
    GyRoPE,
    HyperRMSNorm,
    HyperEmbedding,
    HyperLMHead,
    HyperLinear,
    HyperDropout,
)

# NOTE: ModelConfig lives in `tardits.models.base` (it is shared by all models and
# is intentionally not re-exported here, so `modules` does not depend on `models`).

__all__ = [
    "Trainer",
    "TrainerConfig",
    "MLFlowConfig",
    "SAIL",
    "GyRoPE",
    "LMCoreMixin",
    "HyperRMSNorm",
    "HyperEmbedding",
    "HyperLMHead",
    "HyperLinear",
    "HyperDropout",
    "get_device",
]
