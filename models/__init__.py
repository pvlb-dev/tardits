from .real import RealGPT
from .complex import InterFormer as ComplexGPT
from .quaternion import HyperFormer as QuatGPT
from .octonion import KrakenFormer as OctoGPT
from .sedenion import HydraFormer as SedenionGPT

__all__ = ["RealGPT, ComplexGPT", "QuatGPT", "OctoGPT, SedenionGPT"]
