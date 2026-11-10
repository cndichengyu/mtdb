"""MTDB: Multi-Timescale Dynamical Base — a non-Transformer training substrate.

This package implements the bottom layer of a hybrid AGI architecture:
multi-timescale sparse dynamics, phase synchronization routing, working and
attractor memory, three-factor local plasticity, sleep replay consolidation,
and a compressed interface protocol for upper-layer world models.
"""

from .config import ExperimentConfig

__version__ = "0.3.0"
__all__ = ["ExperimentConfig", "__version__"]
