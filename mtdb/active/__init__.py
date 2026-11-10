"""Active inference / deliberation layer for MTDB.

This package adds a deliberately small but real "thinking" loop on top of the
MTDB base: internal rollouts under candidate observations/actions, scored by
expected free energy, on an active-sensing benchmark. It is not a claim of AGI;
see ACTIVE_INFERENCE.md for the scope and limitations.
"""

from .agent import ActiveSensingAgent
from .sensing import ActiveSensingEnv, SensingEpisode

__all__ = ["ActiveSensingEnv", "SensingEpisode", "ActiveSensingAgent"]
