"""PPO — Proximal Policy Optimization solver.

Requires torch. If torch is not installed, importing this module raises
``ImportError`` (re-raised from the optional-import guard, review M-2):
consumers detect availability with ``except ImportError`` instead of
receiving silently-None symbols.
"""

try:
    from .networks import ActorCriticNetwork
    from .rollout_buffer import RolloutBatch, RolloutBuffer
    from .solver import PPOConfig, PPOSolver

    __all__ = ["PPOSolver", "PPOConfig", "ActorCriticNetwork", "RolloutBuffer", "RolloutBatch"]
except ImportError:
    __all__ = []
    raise
