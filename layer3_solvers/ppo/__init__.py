"""PPO — Proximal Policy Optimization solver.

Requires torch. If torch is not installed, importing this module
will fail gracefully (PPOSolver will not be available).
"""
try:
    from .solver import PPOSolver
    from .networks import ActorCriticNetwork
    from .rollout_buffer import RolloutBuffer, RolloutBatch
    __all__ = ["PPOSolver", "ActorCriticNetwork", "RolloutBuffer", "RolloutBatch"]
except ImportError:
    PPOSolver = None  # type: ignore
    __all__ = []
