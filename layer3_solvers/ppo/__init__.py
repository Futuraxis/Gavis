"""PPO — Proximal Policy Optimization solver.

Requires torch. If torch is not installed, importing this module
will fail gracefully (PPOSolver will not be available).
"""

try:
    from .networks import ActorCriticNetwork
    from .rollout_buffer import RolloutBatch, RolloutBuffer
    from .solver import PPOConfig, PPOSolver

    __all__ = ["PPOSolver", "PPOConfig", "ActorCriticNetwork", "RolloutBuffer", "RolloutBatch"]
except ImportError:
    PPOSolver = None  # type: ignore
    PPOConfig = None  # type: ignore
    __all__ = []
