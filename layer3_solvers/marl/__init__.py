"""MARL — Multi-Agent Reinforcement Learning solvers (QMix / HAPPO / MAAC).

Requires torch. If torch is not installed, importing this module will
fail gracefully (the solvers will not be available).
"""

try:
    from .action_space import ActionSpace
    from .env import resolve_players, run_episode
    from .happo import HAPPOConfig, HAPPOSolver
    from .maac import MAACConfig, MAACSolver
    from .qmix import QMixConfig, QMixSolver

    __all__ = [
        "QMixSolver",
        "QMixConfig",
        "HAPPOSolver",
        "HAPPOConfig",
        "MAACSolver",
        "MAACConfig",
        "ActionSpace",
        "run_episode",
        "resolve_players",
    ]
except ImportError:
    QMixSolver = HAPPOSolver = MAACSolver = None  # type: ignore
    QMixConfig = HAPPOConfig = MAACConfig = None  # type: ignore
    __all__ = []
