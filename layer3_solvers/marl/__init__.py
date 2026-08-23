"""MARL — Multi-Agent Reinforcement Learning solvers (QMix / HAPPO / MAAC).

Requires torch. If torch is not installed, importing this module raises
``ImportError`` (re-raised from the optional-import guard, review M-2):
consumers detect availability with ``except ImportError`` instead of
receiving silently-None symbols.
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
    __all__ = []
    raise
