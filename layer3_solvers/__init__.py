"""Layer 3: Solvers — game-playing AI algorithms.

All solvers implement ``SolverBase`` and consume games exclusively
through the ``SolverAdapter`` Protocol from Layer 2.
"""

from .base import SolverBase, SolverConfig
from .mcts.solver import MCTS, MCTSConfig
from .cfr.solver import CFR, CFRConfig
from .psro.solver import PSROSolver, PSROConfig
from .hybrid.solver import HybridSolver, HybridConfig

try:
    from .ppo.solver import PPOSolver, PPOConfig
except ImportError:
    PPOSolver = None
    PPOConfig = None

__all__ = [
    "SolverBase",
    "SolverConfig",
    "MCTS", "MCTSConfig",
    "CFR", "CFRConfig",
    "PSROSolver", "PSROConfig",
    "HybridSolver", "HybridConfig",
]
if PPOSolver is not None:
    __all__.extend(["PPOSolver", "PPOConfig"])
