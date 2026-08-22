"""Layer 3: Solvers — game-playing AI algorithms.

All solvers implement ``SolverBase`` and consume games exclusively
through the ``SolverAdapter`` Protocol from Layer 2.
"""

from .base import SolverBase, SolverConfig
from .cfr.solver import CFR, CFRConfig
from .hybrid.solver import HybridConfig, HybridSolver
from .llm.ollama_solver import OllamaConfig, OllamaSolver
from .mcts.solver import MCTS, MCTSConfig
from .werewolf import BayesConfig, BayesSolver

try:
    # PSRO 依赖 gymnasium/tqdm（psro extra，审查 P2-25）— 缺失时降级，
    # 与 PPO/MARL 的守卫一致，保证最小安装可导入。
    from .psro.solver import PSROConfig, PSROSolver
except ImportError:
    PSROSolver = None
    PSROConfig = None

try:
    from .ppo.solver import PPOConfig, PPOSolver
except ImportError:
    PPOSolver = None
    PPOConfig = None

try:
    from .marl.happo import HAPPOConfig, HAPPOSolver
    from .marl.maac import MAACConfig, MAACSolver
    from .marl.qmix import QMixConfig, QMixSolver
except ImportError:
    QMixSolver = HAPPOSolver = MAACSolver = None
    QMixConfig = HAPPOConfig = MAACConfig = None

__all__ = [
    "SolverBase",
    "SolverConfig",
    "MCTS",
    "MCTSConfig",
    "CFR",
    "CFRConfig",
    "PSROSolver",
    "PSROConfig",
    "HybridSolver",
    "HybridConfig",
    "OllamaSolver",
    "OllamaConfig",
    "BayesSolver",
    "BayesConfig",
]
if PPOSolver is not None:
    __all__.extend(["PPOSolver", "PPOConfig"])
if QMixSolver is not None:
    __all__.extend(["QMixSolver", "QMixConfig", "HAPPOSolver", "HAPPOConfig", "MAACSolver", "MAACConfig"])
if PSROSolver is not None:
    __all__.extend(["PSROSolver", "PSROConfig"])
