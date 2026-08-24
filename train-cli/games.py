"""Gavis 游戏注册表 — 训练与运行时装配所需的一切信息以配置形式集中声明.

**游戏注册制**：每个游戏在 ``GAMES`` 中登记一个 ``GameSpec`` 条目，声明引擎
构造（rules 文件 + 变种/人数，v5.2 variants 声明式）、座位、可训练求解器管线
及各自默认超参/局数/评估设置、以及运行时可用求解器与配置覆盖。训练脚本
(``train.py``) 只读这张表，**不含任何 per-game 分支**——新游戏接入 = 新增一个
登记条目，无需修改任何训练/装配逻辑。

训练部分（``solvers`` / ``SolverPipeline``）：

- ``entry``   "train" → ``solver.train(episodes)``；"solve" → ``solver.solve(initial)``
- ``episodes`` train() 局数（solve 模式忽略；PSRO 的 episodes=迭代数）
- ``config``  传给 config-class 的 kwargs；路径值可用 ``$OUTDIR`` 占位符
  （由 train.py 展开为每游戏输出目录）
- ``save``    产物文件名（None → 不额外保存）
- ``per_player`` True → 每个座位建一个实例（player_id=座位，如贝叶斯狼人杀）

运行时部分（frontend/benchmark 装配用）：

- ``runtime_solvers`` 该游戏运行时可用的求解器名（不含 = 不可用，数据驱动）
- ``runtime_configs`` 每求解器的运行时配置覆盖（如德州 Hybrid 开不完全信息）
- ``create_solver(game_id, name, engine, seed, budget, **kwargs)`` 通用工厂：
  查表校验 → 合并默认/覆盖 → 按名实例化，零 if-分支。

``SOLVER_FACTORY``（训练）与 ``RUNTIME_FACTORY``（运行时）都是**按求解器**的
注册，不是按游戏——可选依赖（torch/psro）缺失时相应 class 为 ``None``，
实例化时给出明确报错。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from layer2_engine.core.engine import GameEngine
from layer3_solvers import (
    CFR,
    MCTS,
    BayesConfig,
    BayesSolver,
    CFRConfig,
    HAPPOConfig,
    HAPPOSolver,
    HybridConfig,
    HybridSolver,
    MAACConfig,
    MAACSolver,
    MCTSConfig,
    OllamaConfig,
    OllamaSolver,
    PPOConfig,
    PPOSolver,
    PSROConfig,
    PSROSolver,
    QMixConfig,
    QMixSolver,
    SolverConfig,
)
from layer3_solvers.base import SolverBase, SolverMetrics
from layer3_solvers.mahjong.heuristic import MahjongHeuristicAI

# ── 求解器工厂（训练）──────────────────────────────────────────────
# name → (solver_class, config_class)；可选依赖缺失时 class 为 None
# （实例化时由 train.py 报错，注册本身不失败）。


def _optional(cls: Any, cfg_cls: Any) -> tuple[Any, Any]:
    """包装可选依赖求解器；导入失败时两者均为 None。"""
    return (cls, cfg_cls)


SOLVER_FACTORY: dict[str, tuple[Any, Any]] = {
    "hybrid": (HybridSolver, HybridConfig),
    "cfr": (CFR, CFRConfig),
    "ppo": _optional(PPOSolver, PPOConfig),
    "psro": _optional(PSROSolver, PSROConfig),
    "qmix": _optional(QMixSolver, QMixConfig),
    "happo": _optional(HAPPOSolver, HAPPOConfig),
    "maac": _optional(MAACSolver, MAACConfig),
    "bayes": (BayesSolver, BayesConfig),
}


# ── 求解器工厂（运行时）────────────────────────────────────────────
# name → callable(engine, cfg_dict) → SolverBase；按求解器注册，非按游戏。


def _make_random(engine: GameEngine, cfg: dict[str, Any]) -> SolverBase:
    return RandomSolver(engine, cfg.get("seed"))


def _make_mahjong(engine: GameEngine, cfg: dict[str, Any]) -> SolverBase:
    return MahjongHeuristicAI(engine, SolverConfig(seed=cfg.get("seed")))


def _make_ollama(engine: GameEngine, cfg: dict[str, Any]) -> SolverBase:
    return OllamaSolver(engine, OllamaConfig(model=cfg.get("model", "qwen3:8b")), player_id=cfg.get("player_id"))


def _make_mcts(engine: GameEngine, cfg: dict[str, Any]) -> SolverBase:
    return MCTS(engine, MCTSConfig(**cfg))


def _make_cfr(engine: GameEngine, cfg: dict[str, Any]) -> SolverBase:
    return CFR(engine, CFRConfig(**cfg))


def _make_hybrid(engine: GameEngine, cfg: dict[str, Any]) -> SolverBase:
    return HybridSolver(engine, HybridConfig(**cfg))


RUNTIME_FACTORY: dict[str, Callable[[GameEngine, dict[str, Any]], SolverBase]] = {
    "mcts": _make_mcts,
    "cfr": _make_cfr,
    "hybrid": _make_hybrid,
    "random": _make_random,
    "mahjong": _make_mahjong,
    "ollama": _make_ollama,
}

#: 运行时求解器默认配置（按求解器；游戏级覆盖见 GameSpec.runtime_configs）。
RUNTIME_DEFAULTS: dict[str, Mapping[str, Any]] = {
    "mcts": {"budget": 3000},
    "cfr": {"iterations": 1000, "depth_limit": 8},
    "hybrid": {"mode": "search", "cfr_iterations": 1000, "cfr_depth_limit": 8, "mcts_budget": 3000},
    "random": {},
    "mahjong": {},
    "ollama": {"model": "qwen3:8b"},
}

#: 调用期 budget 参数注入的配置字段名（按求解器）。
_BUDGET_FIELD: Mapping[str, str] = {"mcts": "budget", "hybrid": "mcts_budget"}


@dataclass(frozen=True)
class EngineSpec:
    """引擎构造配置（v5.2：变种/人数由规则 JSON 声明，纯数据选择）。"""

    rules: str  # rules/<rules>.json
    variant: str | None = None
    player_count: int | None = None


@dataclass(frozen=True)
class SolverPipeline:
    """一个求解器在该游戏上的训练管线（全部配置驱动）。"""

    solver: str  # SOLVER_FACTORY 中的名字
    entry: str = "train"  # "train" → solver.train(episodes)；"solve" → solver.solve(initial)
    episodes: int = 100  # train() 局数（solve 模式忽略；PSRO 的 episodes=迭代数）
    config: Mapping[str, Any] = field(default_factory=dict)  # 传给 config-class 的 kwargs
    save: str | None = None  # 产物文件名（None → 不额外保存）
    per_player: bool = False  # True → 每座位建一个实例（player_id=座位）
    eval: bool = True  # 训练后是否跑 vs 均匀随机评估


@dataclass(frozen=True)
class GameSpec:
    """一个已登记游戏的完整配置（训练管线 + 运行时装配信息）。"""

    game_id: str
    display_name: str
    engine: EngineSpec
    players: tuple[str, ...]  # 座位顺序（先手在前），评估轮换使用
    solvers: Mapping[str, SolverPipeline]  # 可训练求解器管线
    eval_episodes: int = 20  # 训练后每个评估对手的默认局数
    eval_opponents: tuple[str, ...] = ("random",)  # 评估对手：random|self|mcts
    #   random → 均匀随机；self → 自己镜像（自博弈）；mcts → MCTS 基线
    #   （mcts 未在 runtime_solvers 登记时该列自动跳过）。
    runtime_solvers: tuple[str, ...] = ()  # 运行时可用求解器（数据驱动装配）
    runtime_configs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # 运行时配置覆盖


# ── 注册表 ─────────────────────────────────────────────────────────
# 默认超参/局数沿用既有实测标定（train_hybrid / train_marl / benchmark_all）：
#   moon_chess:        CFR 800 iters ≈ 3 min；PSRO 5×2000 ≈ 35 s；MARL 600 局 ≈ 10 s
#   stochastic_gomoku: CFR 50 iters ≈ 5-8 min（9×9 深限）
#   texas_holdem:      CFR 1000 iters ≈ 75 s；MARL 400 局 ≈ 30 s
#   mahjong_*:         MARL 200 局 ≈ 12 min（约 3.5 s/局）；评估建议 4-8 局

_MAHJONG_SOLVERS: Mapping[str, SolverPipeline] = {
    "qmix": SolverPipeline("qmix", episodes=200, save="qmix.pt"),
    "happo": SolverPipeline("happo", episodes=200, save="happo.pt"),
    "maac": SolverPipeline("maac", episodes=200, save="maac.pt"),
}


def _mahjong_spec(game_id: str, display_name: str, variant: str) -> GameSpec:
    """构造麻将变种登记条目（同一 rules 文件 + variants 声明选择）。"""
    return GameSpec(
        game_id=game_id,
        display_name=display_name,
        engine=EngineSpec(rules="mahjong.json", variant=variant, player_count=2),
        players=("p0", "p1"),
        eval_episodes=8,
        eval_opponents=("random", "mahjong"),  # 麻将启发式基线（已登记 runtime_solvers）
        solvers=_MAHJONG_SOLVERS,
        runtime_solvers=("mahjong", "random"),
    )


GAMES: dict[str, GameSpec] = {
    "moon_chess": GameSpec(
        game_id="moon_chess",
        display_name="月亮棋",
        engine=EngineSpec(rules="moon_chess.json"),
        players=("p_black", "p_white"),
        solvers={
            "hybrid": SolverPipeline(
                "hybrid",
                episodes=1,
                config={
                    "mode": "search",
                    "mcts_budget": 300,
                    "cfr_iterations": 800,
                    "cfr_depth_limit": 6,
                    "psro_iters": 5,
                    "psro_steps_per_iter": 2000,
                    "opponent_model": "psro",
                    "cfr_table_path": "$OUTDIR/cfr_table.json",
                },
            ),
            "cfr": SolverPipeline("cfr", entry="solve", config={"iterations": 800, "depth_limit": 6}),
            "ppo": SolverPipeline("ppo", episodes=300, config={"state_dim": 38, "action_dim": 9}, save="ppo.pt"),
            "psro": SolverPipeline(
                "psro", episodes=5, config={"num_iters": 5, "num_steps_per_iter": 2000}, save="psro_pool.npz"
            ),
            "qmix": SolverPipeline("qmix", episodes=2000, save="qmix.pt"),
            "happo": SolverPipeline("happo", episodes=2000, save="happo.pt"),
            "maac": SolverPipeline("maac", episodes=2000, save="maac.pt"),
        },
        eval_opponents=("random", "mcts", "self"),
        runtime_solvers=("mcts", "cfr", "hybrid", "random"),
    ),
    "stochastic_gomoku": GameSpec(
        game_id="stochastic_gomoku",
        display_name="随机五子棋",
        engine=EngineSpec(rules="stochastic_gomoku.json"),
        players=("p_black", "p_white"),
        solvers={
            "hybrid": SolverPipeline(
                "hybrid",
                episodes=1,
                config={
                    "mode": "search",
                    "mcts_budget": 200,
                    "cfr_iterations": 50,
                    "cfr_depth_limit": 5,
                    "opponent_model": "cfr",
                    "cfr_table_path": "$OUTDIR/cfr_table.json",
                },
            ),
            "cfr": SolverPipeline("cfr", entry="solve", config={"iterations": 50, "depth_limit": 5}),
        },
        eval_opponents=("random", "mcts"),
        runtime_solvers=("mcts", "cfr", "hybrid", "random"),
    ),
    "texas_holdem": GameSpec(
        game_id="texas_holdem",
        display_name="德州扑克",
        engine=EngineSpec(rules="texas_holdem.json"),
        players=("p_sb", "p_bb"),
        solvers={
            "hybrid": SolverPipeline(
                "hybrid",
                episodes=1,
                config={
                    "mode": "search",
                    "imperfect_information": True,
                    "mcts_budget": 300,
                    "cfr_iterations": 1000,
                    "cfr_depth_limit": 8,
                    "opponent_model": "cfr",
                    "cfr_table_path": "$OUTDIR/cfr_table.json",
                },
            ),
            "qmix": SolverPipeline(
                "qmix",
                episodes=6000,
                config={"epsilon_decay_steps": 20000},
                save="qmix.pt",
            ),
            "happo": SolverPipeline("happo", episodes=6000, save="happo.pt"),
            "maac": SolverPipeline("maac", episodes=6000, save="maac.pt"),
        },
        eval_opponents=("random", "mcts", "self"),
        runtime_solvers=("mcts", "hybrid", "random"),  # 德州运行时禁用 CFR（不完全信息）
        runtime_configs={"hybrid": {"imperfect_information": True}},
    ),
    "mahjong_guangdong": _mahjong_spec("mahjong_guangdong", "广东麻将（鸡胡）", "guangdong"),
    "mahjong_hongzhong": _mahjong_spec("mahjong_hongzhong", "红中麻将", "hongzhong"),
    "mahjong_blood": _mahjong_spec("mahjong_blood", "血战到底", "blood"),
    "werewolf": GameSpec(
        game_id="werewolf",
        display_name="狼人杀",
        engine=EngineSpec(rules="werewolf.json"),
        players=("p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"),
        eval_episodes=4,
        solvers={
            # 贝叶斯求解器无需训练（train 为 no-op），登记为可评估管线；
            # per_player → 每个座位一个实例（player_id=座位）。
            "bayes": SolverPipeline("bayes", episodes=1, per_player=True),
        },
        eval_opponents=("random", "self"),
        runtime_solvers=("ollama", "random"),
    ),
}


# ── 运行时装配（通用工厂，全部由注册表数据驱动）───────────────────


class RandomSolver(SolverBase):
    """均匀随机策略 — 基准求解器（注册表宿主）。"""

    def __init__(self, engine: GameEngine, seed: int | None = None) -> None:
        super().__init__(engine, SolverConfig(seed=seed))
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random"

    def select_action(self, state) -> Any | None:
        legal = self.engine.get_legal_actions(state)
        return self._rng.choice(legal) if legal else None

    def train(self, episodes: int, **kwargs: Any) -> SolverMetrics:
        return SolverMetrics(episodes=episodes)


def create_solver(
    game_id: str,
    name: str,
    engine: GameEngine,
    seed: int,
    budget: int,
    **kwargs: Any,
) -> SolverBase:
    """通用运行时求解器工厂 — 查注册表装配，无 per-game 分支。

    - 未知游戏 / 求解器不适用于该游戏 → ``ValueError``（数据驱动校验）。
    - ``budget`` 按 ``_BUDGET_FIELD`` 注入对应配置字段（如 mcts.budget /
      hybrid.mcts_budget）。
    - 额外 kwargs（``empirical_table`` / ``model`` / ``player_id`` …）合并进
      配置，由 config-class 自行拒绝未知字段。
    """
    factory = RUNTIME_FACTORY.get(name)
    if factory is None:
        raise ValueError(f"未知求解器: {name}（已注册: {', '.join(RUNTIME_FACTORY)}）")
    spec = GAMES.get(game_id)
    if spec is None:
        raise ValueError(f"未知游戏: {game_id}（已登记: {', '.join(GAMES)}）")
    if name not in spec.runtime_solvers:
        raise ValueError(f"求解器 {name} 不适用于 {game_id}（可选: {', '.join(spec.runtime_solvers)}）")
    cfg: dict[str, Any] = dict(RUNTIME_DEFAULTS.get(name, {}))
    cfg.update(spec.runtime_configs.get(name, {}))
    if name in _BUDGET_FIELD:
        cfg[_BUDGET_FIELD[name]] = budget
    cfg.update(kwargs)
    cfg.setdefault("seed", seed)
    return factory(engine, cfg)


class DefaultSolverProvider:
    """``SolverProvider`` 协议实现 — 完全数据驱动（无 per-game 分支）。

    在线学习：``attach_online_models`` 挂 ``OnlineModelStore``，创建 Hybrid
    时注入该游戏已发布的经验对手表（``empirical_table=...``）；显式传入的
    ``empirical_table`` kwarg 优先。
    """

    def __init__(self, online_models: Any | None = None) -> None:
        self.online_models = online_models

    def attach_online_models(self, online_models: Any) -> None:
        self.online_models = online_models

    def create_solver(
        self,
        game_id: str,
        name: str,
        engine: GameEngine,
        seed: int,
        budget: int,
        **kwargs: Any,
    ) -> SolverBase:
        if name == "hybrid" and self.online_models is not None:
            table = self.online_models.current_table(game_id)
            if table is not None:
                kwargs.setdefault("empirical_table", table)
        return create_solver(game_id, name, engine, seed, budget, **kwargs)


default_provider = DefaultSolverProvider()


def registered_game_ids() -> tuple[str, ...]:
    """已登记的游戏 id（按声明顺序）。"""
    return tuple(GAMES.keys())


def registered_solver_names() -> tuple[str, ...]:
    """训练可用的求解器名（按注册顺序）。"""
    return tuple(SOLVER_FACTORY.keys())
