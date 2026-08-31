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

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
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
    # 显式配置 > LLM_MODEL/LLM_BASE_URL 环境变量 > 内置默认（与统一客户端
    # 同优先级）——平台 LLM 配置页保存后经 env 桥让社交族求解器即时生效。
    model = cfg.get("model") or os.environ.get("LLM_MODEL", "").strip() or "qwen3:8b"
    base_url = cfg.get("base_url") or os.environ.get("LLM_BASE_URL", "").strip() or "http://localhost:11434"
    # 难度两维(平台 difficulty×pacing 3×3)透传给 OllamaConfig——
    # difficulty 选 ROLE_GUIDE 策略档 + 卧底词对档;pacing 调发言温度。
    return OllamaSolver(
        engine,
        OllamaConfig(
            model=model,
            base_url=base_url,
            difficulty=str(cfg.get("difficulty") or "normal"),
            pacing=str(cfg.get("pacing") or "standard"),
        ),
        player_id=cfg.get("player_id"),
    )


def _make_mcts(engine: GameEngine, cfg: dict[str, Any]) -> SolverBase:
    return MCTS(engine, MCTSConfig(**cfg))


def _make_maac(engine: GameEngine, cfg: dict[str, Any]) -> SolverBase:
    """Runtime MAAC solver: loads the trained checkpoint when available.

    ``cfg["model_path"]`` points at the saved ``maac.pt`` artifact (the
    train-cli ``create_solver`` injects ``models/train/<game_id>/maac.pt``
    by default).  When the checkpoint is missing the factory falls back to
    the game's heuristic solver so the platform never breaks — the
    fallback is the point of the pre-training default behavior.
    """
    model_path = cfg.get("model_path")
    try:
        solver = MAACSolver(engine, MAACConfig(seed=cfg.get("seed"), device=cfg.get("device", "cpu")))
    except Exception as exc:  # torch 缺失或引擎/动作空间异常 — 平台不能因此崩溃
        print(f"  [提示] MAAC 实例化失败（{exc}）— 回退到麻将启发式")
        return MahjongHeuristicAI(engine, SolverConfig(seed=cfg.get("seed")))
    if model_path and Path(model_path).exists():
        solver.load(str(model_path))
        return solver
    print(f"  [提示] MAAC 模型不存在: {model_path} — 回退到麻将启发式")
    return MahjongHeuristicAI(engine, SolverConfig(seed=cfg.get("seed")))


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
    "maac": _make_maac,
    "ollama": _make_ollama,
}

#: 运行时求解器默认配置（按求解器；游戏级覆盖见 GameSpec.runtime_configs）。
RUNTIME_DEFAULTS: dict[str, Mapping[str, Any]] = {
    "mcts": {"budget": 3000},
    "cfr": {"iterations": 1000, "depth_limit": 8},
    "hybrid": {"mode": "search", "cfr_iterations": 1000, "cfr_depth_limit": 8, "mcts_budget": 3000},
    "random": {},
    "mahjong": {},
    "maac": {"device": "cpu"},  # 运行时 MAAC：加载 <game>/maac.pt（缺文件回退启发式）
    # ollama 无默认值：_make_ollama 按 显式 > LLM_MODEL/LLM_BASE_URL env > 默认 解析，
    # 这样平台 LLM 配置页改的端点/模型也能覆盖社交族求解器。
    "ollama": {},
}

#: 调用期 budget 参数注入的配置字段名（按求解器）。
_BUDGET_FIELD: Mapping[str, str] = {"mcts": "budget", "hybrid": "mcts_budget"}

#: 训练产物根目录（train.py 把每游戏产物写到 ``<root>/models/train/<game_id>/``）。
#: ``create_solver`` 为 ``maac`` 注入默认 ``model_path`` 指向该目录下的
#: ``maac.pt`` —— 平台/基准装配时无需手工传路径；产物缺失时运行时工厂回退启发式。
_MODELS_TRAIN_DIR: Path = Path(__file__).resolve().parent.parent / "models" / "train"

#: ``allow_unknown=True`` 时允许为未登记游戏实例化的运行时求解器名
#: （平台自定义游戏族经 SolverProvider 装配使用；其余名字维持 ValueError）。
_RUNTIME_UNKNOWN_ALLOWED: tuple[str, ...] = ("mcts", "random", "ollama", "mahjong", "hybrid")


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
    #: 每评估对手的搜索预算覆盖（如 {"mcts": 30}）；缺省用 train.py 全局预算。
    #: 麻将的 MCTS 每决策 ~200ms/迭代，全局 300 预算在此游戏上不现实——登记
    #: 覆盖让内置评估的实际 MCTS 基线预算可执行同时保持对比语义一致。
    eval_budgets: Mapping[str, int] = field(default_factory=dict)
    # 每评估对手的搜索预算覆盖（缺省用 train.py 的全局 EVAL_MCTS_BUDGET）。
    # 麻将 MCTS 每决策成本 ~200ms/迭代，全局 300 预算在该游戏上会跑数小时——
    # 麻将登记一条 {"mcts": 30} 使内置评估在相同语义下可实际执行。
    eval_budgets: Mapping[str, int] = field(default_factory=dict)


# ── 注册表 ─────────────────────────────────────────────────────────
# 默认超参/局数沿用既有实测标定（train_hybrid / train_marl / benchmark_all）：
#   moon_chess:        CFR 800 iters ≈ 3 min；PSRO 5×20000 ≈ 3-6 min（双座位 BR）；MARL 600 局 ≈ 10 s
#   stochastic_gomoku: CFR 50 iters ≈ 5-8 min（9×9 深限）
#   texas_holdem:      CFR 1000 iters ≈ 75 s；MARL 400 局 ≈ 30 s
#   mahjong_* (4 人):  MARL 200 局 ≈ 12 min（约 3.5 s/局）；评估建议 4-8 局

#: 麻将 MARL 默认对手编排配置（训练对手编排机制的一站式入口）。
#: 开启 pfsp 优先虚构自博弈：每 ``checkpoint_interval`` 局把当前策略冻结入池
#: （容量 ``pool_capacity``），学习器座位逐局轮换，对手座位按胜率加权采样池
#: 快照；池空/warmup 阶段自动退化为纯自博弈（与旧行为兼容）。``eval_interval``
#: 做 vs-random 固定基线曲线采样，直接产出"训练曲线平滑度"证据。
_MAHJONG_OPPONENT_CFG: Mapping[str, Any] = {
    "opponent_enabled": True,
    "opponent_mode": "pfsp",  # self | uniform | pfsp | curriculum
    "opponent_pool_capacity": 32,
    "opponent_checkpoint_interval": 25,  # 每 25 局冻结一次当前策略入池
    "opponent_warmup": 100,  # 起步 100 局纯自博弈（先有基础行为再入池）
    "opponent_pfsp_alpha": 1.0,
    "opponent_pfsp_floor": 0.1,
    "opponent_pfsp_priority": "win",  # win → p∝win_rate^α；lose → p∝(1−win_rate)^α
    "opponent_recency_decay": 0.9,
    "opponent_win_memory": 50,
    "opponent_role_alternate": True,
    "eval_interval": 50,  # 每 50 局做一次 vs-random 曲线采样
    "eval_episodes": 5,
}

_MAHJONG_SOLVERS: Mapping[str, SolverPipeline] = {
    "qmix": SolverPipeline("qmix", episodes=200, config={**_MAHJONG_OPPONENT_CFG}, save="qmix.pt"),
    "happo": SolverPipeline("happo", episodes=200, config={**_MAHJONG_OPPONENT_CFG}, save="happo.pt"),
    "maac": SolverPipeline("maac", episodes=200, config={**_MAHJONG_OPPONENT_CFG}, save="maac.pt"),
}


def _mahjong_spec(game_id: str, display_name: str, variant: str) -> GameSpec:
    """构造麻将变种登记条目（同一 rules 文件 + variants 声明选择）。

    麻将标准人数为 4 人：引擎按 4 人装配，训练（MARL）与评估都在
    四座位（p0-p3）上进行，与 rules/mahjong.json 的默认声明一致。
    """
    return GameSpec(
        game_id=game_id,
        display_name=display_name,
        engine=EngineSpec(rules="mahjong.json", variant=variant, player_count=4),
        players=("p0", "p1", "p2", "p3"),
        eval_episodes=8,
        eval_opponents=("random", "mahjong", "mcts"),  # 启发式 + MCTS 基线（均登记 runtime_solvers）
        solvers=_MAHJONG_SOLVERS,
        runtime_solvers=("mahjong", "random", "mcts", "maac"),
        # MCTS 在麻将上每决策 ~200ms/迭代：浅 rollout 压低评估成本；训练好的
        # MAAC 模型（maac.pt）经 create_solver("maac") 自动注入默认路径加载。
        runtime_configs={"mcts": {"rollout_depth": 8}},
        eval_budgets={"mcts": 30},
    )


_UNO_SOLVERS: Mapping[str, SolverPipeline] = {
    # UNO 手牌部分可观测（visibility 声明）→ 与德州同样的不完全信息配置。
    "hybrid": SolverPipeline(
        "hybrid",
        episodes=1,
        config={
            "mode": "search",
            "imperfect_information": True,
            "mcts_budget": 300,
            "cfr_iterations": 300,
            "cfr_depth_limit": 6,
            "opponent_model": "cfr",
            "cfr_table_path": "$OUTDIR/cfr_table.json",
        },
    ),
}


def _uno_spec(game_id: str, display_name: str, variant: str) -> GameSpec:
    """构造 UNO 变种登记条目（同一 rules 文件 + variants 声明选择，4 人默认）。"""
    return GameSpec(
        game_id=game_id,
        display_name=display_name,
        engine=EngineSpec(rules="uno.json", variant=variant, player_count=4),
        players=("p0", "p1", "p2", "p3"),
        eval_episodes=6,
        eval_opponents=("random", "self", "mcts"),
        solvers=_UNO_SOLVERS,
        runtime_solvers=("mcts", "hybrid", "random"),
        runtime_configs={"hybrid": {"imperfect_information": True}},
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
            "ppo": SolverPipeline(
                "ppo",
                # 自博弈 + 双座位轮换 + 零和 bootstrap 取负（审查 2026-08: 旧默认
                # 对手=random 且只训练黑方座位，next_value 又用了对手视角 → 自博弈
                # 塌缩；修复后 seed 5 黑/白 vs random 0.88/0.77、seed 42 0.76/0.74
                # @N=100，均高于随机基线 0.57/0.425）。局数 300 → 600、entropy
                # 0.01 → 0.05 进一步抑制塌缩。
                episodes=600,
                config={
                    "state_dim": 38,
                    "action_dim": 9,
                    "opponent": "self",
                    "entropy_coef": 0.05,
                },
                save="ppo.pt",
            ),
            "psro": SolverPipeline(
                "psro",
                episodes=5,
                # 训练对手不是随机：PSRO 的 BR 对着 Nash 混合训练且双座位交替
                # （tabular_q 按局轮换训练座位，共享表对黑/白都有效）；预算
                # 2000 → 20000 步、元博弈 Ne 10 → 30（审查 2026-08: 旧配置
                # 在 19683 状态上只够 ~100 局，BR≈随机 → 池塌缩成 2）。
                config={"num_iters": 5, "num_steps_per_iter": 20000, "evaluation_episodes": 30},
                save="psro_pool.npz",
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
    "mahjong_blood": _mahjong_spec("mahjong_blood", "血流成河", "blood"),
    "mahjong_sichuan": _mahjong_spec("mahjong_sichuan", "四川麻将（血战到底）", "sichuan"),
    "mahjong_changsha": _mahjong_spec("mahjong_changsha", "长沙麻将（258将）", "changsha"),
    "mahjong_taiwan": _mahjong_spec("mahjong_taiwan", "台湾麻将（16张）", "taiwan"),
    "mahjong_international": _mahjong_spec("mahjong_international", "国际麻将（国标）", "international"),
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
    "undercover": GameSpec(
        game_id="undercover",
        display_name="谁是卧底",
        # v5.2 声明式：scenario(词对) + 人数由 rules JSON 的 variants 选择
        # （1卧底+1白板+N平民；人数 4..12 可用 player_count 覆盖）。
        engine=EngineSpec(rules="undercover.json", variant="fruit_normal", player_count=8),
        players=("p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"),
        eval_episodes=4,
        solvers={},  # 暂无专用可训练求解器（与狼人杀同类的自由发言桌游）
        eval_opponents=("random", "self"),
        runtime_solvers=("ollama", "random"),
    ),
    "uno": _uno_spec("uno", "UNO（经典）", "classic"),
    "uno_seven_zero": _uno_spec("uno_seven_zero", "UNO 7-0（换手/移交）", "seven_zero"),
    "uno_jump_in": _uno_spec("uno_jump_in", "UNO 抢牌", "jump_in"),
    "uno_stacking": _uno_spec("uno_stacking", "UNO +2叠加", "stacking"),
    "uno_draw_until": _uno_spec("uno_draw_until", "UNO 摸到能打", "draw_until"),
    "uno_strict_wild4": _uno_spec("uno_strict_wild4", "UNO 严格+4", "strict_wild4"),
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
    *,
    allow_unknown: bool = False,
    **kwargs: Any,
) -> SolverBase:
    """通用运行时求解器工厂 — 查注册表装配，无 per-game 分支。

    - 未知游戏 / 求解器不适用于该游戏 → ``ValueError``（数据驱动校验）。
    - ``allow_unknown=True``：游戏未登记时仍按
      ``RUNTIME_FACTORY[name]`` + ``RUNTIME_DEFAULTS`` + 预算注入实例化
      （平台自定义游戏族装配用，仅限 ``_RUNTIME_UNKNOWN_ALLOWED`` 中的
      通用运行时求解器名）。
    - ``budget`` 按 ``_BUDGET_FIELD`` 注入对应配置字段（如 mcts.budget /
      hybrid.mcts_budget）。
    - 额外 kwargs（``empirical_table`` / ``model`` / ``player_id`` …）合并进
      配置，由 config-class 自行拒绝未知字段。
    - ``rollout_policy``（str）是装配指令而非 config 字段：弹出后在
      hybrid 构造后注入其内部 MCTS 的 ``rollout_policy``（如 UNO 启发式
      先验），使裸 MCTS 的随机 rollout 信号变强。仅 hybrid 生效；非
      hybrid 传此指令静默跳过。
    """
    factory = RUNTIME_FACTORY.get(name)
    if factory is None:
        raise ValueError(f"未知求解器: {name}（已注册: {', '.join(RUNTIME_FACTORY)}）")
    # rollout_policy 是装配指令（字符串名），不进 config-class（HybridConfig
    # 拒绝未知字段）；弹出后在 hybrid 构造后注入其内部 MCTS。
    rollout_policy_name = kwargs.pop("rollout_policy", None)
    spec = GAMES.get(game_id)
    if spec is None:
        if allow_unknown and name in _RUNTIME_UNKNOWN_ALLOWED:
            cfg: dict[str, Any] = dict(RUNTIME_DEFAULTS.get(name, {}))
            if name in _BUDGET_FIELD:
                cfg[_BUDGET_FIELD[name]] = budget
            cfg.update(kwargs)
            cfg.setdefault("seed", seed)
            solver = factory(engine, cfg)
        else:
            raise ValueError(f"未知游戏: {game_id}（已登记: {', '.join(GAMES)}）")
    else:
        if name not in spec.runtime_solvers:
            raise ValueError(f"求解器 {name} 不适用于 {game_id}（可选: {', '.join(spec.runtime_solvers)}）")
        cfg = dict(RUNTIME_DEFAULTS.get(name, {}))
        cfg.update(spec.runtime_configs.get(name, {}))
        if name in _BUDGET_FIELD:
            cfg[_BUDGET_FIELD[name]] = budget
        cfg.update(kwargs)
        if name == "maac":
            # 训练产物默认路径：<root>/models/train/<game_id>/maac.pt。显式传入的
            # ``model_path`` kwarg 优先；产物缺失时运行时工厂回退到该游戏启发式，
            # 平台默认 AI 因此是"已训练 MAAC，否则不崩的启发式"。
            cfg.setdefault("model_path", str(_MODELS_TRAIN_DIR / game_id / "maac.pt"))
        cfg.setdefault("seed", seed)
        solver = factory(engine, cfg)
    _inject_rollout_policy(solver, name, rollout_policy_name, engine, seed)
    return solver


def _rollout_policy_factory(name: str) -> Callable[[GameEngine, int], Any] | None:
    """按名取 rollout_policy 工厂（懒 import 避免顶层循环依赖）。

    当前注册：``"uno"`` → ``UnoRolloutPolicy``。新增策略在此分支即可，
    不污染 config-class（装配指令与配置分离）。
    """
    if name == "uno":
        from layer3_solvers.uno.heuristic import UnoRolloutPolicy

        return lambda engine, seed: UnoRolloutPolicy(engine, seed)
    return None


def _inject_rollout_policy(
    solver: SolverBase, solver_name: str, policy_name: str | None, engine: GameEngine, seed: int
) -> None:
    """hybrid 专属：把 rollout 启发式设到其内部 MCTS 的 ``rollout_policy``。

    非 hybrid / 无 policy_name 时静默跳过；policy_name 未知 → ValueError
    （装配指令拼错应尽早暴露，而非静默退化）。注意此注入只影响裸 MCTS
    的 rollout 路径（hybrid 无 ``hiddenWorld`` / 无 CFR 表时走的
    ``_select_search`` 分支）；PIMC（``_opponent_mcts``）用 hybrid 自己的
    ``_rollout_prior``，不受此影响。
    """
    if not policy_name:
        return
    if solver_name != "hybrid":
        return  # 非 hybrid 无内部 MCTS，rollout_policy 指令无意义（静默跳过）
    factory = _rollout_policy_factory(policy_name)
    if factory is None:
        raise ValueError(f"未知 rollout_policy: {policy_name}（已注册: uno）")
    mcts = getattr(solver, "mcts", None)
    if mcts is None:  # pragma: no cover — hybrid 构造必有 mcts
        return
    mcts.rollout_policy = factory(engine, seed)


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
                # 注入经验表后必须同时把对手模型切到 "empirical"，否则
                # HybridConfig.opponent_model 默认 "uniform"，经验表从未被读取。
                kwargs.setdefault("opponent_model", "empirical")
        return create_solver(game_id, name, engine, seed, budget, **kwargs)


default_provider = DefaultSolverProvider()


def registered_game_ids() -> tuple[str, ...]:
    """已登记的游戏 id（按声明顺序）。"""
    return tuple(GAMES.keys())


def registered_solver_names() -> tuple[str, ...]:
    """训练可用的求解器名（按注册顺序）。"""
    return tuple(SOLVER_FACTORY.keys())
