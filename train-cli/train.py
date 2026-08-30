#!/usr/bin/env python3
"""统一的抽象训练脚本 — Gavis train-cli.

对所有已登记游戏通用：游戏的引擎构造（rules + 变种/人数）、座位、可训练
求解器管线、默认超参/局数/评估设置全部来自 ``games.py`` 注册表
（``GAMES``）。本脚本**不含任何 per-game 分支**——新游戏接入 = 在注册表
增加一个 ``GameSpec`` 条目即可。

训练管线（``SolverPipeline``）：
  - ``entry="train"`` → ``solver.train(episodes=...)``（Hybrid/MARL/PPO/PSRO）
  - ``entry="solve"``  → ``solver.solve(initial_state)``（CFR 等效训练）
  - ``per_player=True`` → 每个座位建一个实例（player_id=座位，如贝叶斯狼人杀）
训练后按注册表 ``eval`` 设置运行通用评估（座位轮换；对手由注册表
``eval_opponents`` 数据驱动：random 均匀随机 / self 自博弈镜像 / 或任何
已登记运行时求解器（如 mcts 搜索基线、mahjong 启发式基线））。

产物（``<out-dir>/<game>/``）：
  - ``{save 名}``         各求解器产物（如 qmix.pt / cfr 表由 $OUTDIR 路径落盘）
  - ``config.json``       引擎/座位/各管线配置（可复现）
  - ``metrics.json``      每求解器训练指标 + 评估指标 + 耗时

Usage::

    python train-cli/train.py [--game all|moon_chess|stochastic_gomoku|texas_holdem|
                                      mahjong_guangdong|mahjong_hongzhong|mahjong_blood|
                                      mahjong_sichuan|mahjong_changsha|mahjong_taiwan|
                                      mahjong_international|werewolf]
                              [--solver all|hybrid|cfr|ppo|psro|qmix|happo|maac|bayes]
                              [--episodes N] [--seed N] [--device auto|cpu|cuda]
                              [--out-dir models/train] [--eval-episodes N]
                              [--preset full|quick]  # 默认 full=完整训练; quick=演示校准
                              [--config-override KEY=VALUE,...]  # 管线配置覆盖
                              [--skip-eval] [--list] [--verbose]

例（大参数 + 训练对手编排）::

    python train-cli/train.py --game mahjong_guangdong --solver qmix \\
        --episodes 1200 --config-override hidden_dim=512,opponent_enabled=true, \\
        opponent_mode=pfsp,opponent_checkpoint_interval=25,eval_interval=100

编排默认值见 ``games.py`` 的 ``_MAHJONG_OPPONENT_CFG``（麻将管线默认开启
PFSP 对手池），设计文档：``docs/design/training-opponent-scheduling.md``。

``--preset`` 选择训练预设：``full``（默认）按注册表完整训练；``quick`` 在
运行时按系数缩放注册表读出的局数与预算类超参（演示校准，几秒~几十秒级），
不改 ``games.py``。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

# ── 路径引导（train-cli 目录含连字符，不能作为包导入；本文件可直接执行）──
_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from games import GAMES, SOLVER_FACTORY, GameSpec, SolverPipeline, create_solver  # noqa: E402

from layer2_engine.core.engine import GameEngine  # noqa: E402
from layer3_solvers.base import SolverBase, SolverMetrics  # noqa: E402

#: 评估对局最大步数护栏（超出按当前效用截断，防死循环）。
MAX_EVAL_STEPS = 600

#: 评估对手 —— MCTS 基线的通用搜索预算（与 Hybrid 自身 mcts_budget 同量级，
#: 使“vs 基线”与“vs 自己”的对比在同一规模下进行）。
EVAL_MCTS_BUDGET = 300

#: 训练预设缩放系数：full=1.0（原样），quick=0.2（演示校准）。
PRESET_FACTORS: dict[str, float] = {"full": 1.0, "quick": 0.2}

#: quick 缩放会波及的管线 config 键（迭代/预算类；depth 改变搜索语义，不缩放）。
_SCALED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "mcts_budget",
        "cfr_iterations",
        "iterations",
        "psro_iters",
        "psro_steps_per_iter",
        "num_iters",
        "num_steps_per_iter",
        "evaluation_episodes",
    }
)

#: quick 缩放下限（key → 最小保留值），避免缩到 0/1 导致策略退化。
_QUICK_FLOORS: dict[str, int] = {
    "episodes": 1,
    "eval_episodes": 2,
    "eval_mcts_budget": 50,
    "mcts_budget": 20,
    "cfr_iterations": 10,
    "iterations": 10,
    "psro_iters": 1,
    "psro_steps_per_iter": 200,
    "num_iters": 1,
    "num_steps_per_iter": 2000,
    "evaluation_episodes": 4,
}


def _preset_factor(preset: str) -> float:
    """返回预设的缩放系数（未知值已在 argparse ``choices`` 拦截）。"""
    return PRESET_FACTORS.get(preset, 1.0)


def _scale_int(value: int, factor: float, floor: int) -> int:
    """按系数缩放整数值并施加下限（quick 演示校准；full 为恒等）。"""
    return max(floor, int(value * factor))


def apply_preset(pipeline: SolverPipeline, preset: str) -> SolverPipeline:
    """按预设缩放训练管线（episodes + 预算类 config 键）；full 原样返回。

    只改注册表读出的默认值（``--episodes`` / ``--config-override`` 显式
    覆盖在调用方优先），不改 ``games.py`` 注册表本身。
    """
    factor = _preset_factor(preset)
    if factor >= 1.0:
        return pipeline
    scaled_config = {
        key: _scale_int(value, factor, _QUICK_FLOORS.get(key, 1))
        if key in _SCALED_CONFIG_KEYS and isinstance(value, int)
        else value
        for key, value in pipeline.config.items()
    }
    return replace(
        pipeline,
        episodes=_scale_int(pipeline.episodes, factor, _QUICK_FLOORS["episodes"]),
        config=scaled_config,
    )


# ── 引擎 ───────────────────────────────────────────────────────────


def build_engine(spec: GameSpec, seed: int) -> GameEngine:
    """按注册表 EngineSpec 构造引擎（v5.2：变种/人数纯数据选择）。"""
    rules_path = _ROOT / "rules" / spec.engine.rules
    with open(rules_path, encoding="utf-8") as f:
        rules = json.load(f)
    kwargs: dict[str, Any] = {}
    if spec.engine.variant is not None:
        kwargs["variant"] = spec.engine.variant
    if spec.engine.player_count is not None:
        kwargs["player_count"] = spec.engine.player_count
    return GameEngine(rules, seed=seed, **kwargs)


# ── 求解器装配 ─────────────────────────────────────────────────────


def _expand_outdir(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """把配置中 ``$OUTDIR`` 占位符展开为实际输出目录（路径配置与目录解耦）。"""
    expanded = {}
    for key, value in config.items():
        if isinstance(value, str) and "$OUTDIR" in value:
            expanded[key] = value.replace("$OUTDIR", str(out_dir))
        else:
            expanded[key] = value
    return expanded


def _coerce_value(raw: str) -> Any:
    """把 CLI 覆盖值解析为 bool/int/float/None/字符串（数据驱动覆盖）。"""
    s = raw.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _merge_overrides(kwargs: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """把 ``KEY=VALUE``（支持点路径）合并进配置字典；KEY 不存在时直接新增。"""
    for kv in overrides or []:
        key, _, val = kv.partition("=")
        parts = [p.strip() for p in key.split(".")]
        d = kwargs
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = _coerce_value(val)
    return kwargs


def make_solver(
    name: str,
    engine: GameEngine,
    pipeline: SolverPipeline,
    seed: int,
    device: str,
    out_dir: Path,
    player_id: str | None = None,
    overrides: list[str] | None = None,
) -> SolverBase:
    """按注册表 SOLVER_FACTORY 实例化求解器（数据驱动，无 per-game 分支）。"""
    entry = SOLVER_FACTORY[name]
    if entry is None or entry[0] is None:
        raise RuntimeError(f"求解器 {name} 需要可选依赖（torch/psro extra），当前环境不可用——请安装后重试")
    cls, cfg_cls = entry
    kwargs = dict(pipeline.config or {})
    kwargs = _merge_overrides(kwargs, overrides)
    kwargs = _expand_outdir(kwargs, out_dir)
    kwargs.setdefault("seed", seed)
    kwargs.setdefault("device", device)
    config = cfg_cls(**kwargs) if cfg_cls is not None else None
    if player_id is not None:
        return cls(engine, config, player_id=player_id)
    return cls(engine, config)


# ── 训练入口 ───────────────────────────────────────────────────────


def _metrics_dict(result: Any) -> dict[str, Any]:
    """把训练返回值规范化为可序列化指标（SolverMetrics 或求解器自有形状）。"""
    if isinstance(result, SolverMetrics):
        return {
            "episodes": result.episodes,
            "win_rate": result.win_rate,
            "avg_return": result.avg_return,
            "extra": dict(result.extra),
        }
    if isinstance(result, dict):  # CFR solve() → root 策略 dict
        return {"strategy_keys": len(result)}
    return {"raw": repr(result)}


def run_entry(solver: SolverBase, pipeline: SolverPipeline, episodes: int, verbose: bool) -> dict[str, Any]:
    """按管线 entry 执行训练/求解（\"train\" → train(episodes)；\"solve\" → solve(initial)）。"""
    if pipeline.entry == "solve":
        result = solver.solve(solver.engine.create_initial_state(), verbose=verbose)
        return _metrics_dict(result)
    result = solver.train(episodes=episodes, verbose=verbose)
    return _metrics_dict(result)


def _save_implemented(solver: SolverBase) -> bool:
    """求解器是否真正实现了 save()（默认 SolverBase.save 是 no-op）。"""
    return type(solver).save is not SolverBase.save


def save_artifact(solver: SolverBase, pipeline: SolverPipeline, out_dir: Path) -> Path | None:
    """按管线 save 名保存产物；未实现 save() 或未声明 save 名则跳过。"""
    if pipeline.save is None or not _save_implemented(solver):
        return None
    path = out_dir / pipeline.save
    solver.save(str(path))
    return path


# ── 通用评估（vs 均匀随机，座位轮换）───────────────────────────────


def play_episode(
    engine: GameEngine, owners: dict[str, SolverBase | None], rng: random.Random
) -> tuple[int, dict[str, float]]:
    """通用对局循环：chance 采样 + 按座位分派求解器/随机，返回 (步数, 各方效用)。"""
    state = engine.create_initial_state()
    steps = 0
    while not engine.is_terminal(state) and steps < MAX_EVAL_STEPS:
        node = engine.get_node_type(state)
        if node == "chance":
            outcomes = engine.get_chance_outcomes(state)
            if not outcomes:
                break
            probs = [float(getattr(o, "probability", 0.0) or 0.0) for o in outcomes]
            if sum(probs) <= 0:
                state = engine.apply_chance(state, rng.choice(outcomes))
            else:
                state = engine.apply_chance(state, rng.choices(outcomes, weights=probs, k=1)[0])
            steps += 1
            continue
        if node != "player":
            break
        current = engine.get_current_player(state)
        solver = owners.get(current)
        legal = engine.get_legal_actions(state)
        if not legal:
            break
        # own 座位用求解器；其余座位（owners 中为 None）按均匀随机落子——
        # 这是评估协议“vs 均匀随机”的语义，不能把 None 当成“无代理→中止”。
        if solver is None:
            action = rng.choice(legal)
        else:
            action = solver.select_action(state)
            if action is None:
                action = rng.choice(legal)
        state = engine.apply_action(state, action)
        steps += 1
    payoffs = {p: float(engine.get_utility(state, p)) for p in owners}
    return steps, payoffs


def evaluate(
    engine: GameEngine,
    spec: GameSpec,
    solver: SolverBase | None,
    per_player_instances: list[SolverBase] | None,
    episodes: int,
    base_seed: int,
    opponents: tuple[str, ...] = ("random",),
    eval_budget: int = EVAL_MCTS_BUDGET,
) -> dict[str, Any]:
    """通用评估：每局只有一个"own 座位"由被评求解器执掌（顺次轮换），其余座位按对手类型落子。

    返回 ``{对手类型: 结果}``。对手类型（**数据驱动**，注册表 ``eval_opponents`` 声明）：

    - ``random`` — 均匀随机（内置基准下限）。
    - ``self``   — 自己镜像（内置；per_player 时各座位用自身实例互博）。
    - 其余名字  — 必须是该游戏 ``runtime_solvers`` 里已登记的求解器（如
      ``mahjong`` 启发式 / ``mcts`` / ``ollama``），经 ``create_solver`` 通用装配
      （预算用 ``eval_budget``，默认 ``EVAL_MCTS_BUDGET``）；未登记则该列自动跳过并提示。

    - ``solver`` != None（普通管线）→ own 座位用该实例。
    - ``per_player_instances`` != None → own 座位用对应座位的实例（player_id 绑定）。
    """
    baselines: dict[str, SolverBase] = {}
    for opp in opponents:
        if opp in ("random", "self"):
            continue
        if opp in spec.runtime_solvers:
            baselines[opp] = create_solver(spec.game_id, opp, engine, base_seed, budget=eval_budget)
        else:
            print(f"  [提示] {spec.game_id} 未登记 '{opp}' 运行时求解器，跳过该评估列")

    results: dict[str, Any] = {}
    for opponent in opponents:
        if opponent not in ("random", "self") and opponent not in baselines:
            continue
        t0 = time.perf_counter()
        utils: list[float] = []
        for ep in range(episodes):
            rng = random.Random(base_seed + ep * 31 + 7)
            owned = ep % len(spec.players)
            owners: dict[str, SolverBase | None] = {}
            for i, seat in enumerate(spec.players):
                owned_solver = per_player_instances[i] if per_player_instances is not None else solver
                if opponent == "random":
                    owners[seat] = owned_solver if i == owned else None
                elif opponent == "self":
                    owners[seat] = owned_solver
                else:
                    owners[seat] = owned_solver if i == owned else baselines[opponent]
            _, payoffs = play_episode(engine, owners, rng)
            utils.append(payoffs.get(spec.players[owned], 0.0))
        elapsed = time.perf_counter() - t0
        wins = sum(u > 0 for u in utils)
        draws = sum(u == 0 for u in utils)
        results[opponent] = {
            "opponent": opponent,
            "episodes": episodes,
            "wins": wins,
            "draws": draws,
            "losses": episodes - wins - draws,
            "win_rate": round(wins / episodes, 4),
            "avg_utility": round(sum(utils) / episodes, 4),
            "seconds": round(elapsed, 2),
        }
    return results


# ── 单游戏训练 ─────────────────────────────────────────────────────


def train_one(
    engine: GameEngine,
    spec: GameSpec,
    name: str,
    pipeline: SolverPipeline,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    """训练一个 (游戏, 求解器) 管线并返回指标记录。"""
    pipeline = apply_preset(pipeline, args.preset)
    episodes = args.episodes if args.episodes is not None else pipeline.episodes
    print(f"\n── 求解器 {name} @ {spec.game_id}  ({pipeline.entry}, {episodes} 局) ──")
    t0 = time.perf_counter()

    if pipeline.per_player:
        instances = [
            make_solver(
                name, engine, pipeline, args.seed, args.device, out_dir, player_id=seat, overrides=args.config_overrides
            )
            for seat in spec.players
        ]
        metrics = run_entry(instances[0], pipeline, episodes, args.verbose)
        record: dict[str, Any] = {
            "solver": name,
            "entry": pipeline.entry,
            "episodes": episodes,
            "per_player": True,
            "train_seconds": round(time.perf_counter() - t0, 2),
            "metrics": metrics,
            "config": dict(pipeline.config),
        }
    else:
        solver = make_solver(name, engine, pipeline, args.seed, args.device, out_dir, overrides=args.config_overrides)
        metrics = run_entry(solver, pipeline, episodes, args.verbose)
        artifact = save_artifact(solver, pipeline, out_dir)
        record = {
            "solver": name,
            "entry": pipeline.entry,
            "episodes": episodes,
            "train_seconds": round(time.perf_counter() - t0, 2),
            "metrics": metrics,
            "config": dict(pipeline.config),
        }
        if artifact is not None:
            record["artifact"] = str(artifact)

    if pipeline.eval and not args.skip_eval:
        n = args.eval_episodes or _scale_int(
            spec.eval_episodes, _preset_factor(args.preset), _QUICK_FLOORS["eval_episodes"]
        )
        eval_budget = _scale_int(EVAL_MCTS_BUDGET, _preset_factor(args.preset), _QUICK_FLOORS["eval_mcts_budget"])
        opponents: tuple[str, ...] = args.eval_opponents or spec.eval_opponents
        if pipeline.per_player:
            record["eval"] = evaluate(engine, spec, None, instances, n, args.seed, opponents, eval_budget=eval_budget)
        else:
            record["eval"] = evaluate(engine, spec, solver, None, n, args.seed, opponents, eval_budget=eval_budget)
        for opp, res in record["eval"].items():
            print(f"  评估 vs {opp}: win_rate={res['win_rate']:.3f}  avg_utility={res['avg_utility']:+.3f}")
    print(f"  训练完成: {record['train_seconds']}s")
    return record


def train_game(spec: GameSpec, requested: list[str], args: argparse.Namespace) -> dict[str, Any]:
    """按注册表训练一个游戏；返回汇总指标。"""
    print(f"\n{'-' * 62}")
    print(f"  {spec.display_name} ({spec.game_id})  players={list(spec.players)}")
    print(f"{'-' * 62}")

    engine = build_engine(spec, args.seed)
    out_dir = _ROOT / args.out_dir / spec.game_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_all: dict[str, Any] = {}
    config_doc = {
        "game": spec.game_id,
        "display_name": spec.display_name,
        "engine": {
            "rules": spec.engine.rules,
            "variant": spec.engine.variant,
            "player_count": spec.engine.player_count,
        },
        "players": list(spec.players),
        "seed": args.seed,
        "device": args.device,
        "pipelines": {},
    }

    for name, pipeline in spec.solvers.items():
        if name not in requested:
            continue
        record = train_one(engine, spec, name, pipeline, args, out_dir)
        metrics_all[name] = record
        config_doc["pipelines"][name] = {
            "entry": pipeline.entry,
            "episodes": pipeline.episodes,
            "config": dict(pipeline.config),
            "save": pipeline.save,
            "per_player": pipeline.per_player,
            "eval": pipeline.eval,
        }

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_doc, f, ensure_ascii=False, indent=2)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)
    print(f"\n  → {out_dir / 'config.json'} / {out_dir / 'metrics.json'}")
    return {"game": spec.game_id, "display_name": spec.display_name, **metrics_all}


def print_summary(summaries: list[dict[str, Any]]) -> None:
    """打印跨游戏汇总表（win_rate / 训练耗时）。"""
    print(f"\n{'-' * 62}")
    print("  训练总结")
    print(f"{'-' * 62}")
    for summary in summaries:
        solvers = [k for k in summary if k not in ("game", "display_name")]
        if not solvers:
            continue
        parts = []
        for name in solvers:
            rec = summary[name]
            eval_ = rec.get("eval")
            wr = next(iter(eval_.values()), {}).get("win_rate") if eval_ else None
            parts.append(f"{name}={wr if wr is not None else '-'}")
        print(f"  {summary['game']:24s}  {'  '.join(parts)}")


# ── CLI ────────────────────────────────────────────────────────────


def _resolve_games(game_arg: str) -> list[GameSpec]:
    if game_arg == "all":
        return list(GAMES.values())
    games = [g.strip() for g in game_arg.split(",") if g.strip()]
    unknown = [g for g in games if g not in GAMES]
    if unknown:
        raise SystemExit(f"未知游戏: {', '.join(unknown)}（已登记: {', '.join(GAMES)}）")
    return [GAMES[g] for g in games]


def _resolve_solvers(solver_arg: str, spec: GameSpec) -> list[str]:
    """把 --solver 解析为该游戏实际登记的管线名（交集；未登记的在汇总中提示）。"""
    if solver_arg == "all":
        return list(spec.solvers.keys())
    requested = [s.strip() for s in solver_arg.split(",") if s.strip()]
    available = [s for s in requested if s in spec.solvers]
    skipped = [s for s in requested if s not in spec.solvers]
    if skipped:
        print(f"  [提示] 该游戏未登记求解器: {', '.join(skipped)}（跳过）")
    return available


def _print_registry() -> None:
    """打印注册表一览（游戏 × 训练管线）。"""
    print(f"{'游戏':24s} {'座位':16s} 训练管线")
    print(f"{'─' * 76}")
    for spec in GAMES.values():
        pipelines = ", ".join(
            f"{k}({v.entry})" if v.entry == "solve" else f"{k}({v.entry},{v.episodes})" for k, v in spec.solvers.items()
        )
        print(f"{spec.game_id:24s} {','.join(spec.players):16s} {pipelines}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gavis 统一训练 CLI（配置驱动，无 per-game 逻辑）")
    parser.add_argument("--game", type=str, default="all", help="游戏 id（逗号分隔）或 'all'")
    parser.add_argument("--solver", type=str, default="all", help="求解器名（逗号分隔）或 'all'")
    parser.add_argument("--episodes", type=int, default=None, help="覆盖各管线训练局数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", type=str, default="models/train")
    parser.add_argument(
        "--preset",
        type=str,
        default="full",
        choices=["full", "quick"],
        help="训练预设：full=完整训练（默认）；quick=演示校准（按 0.2 系数缩放注册表局数/预算，带下限）",
    )
    parser.add_argument("--eval-episodes", type=int, default=0, help="覆盖评估局数（0=注册表默认）")
    parser.add_argument(
        "--eval-opponents",
        type=str,
        default=None,
        help="评估对手（逗号分隔: random,self 或任何已登记 runtime_solvers 名字，如 mahjong,mcts；默认用注册表 eval_opponents）",
    )
    parser.add_argument(
        "--config-override",
        type=str,
        default=None,
        help="管线配置覆盖（逗号分隔 KEY=VALUE；点路径如 opponent_mode / hidden_dim；作用于该游戏所有命中的求解器管线）",
    )
    parser.add_argument("--skip-eval", action="store_true", help="跳过训练后评估")
    parser.add_argument("--list", action="store_true", help="打印注册表一览并退出")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.eval_opponents:
        args.eval_opponents = tuple(o.strip() for o in args.eval_opponents.split(",") if o.strip())
    if args.config_override:
        args.config_overrides = [kv.strip() for kv in args.config_override.split(",") if kv.strip()]
    else:
        args.config_overrides = None

    if args.list:
        _print_registry()
        return

    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    args.device = device

    games = _resolve_games(args.game)
    summaries: list[dict[str, Any]] = []
    for spec in games:
        requested = _resolve_solvers(args.solver, spec)
        if not requested:
            print(f"[跳过] {spec.game_id}: 没有命中的已登记求解器")
            continue
        summaries.append(train_game(spec, requested, args))
    print_summary(summaries)


if __name__ == "__main__":
    main()
