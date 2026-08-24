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
训练后按注册表 ``eval`` 设置运行 vs 均匀随机的通用评估（座位轮换）。

产物（``<out-dir>/<game>/``）：
  - ``{save 名}``         各求解器产物（如 qmix.pt / cfr 表由 $OUTDIR 路径落盘）
  - ``config.json``       引擎/座位/各管线配置（可复现）
  - ``metrics.json``      每求解器训练指标 + 评估指标 + 耗时

Usage::

    python train-cli/train.py [--game all|moon_chess|stochastic_gomoku|texas_holdem|
                                      mahjong_guangdong|mahjong_hongzhong|mahjong_blood|werewolf]
                              [--solver all|hybrid|cfr|ppo|psro|qmix|happo|maac|bayes]
                              [--episodes N] [--seed N] [--device auto|cpu|cuda]
                              [--out-dir models/train] [--eval-episodes N]
                              [--skip-eval] [--list] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

# ── 路径引导（train-cli 目录含连字符，不能作为包导入；本文件可直接执行）──
_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from games import GAMES, SOLVER_FACTORY, GameSpec, SolverPipeline  # noqa: E402

from layer2_engine.core.engine import GameEngine  # noqa: E402
from layer3_solvers.base import SolverBase, SolverMetrics  # noqa: E402

#: 评估对局最大步数护栏（超出按当前效用截断，防死循环）。
MAX_EVAL_STEPS = 600


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


def make_solver(
    name: str,
    engine: GameEngine,
    pipeline: SolverPipeline,
    seed: int,
    device: str,
    out_dir: Path,
    player_id: str | None = None,
) -> SolverBase:
    """按注册表 SOLVER_FACTORY 实例化求解器（数据驱动，无 per-game 分支）。"""
    entry = SOLVER_FACTORY[name]
    if entry is None or entry[0] is None:
        raise RuntimeError(f"求解器 {name} 需要可选依赖（torch/psro extra），当前环境不可用——请安装后重试")
    cls, cfg_cls = entry
    kwargs = dict(pipeline.config or {})
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
        if not legal or solver is None:
            break
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
) -> dict[str, Any]:
    """通用评估：每局只有一个"own 座位"由被评求解器执掌（顺次轮换），其余均匀随机。

    - ``solver`` != None（普通管线）→ own 座位用该实例。
    - ``per_player_instances`` != None → own 座位用对应座位的实例（player_id 绑定）。
    """
    t0 = time.perf_counter()
    utils: list[float] = []
    for ep in range(episodes):
        rng = random.Random(base_seed + ep * 31 + 7)
        owned = ep % len(spec.players)
        owners: dict[str, SolverBase | None] = {}
        for i, seat in enumerate(spec.players):
            if per_player_instances is not None:
                owners[seat] = per_player_instances[i] if i == owned else None
            else:
                owners[seat] = solver if i == owned else None
        _, payoffs = play_episode(engine, owners, rng)
        utils.append(payoffs.get(spec.players[owned], 0.0))
    elapsed = time.perf_counter() - t0
    wins = sum(u > 0 for u in utils)
    draws = sum(u == 0 for u in utils)
    return {
        "episodes": episodes,
        "wins": wins,
        "draws": draws,
        "losses": episodes - wins - draws,
        "win_rate": round(wins / episodes, 4),
        "avg_utility": round(sum(utils) / episodes, 4),
        "seconds": round(elapsed, 2),
    }


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
    episodes = args.episodes if args.episodes is not None else pipeline.episodes
    print(f"\n── 求解器 {name} @ {spec.game_id}  ({pipeline.entry}, {episodes} 局) ──")
    t0 = time.perf_counter()

    if pipeline.per_player:
        instances = [
            make_solver(name, engine, pipeline, args.seed, args.device, out_dir, player_id=seat)
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
        solver = make_solver(name, engine, pipeline, args.seed, args.device, out_dir)
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
        n = args.eval_episodes or spec.eval_episodes
        if pipeline.per_player:
            record["eval"] = evaluate(engine, spec, None, instances, n, args.seed)
        else:
            record["eval"] = evaluate(engine, spec, solver, None, n, args.seed)
        print(
            f"  评估 vs 随机: win_rate={record['eval']['win_rate']:.3f}  "
            f"avg_utility={record['eval']['avg_utility']:+.3f}"
        )
    print(f"  训练完成: {record['train_seconds']}s")
    return record


def train_game(spec: GameSpec, requested: list[str], args: argparse.Namespace) -> dict[str, Any]:
    """按注册表训练一个游戏；返回汇总指标。"""
    print(f"\n{'█' * 62}")
    print(f"  {spec.display_name} ({spec.game_id})  players={list(spec.players)}")
    print(f"{'█' * 62}")

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
    print(f"\n{'█' * 62}")
    print("  训练总结")
    print(f"{'█' * 62}")
    for summary in summaries:
        solvers = [k for k in summary if k not in ("game", "display_name")]
        if not solvers:
            continue
        parts = []
        for name in solvers:
            rec = summary[name]
            wr = rec.get("eval", {}).get("win_rate")
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
    parser.add_argument("--eval-episodes", type=int, default=0, help="覆盖评估局数（0=注册表默认）")
    parser.add_argument("--skip-eval", action="store_true", help="跳过 vs 随机评估")
    parser.add_argument("--list", action="store_true", help="打印注册表一览并退出")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

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
