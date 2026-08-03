#!/usr/bin/env python3
"""Hybrid 模型训练 — 在三个游戏上训练 HybridSolver 并保存产物与评估指标.

每个游戏的训练内容由 ``HybridConfig`` 决定：
  - CFR 均衡先验（全部三个游戏）
  - PSRO 策略池（仅 moon_chess — ``GymAdapter`` 只兼容 3×3 格子游戏；
    stochastic_gomoku 9×9 状态空间过大、texas_holdem 非格子游戏，均不可行）

产物保存到 ``<out-dir>/<game>/``：
  - ``cfr_table.json``  CFR 策略表（可被 ``HybridConfig(cfr_table_path=...)`` 加载）
  - ``psro_pool.npz``   PSRO 策略池 + Nash 混合（仅 moon_chess）
  - ``config.json``     训练所用 HybridConfig
  - ``metrics.json``    训练耗时 + vs 随机对手评估（search 模式与 table 模式对比）

Usage:  python -m demos.train_hybrid [--game all|moon_chess|stochastic_gomoku|texas_holdem]
                                     [--cfr-iters N] [--cfr-depth N] [--psro-iters N]
                                     [--mcts-budget N] [--eval-episodes N] [--seed N]
                                     [--out-dir models/hybrid] [--skip-psro] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from layer2_engine.core.engine import GameEngine
from layer2_engine.games.moon_chess import MoonChessAdapter
from layer2_engine.games.texas_holdem import TexasHoldemAdapter
from layer3_solvers import HybridConfig, HybridSolver

GAMES = ("moon_chess", "stochastic_gomoku", "texas_holdem")

# 每游戏的玩家 id（先手在前），用于评估时双方轮流扮演 Hybrid。
GAME_PLAYERS = {
    "moon_chess": ["p_black", "p_white"],
    "stochastic_gomoku": ["p_black", "p_white"],
    "texas_holdem": ["p_sb", "p_bb"],
}

# 深度训练默认配置（按本机实测耗时标定，全部可用命令行覆盖）：
#   moon_chess:     CFR 800 iters ≈ 3 min,  PSRO 5×2000 ≈ 35 s
#   stochastic_gomoku: CFR 50 iters ≈ 5-8 min (9×9 外部采样在根节点展开 81 动作，深度受限)
#   texas_holdem:   CFR 1000 iters ≈ 75 s (对手模型搜索, imperfect_information)
DEFAULT_CONFIGS: dict[str, dict] = {
    "moon_chess": dict(
        cfr_iterations=800,
        cfr_depth_limit=6,
        psro_iters=5,
        psro_steps_per_iter=2000,
        mcts_budget=300,
        opponent_model="psro",
    ),
    "stochastic_gomoku": dict(
        cfr_iterations=50,
        cfr_depth_limit=5,
        mcts_budget=200,
        opponent_model="cfr",
    ),
    "texas_holdem": dict(
        cfr_iterations=1000,
        cfr_depth_limit=8,
        mcts_budget=300,
        opponent_model="cfr",
        imperfect_information=True,
    ),
}


def load_adapter(game: str, seed: int):
    """Load the appropriate game adapter/engine."""
    if game == "moon_chess":
        return MoonChessAdapter(seed=seed)
    if game == "stochastic_gomoku":
        rules_path = Path(__file__).resolve().parent.parent / "rules" / "stochastic_gomoku.json"
        with open(rules_path, "r", encoding="utf-8") as f:
            return GameEngine(json.load(f), seed=seed)
    return TexasHoldemAdapter(seed=seed)


def make_solver(adapter, game: str, args, cfr_table_path: str) -> HybridSolver:
    """Build the HybridSolver: game defaults + CLI overrides."""
    cfg = dict(DEFAULT_CONFIGS[game])
    if args.cfr_iters is not None:
        cfg["cfr_iterations"] = args.cfr_iters
    if args.cfr_depth is not None:
        cfg["cfr_depth_limit"] = args.cfr_depth
    if args.mcts_budget is not None:
        cfg["mcts_budget"] = args.mcts_budget
    if args.psro_iters is not None:
        cfg["psro_iters"] = args.psro_iters
    if args.skip_psro:
        cfg["opponent_model"] = "cfr"
    if game != "moon_chess":
        cfg.pop("psro_iters", None)
        cfg.pop("psro_steps_per_iter", None)
    return HybridSolver(adapter, HybridConfig(seed=args.seed, cfr_table_path=cfr_table_path, **cfg))


def resolve_chance(adapter, state: dict) -> dict:
    """Advance through pending chance nodes (deal, vanish, showdown)."""
    if hasattr(adapter, "resolve_chance"):
        return adapter.resolve_chance(state)
    while adapter.get_node_type(state) == "chance":
        _, state = adapter.sample_chance(state)
    return state


def play_episode(adapter, players: list[str], solver, hybrid_side: int, seed: int) -> float:
    """One episode: Hybrid plays as ``players[hybrid_side]`` vs uniform random.

    Returns the Hybrid-side utility (win=1 / draw=0 / loss=-1 for grid
    games, chip differential for poker).
    """
    rng = random.Random(seed)
    state = adapter.create_initial_state()
    hybrid_id = players[hybrid_side]
    guard = 0
    while not adapter.is_terminal(state) and guard < 200:
        nt = adapter.get_node_type(state)
        if nt == "chance":
            state = resolve_chance(adapter, state)
            guard += 1
            continue
        if nt != "player":
            break
        actions = adapter.get_legal_actions(state)
        if not actions:
            break
        if adapter.get_current_player(state) == hybrid_id:
            action = solver.select_action(state)
            if action is None:  # table 模式未覆盖的信息集 → 均匀随机回退
                action = rng.choice(actions)
        else:
            action = rng.choice(actions)
        state = adapter.apply_action(state, action)
        guard += 1
    return adapter.get_utility(state, hybrid_id)


def evaluate(adapter, players: list[str], solver, episodes: int, base_seed: int) -> dict:
    """Hybrid vs random, alternating sides.  Returns win/draw/loss + utility."""
    t0 = time.perf_counter()
    utils = [play_episode(adapter, players, solver, ep % 2, base_seed + ep) for ep in range(episodes)]
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
        "seconds": round(elapsed, 1),
    }


def train_game(game: str, args) -> Optional[dict]:
    """Train Hybrid for one game, save artifacts, evaluate, return summary."""
    print(f"\n{'█' * 60}")
    print(f"  训练 {game}")
    print(f"{'█' * 60}")

    adapter = load_adapter(game, args.seed)
    players = GAME_PLAYERS[game]
    out_dir = Path(args.out_dir) / game
    out_dir.mkdir(parents=True, exist_ok=True)
    cfr_path = out_dir / "cfr_table.json"

    # ── 训练（CFR 先验 + moon_chess 附加 PSRO 池）──
    solver = make_solver(adapter, game, args, str(cfr_path))  # train() 自动保存 CFR 表
    t0 = time.perf_counter()
    solver.train(1, verbose=args.verbose)
    train_seconds = round(time.perf_counter() - t0, 1)
    print(
        f"  训练完成: {train_seconds}s   "
        f"info_sets={len(solver.cfr.info_sets)}  "
        f"psro_pool={len(solver._pool) if solver._pool else '-'}"
    )

    # ── 保存产物 ──
    if solver._pool:
        solver.psro.save(str(out_dir / "psro_pool.npz"))
    config_dict = asdict(solver.config)
    config_dict["game"] = game
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    # ── 评估: search 模式 vs table 模式（纯 CFR 表）──
    metrics = {
        "game": game,
        "seed": args.seed,
        "train_seconds": train_seconds,
        "cfr_info_sets": len(solver.cfr.info_sets),
        "psro_pool_size": len(solver._pool) if solver._pool else None,
    }
    episodes = args.eval_episodes
    if episodes > 0:
        metrics["eval_search"] = evaluate(adapter, players, solver, episodes, args.seed)
        table = HybridSolver(adapter, HybridConfig(seed=args.seed, mode="table", cfr_table_path=str(cfr_path)))
        metrics["eval_table"] = evaluate(adapter, players, table, episodes, args.seed + 10_000)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def print_summary(results: list[dict]) -> None:
    """Print the cross-game summary table."""
    print(f"\n{'█' * 60}")
    print("  训练总结")
    print(f"{'█' * 60}")
    print(f"  {'游戏':18s} {'info_sets':9s} {'PSRO池':6s} {'search胜率':10s} {'table胜率':10s} {'训练(s)':8s}")
    print(f"  {'─' * 64}")
    for m in results:
        s = m.get("eval_search", {}).get("win_rate")
        t = m.get("eval_table", {}).get("win_rate")
        pool = m.get("psro_pool_size")
        print(
            f"  {m['game']:18s} {m['cfr_info_sets']:9d} "
            f"{pool if pool is not None else '-':>6} "
            f"{s if s is not None else '-':>10} "
            f"{t if t is not None else '-':>10} "
            f"{m['train_seconds']:8.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Gavis Hybrid 模型训练 — 三个游戏训练 + 评估 + 保存产物",
    )
    parser.add_argument("--game", type=str, default="all", choices=["all", *GAMES])
    parser.add_argument("--cfr-iters", type=int, default=None, help="CFR 迭代数（默认按游戏深度档）")
    parser.add_argument("--cfr-depth", type=int, default=None, help="CFR 深度限制（默认按游戏）")
    parser.add_argument("--psro-iters", type=int, default=None, help="PSRO 迭代数（仅 moon_chess）")
    parser.add_argument("--mcts-budget", type=int, default=None, help="在线搜索预算")
    parser.add_argument("--eval-episodes", type=int, default=20, help="评估局数（双方各半；0 = 跳过评估）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="models/hybrid")
    parser.add_argument("--skip-psro", action="store_true", help="moon_chess 不训练 PSRO 池")
    parser.add_argument("--verbose", action="store_true", help="显示 CFR/PSRO 训练进度")
    args = parser.parse_args()

    games = GAMES if args.game == "all" else (args.game,)
    results = []
    for game in games:
        metrics = train_game(game, args)
        if metrics is not None:
            results.append(metrics)
    print_summary(results)


if __name__ == "__main__":
    main()
