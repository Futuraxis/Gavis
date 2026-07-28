#!/usr/bin/env python3
"""Unified Benchmark — runs all four solvers on the same game and compares.

Usage:  python -m demos.benchmark_all [--game moon_chess|stochastic_gomoku] [--episodes N]

This is the "一键横评" entry point that validates the Layer 2→3 integration.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from layer2_engine.core.engine import GameEngine
from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter
from layer3_solvers import (
    MCTS,
    CFR,
    PPOSolver,
    PSROSolver,
    SolverBase,
    SolverConfig,
    MCTSConfig,
    CFRConfig,
    PPOConfig,
    PSROConfig,
)


def load_engine(game: str, seed: int = 42):
    """Load the appropriate game engine."""
    if game == 'stochastic_gomoku':
        rules_path = Path(__file__).resolve().parent.parent / 'rules' / 'stochastic_gomoku.json'
        with open(rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        return GameEngine(rules, seed=seed), {}
    elif game == 'moon_chess':
        adapter = MoonChessAdapter(seed=seed)
        versions = {'feature_dim': 38, 'action_dim': 9}
        return adapter, versions
    else:
        raise ValueError(f"Unknown game: {game}")


def create_solver(name: str, engine, extras: dict) -> SolverBase:
    """Factory to create a solver by name."""
    if name == 'mcts':
        return MCTS(engine, MCTSConfig(seed=42, budget=3000))
    elif name == 'cfr':
        return CFR(engine, CFRConfig(seed=42, iterations=500))
    elif name == 'ppo':
        cfg = PPOConfig(
            seed=42,
            state_dim=extras.get('feature_dim', 38),
            action_dim=extras.get('action_dim', 9),
        )
        return PPOSolver(engine, cfg)
    elif name == 'psro':
        return PSROSolver(engine, PSROConfig(seed=42, num_iters=5, num_steps_per_iter=2000))
    else:
        raise ValueError(f"Unknown solver: {name}")


def benchmark_one(solver, engine, episodes: int, label: str) -> dict:
    """Run one solver and return metrics."""
    print(f'\n{"─"*50}')
    print(f'  {label} ({solver.name})')
    print(f'{"─"*50}')

    # Quick play test
    state = engine.create_initial_state()
    moves = 0
    t0 = time.time()
    while not engine.is_terminal(state) and moves < 50:
        nt = engine.get_node_type(state)
        if nt == 'player':
            action = solver.select_action(state)
            if action is None:
                break
            state = engine.apply_action(state, action)
            moves += 1
        elif nt == 'chance':
            _, state = engine.sample_chance(state)
        else:
            break
    elapsed = time.time() - t0

    winner = state['env'].get('winner')
    result = {
        'solver': label,
        'moves': moves,
        'elapsed_s': round(elapsed, 3),
        'winner': winner,
        'avg_s_per_move': round(elapsed / max(1, moves), 4),
    }
    print(f'    步数: {moves}  |  耗时: {elapsed:.2f}s  |  胜者: {winner}')
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Gavis Unified Benchmark — compare all solvers on the same game',
    )
    parser.add_argument('--game', type=str, default='moon_chess',
                        choices=['moon_chess', 'stochastic_gomoku'])
    parser.add_argument('--episodes', type=int, default=10,
                        help='Training episodes (only affects PPO/PSRO)')
    parser.add_argument('--solvers', type=str, nargs='+',
                        default=['mcts', 'cfr', 'ppo', 'psro'],
                        choices=['mcts', 'cfr', 'ppo', 'psro'])
    args = parser.parse_args()

    print(f'\n{"█"*60}')
    print(f'  Gavis 统一基准评测')
    print(f'  游戏: {args.game}  求解器: {", ".join(args.solvers)}')
    print(f'{"█"*60}')

    engine, extras = load_engine(args.game)
    results = []

    # Run MCTS first (no training needed)
    if 'mcts' in args.solvers:
        solver = create_solver('mcts', engine, extras)
        results.append(benchmark_one(solver, engine, args.episodes, 'MCTS'))

    # Run CFR
    if 'cfr' in args.solvers:
        solver = create_solver('cfr', engine, extras)
        # Need to train first
        print(f'\n  训练 CFR...')
        state = engine.create_initial_state()
        solver.solve(state, verbose=False)
        results.append(benchmark_one(solver, engine, args.episodes, 'CFR'))

    # Run PPO
    if 'ppo' in args.solvers:
        solver = create_solver('ppo', engine, extras)
        print(f'\n  训练 PPO ({args.episodes} episodes)...')
        solver.train(episodes=args.episodes, verbose=False)
        results.append(benchmark_one(solver, engine, args.episodes, 'PPO'))

    # Run PSRO
    if 'psro' in args.solvers:
        solver = create_solver('psro', engine, extras)
        print(f'\n  训练 PSRO ({args.episodes} iters)...')
        solver.train(episodes=args.episodes, verbose=False)
        results.append(benchmark_one(solver, engine, args.episodes, 'PSRO'))

    # Summary
    print(f'\n{"█"*60}')
    print(f'  评测总结')
    print(f'{"█"*60}')
    print(f'  {"求解器":12s} {"步数":6s} {"耗时(s)":10s} {"秒/步":8s} {"胜者":8s}')
    print(f'  {"─"*44}')
    for r in results:
        print(f'  {r["solver"]:12s} {r["moves"]:6d} {r["elapsed_s"]:10.3f} {r["avg_s_per_move"]:8.4f} {str(r["winner"] or "平局"):8s}')


if __name__ == '__main__':
    main()
