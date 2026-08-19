#!/usr/bin/env python3
"""CFR Demo — trains and evaluates CFR on Stochastic Gomoku.

Usage:  python -m demos.demo_cfr [--iters N] [--size N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import clone_state
from layer3_solvers.base import SolverConfig
from layer3_solvers.cfr import CFR

SYMBOLS = {"p_black": "●", "p_white": "○", None: "·"}


def render_board(state):
    bs = state["board_size"]
    board = state["_board"]
    lines = ["   " + "".join(f"{i:2}" for i in range(bs))]
    for y in range(bs):
        row = f"{y:2} "
        for x in range(bs):
            row += " " + SYMBOLS.get(board[y * bs + x], "?")
        lines.append(row)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CFR Demo — Stochastic Gomoku")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--size", type=int, default=5, help="Board size (CFR works best on small boards)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cfr-plus", action="store_true")
    args = parser.parse_args()

    rules_path = Path(__file__).resolve().parent.parent / "rules" / "stochastic_gomoku.json"
    with open(rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    rules["constants"]["board_size"] = args.size

    engine = GameEngine(rules, seed=args.seed)
    state = engine.create_initial_state()

    print(f"CFR 训练 — 随机五子棋 {args.size}×{args.size}")
    print(f"迭代: {args.iters}  CFR+: {not args.no_cfr_plus}")
    print("═" * 50)

    cfr = CFR(
        engine,
        SolverConfig(
            seed=args.seed,
            verbose=True,
        ),
    )
    cfr.iterations = args.iters
    cfr.use_cfr_plus = not args.no_cfr_plus

    print("\n训练中...\n")
    t0 = time.perf_counter()
    strategy = cfr.solve(state, verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"\n训练完成: {elapsed:.1f}s  info_sets={len(cfr.info_sets)}")

    # Show strategy at root
    print("\n根节点策略:")
    sorted_strat = sorted(strategy.items(), key=lambda kv: kv[1], reverse=True)
    for key, prob in sorted_strat[:10]:
        bar = "█" * int(prob * 40)
        print(f"  {key:20s}  {prob:.4f}  {bar}")

    # Evaluate vs random
    print("\n评估: CFR vs Random, 100 局")
    import random

    rng = random.Random(args.seed)
    wins = losses = draws = 0
    for g in range(100):
        s = clone_state(state)
        while not engine.is_terminal(s):
            nt = engine.get_node_type(s)
            if nt == "player":
                cp = engine.get_current_player(s)
                if cp == "p_black":
                    a = cfr.select_action(s)
                else:
                    acts = engine.get_legal_actions(s)
                    a = rng.choice(acts) if acts else None
                if a is None:
                    break
                s = engine.apply_action(s, a)
            elif nt == "chance":
                _, s = engine.sample_chance(s)
            else:
                break
        w = s["env"].get("winner")
        if w == "p_black":
            wins += 1
        elif w == "p_white":
            losses += 1
        else:
            draws += 1
    print(f"  CFR 胜: {wins}  随机胜: {losses}  平: {draws}")


if __name__ == "__main__":
    main()
