#!/usr/bin/env python3
"""Scratch probe: MCTS calibration with forced-decision skip (temp file)."""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train-cli"))

from layer2_engine.core.engine import GameEngine
from layer3_solvers.base import SolverConfig
from layer3_solvers.mahjong.heuristic import MahjongHeuristicAI
from layer3_solvers.mcts.solver import MCTS, MCTSConfig


def build_engine(variant: str) -> GameEngine:
    rules = json.loads((ROOT / "rules" / "mahjong.json").read_text(encoding="utf-8"))
    return GameEngine(rules, seed=42, variant=variant, player_count=4)


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "guangdong"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rollout_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    engine = build_engine(variant)
    rng = random.Random(7)
    heuristic = MahjongHeuristicAI(engine, SolverConfig(seed=0))
    mcts = MCTS(engine, MCTSConfig(budget=budget, seed=1, rollout_depth=rollout_depth))

    owners = {"p0": mcts, "p1": heuristic, "p2": heuristic, "p3": heuristic}
    state = engine.create_initial_state()
    steps = 0
    mdec = 0
    mtime = 0.0
    forced = 0
    real = 0
    t0 = time.perf_counter()
    while not engine.is_terminal(state) and steps < 800:
        node = engine.get_node_type(state)
        if node == "chance":
            outcomes = engine.get_chance_outcomes(state)
            probs = [float(getattr(o, "probability", 0.0) or 0.0) for o in outcomes]
            state = (
                engine.apply_chance(state, rng.choices(outcomes, weights=probs, k=1)[0])
                if sum(probs) > 0
                else engine.apply_chance(state, rng.choice(outcomes))
            )
            steps += 1
            continue
        if node != "player":
            break
        current = engine.get_current_player(state)
        solver = owners.get(current)
        legal = engine.get_legal_actions(state)
        if not legal:
            break
        if solver is not None and len(legal) == 1:
            # forced decision: take it directly (no search / no model call)
            action = legal[0]
            forced += 1
        elif solver is not None:
            t1 = time.perf_counter()
            action = solver.select_action(state)
            dt = time.perf_counter() - t1
            real += 1
            if isinstance(solver, MCTS):
                mdec += 1
                mtime += dt
            if action is None:
                action = rng.choice(legal)
        else:
            action = rng.choice(legal)
        state = engine.apply_action(state, action)
        steps += 1
    wall = time.perf_counter() - t0
    utils = {p: float(engine.get_utility(state, p)) for p in ("p0", "p1", "p2", "p3")}
    print(f"variant={variant} budget={budget} depth={rollout_depth}")
    print(f"  episode steps={steps} forced={forced} real(all)={real} mcts_decisions={mdec} "
          f"mcts_total={mtime:.1f}s per_dec={mtime / max(1, mdec):.2f}s wall={wall:.1f}s")
    print(f"  payoffs: {utils}")


if __name__ == "__main__":
    main()