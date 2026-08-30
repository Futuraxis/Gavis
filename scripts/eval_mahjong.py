#!/usr/bin/env python3
"""scripts/eval_mahjong.py — reusable 4-player mahjong match evaluation.

Matches two solver handles against each other on a mahjong variant with
seat rotation and reports win rates / utilities:

- ``1v3`` mode (default): A occupies one rotating seat; B drives the
  other three (the standard "does A beat B at mahjong" test).
- ``2v2`` mode: A drives seats 0-1 and B drives 2-3, the pairs swapped
  on alternating episodes to cancel first-seat advantage.

Solver names: ``mahjong`` (heuristic), ``random``, ``mcts`` (budget +
rollout depth configurable), ``maac`` (trained checkpoint via ``--model``;
refuses to run untrained unless ``--allow-untrained``).

Forced decisions (exactly one legal action — e.g. no-choice ``claim_pass``
states) are executed directly without calling the solver: ~75% of a
mahjong game's steps are forced, so this cuts evaluation wall time by an
order of magnitude while staying semantically identical.

Winner rule: exactly one player has positive utility on a won hand
(winner +fan pay, losers negative); all-zero payoffs = wall/rule draw.

Usage::

    python scripts/eval_mahjong.py --variant guangdong --episodes 10 \
        --solver-a maac --solver-b mcts --b-budget 10 --rollout-depth 8 \
        --model models/train/mahjong_guangdong/maac.pt
    python scripts/eval_mahjong.py --variant guangdong --episodes 8 \
        --solver-a maac --solver-b mahjong \
        --model models/train/mahjong_guangdong/maac.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "train-cli"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from layer2_engine.core.engine import GameEngine  # noqa: E402
from layer3_solvers.base import SolverBase, SolverConfig  # noqa: E402
from layer3_solvers.mahjong.heuristic import MahjongHeuristicAI  # noqa: E402
from layer3_solvers.marl.maac import MAACConfig, MAACSolver  # noqa: E402
from layer3_solvers.mcts.solver import MCTS, MCTSConfig  # noqa: E402

# 自然牌局以墙尽（~1000-1100 步）或胡牌结束；上限取 1400 避免把晚胡/墙尽
# 截断成误判平局（旧 800 会把 ~20% 的墙尽局截短）。
MAX_STEPS = 1400
SEATS = ("p0", "p1", "p2", "p3")


def build_engine(variant: str, seed: int) -> GameEngine:
    """Fresh engine per game (distinct deals across episodes)."""
    rules = json.loads((_ROOT / "rules" / "mahjong.json").read_text(encoding="utf-8"))
    return GameEngine(rules, seed=seed, variant=variant, player_count=4)


def play_match(
    engine: GameEngine,
    owners: dict[str, SolverBase],
    rng: random.Random,
    max_steps: int = MAX_STEPS,
) -> tuple[int, dict[str, float]]:
    """Play one game; forced (single-legal) decisions skip the solver."""
    state = engine.create_initial_state()
    steps = 0
    while not engine.is_terminal(state) and steps < max_steps:
        node = engine.get_node_type(state)
        if node == "chance":
            outcomes = engine.get_chance_outcomes(state)
            if not outcomes:
                break
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
        legal = engine.get_legal_actions(state)
        if not legal:
            break
        if len(legal) == 1:
            action = legal[0]
        else:
            solver = owners.get(current)
            action = solver.select_action(state) if solver is not None else None
            if action is None:
                action = rng.choice(legal)
        state = engine.apply_action(state, action)
        steps += 1
    payoffs = {p: float(engine.get_utility(state, p)) for p in owners}
    return steps, payoffs


def winner_of(payoffs: dict[str, float]) -> str | None:
    """Positive-utility sole winner; None on a draw (all payoffs zero)."""
    winners = [p for p, u in payoffs.items() if u > 0]
    return winners[0] if len(winners) == 1 else None


def run_arena(
    variant: str,
    make_a: Callable[[GameEngine, int], SolverBase],
    make_b: Callable[[GameEngine, int], SolverBase],
    episodes: int,
    base_seed: int,
    mode: str,
) -> dict:
    """Run the arena; returns aggregate results (JSON-serializable)."""
    a_wins = b_wins = draws = 0
    a_utils: list[float] = []
    b_utils: list[float] = []
    per_ep: list[dict] = []
    t0 = time.perf_counter()
    for ep in range(episodes):
        engine = build_engine(variant, base_seed + ep * 101)
        solver_a = make_a(engine, base_seed + ep * 101 + 1)
        solver_b = make_b(engine, base_seed + ep * 101 + 2)
        rng = random.Random(base_seed + ep * 31 + 7)
        if mode == "1v3":
            a_seat = SEATS[ep % 4]
            owners = {s: (solver_a if s == a_seat else solver_b) for s in SEATS}
            a_team = {a_seat}
        else:  # 2v2, pairs swapped per episode
            a_first = ep % 2 == 0
            a_team = set(SEATS[:2]) if a_first else set(SEATS[2:])
            owners = {s: (solver_a if s in a_team else solver_b) for s in SEATS}
        steps, payoffs = play_match(engine, owners, rng)
        w = winner_of(payoffs)
        if w is None:
            draws += 1
        elif w in a_team:
            a_wins += 1
        else:
            b_wins += 1
        b_team = set(SEATS) - a_team
        a_utils.append(sum(payoffs[s] for s in a_team) / len(a_team))
        b_utils.append(sum(payoffs[s] for s in b_team) / len(b_team))
        per_ep.append({"ep": ep, "steps": steps, "winner": w, "payoffs": payoffs})
    wall = time.perf_counter() - t0
    return {
        "variant": variant,
        "mode": mode,
        "episodes": episodes,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_win_rate": round(a_wins / episodes, 4),
        "b_win_rate": round(b_wins / episodes, 4),
        "a_avg_utility": round(sum(a_utils) / episodes, 4),
        "b_avg_utility": round(sum(b_utils) / episodes, 4),
        "seconds": round(wall, 2),
        "per_episode": per_ep,
    }


def _make_solver(
    name: str,
    budget: int,
    rollout_depth: int,
    model: Path | None,
    allow_untrained: bool,
) -> Callable[[GameEngine, int], SolverBase]:
    def factory(engine: GameEngine, seed: int) -> SolverBase:
        if name == "mahjong":
            return MahjongHeuristicAI(engine, SolverConfig(seed=seed))
        if name == "random":
            import games  # noqa: F401  # train-cli registry (train-cli/ on sys.path)

            return games.RandomSolver(engine, seed)
        if name == "mcts":
            return MCTS(engine, MCTSConfig(budget=budget, seed=seed, rollout_depth=rollout_depth))
        if name == "maac":
            if model is not None and model.exists():
                solver = MAACSolver(engine, MAACConfig(seed=seed, device="cpu"))
                solver.load(str(model))
                return solver
            if allow_untrained:
                return MAACSolver(engine, MAACConfig(seed=seed, device="cpu"))
            raise SystemExit(f"MAAC 模型不存在: {model}（需要先训练，或加 --allow-untrained 用未训练模型）")
        raise SystemExit(f"未知求解器: {name}")
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Mahjong 4-player match arena")
    parser.add_argument("--variant", default="guangdong")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["1v3", "2v2"], default="1v3")
    parser.add_argument("--solver-a", default="maac")
    parser.add_argument("--solver-b", default="mcts")
    parser.add_argument("--a-budget", type=int, default=10, help="MCTS budget when solver==mcts")
    parser.add_argument("--b-budget", type=int, default=10, help="MCTS budget when solver==mcts")
    parser.add_argument("--rollout-depth", type=int, default=8)
    parser.add_argument("--model", default=None, help="MAAC checkpoint path")
    parser.add_argument("--allow-untrained", action="store_true")
    args = parser.parse_args()

    model = Path(args.model) if args.model else None
    make_a = _make_solver(args.solver_a, args.a_budget, args.rollout_depth, model, args.allow_untrained)
    make_b = _make_solver(args.solver_b, args.b_budget, args.rollout_depth, model, args.allow_untrained)
    result = run_arena(args.variant, make_a, make_b, args.episodes, args.seed, args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()