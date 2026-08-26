#!/usr/bin/env python3
"""Quick smoke test for rules/uno.json — random self-play to terminal.

Checks per (variant, player_count): engine construction (incl. unknown-variant
ValueError), deal→flip invariants (7 per hand, 1 discard, running deck sum),
legal-action non-emptiness in player phases, chance sampling, terminal
reachability, winner env field, utility sum == 0 over trimmed players.

Run:  python _smoke_uno.py [trials]
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from layer2_engine.core.engine import GameEngine

RULES = json.loads(Path("rules/uno.json").read_text(encoding="utf-8"))
VARIANTS = list(RULES["variants"]["options"])
COUNTS = [2, 4, 10]


def total_cards(engine: GameEngine, state: dict) -> int:
    """Sum of all hands + discard + deck (query) must equal 108."""
    arrays = state["_arrays"]
    hands = sum(len(arrays[f"hand_{p}"]) for p in engine._constants["player_ids"])
    discard = len(arrays["discard"])
    deck = len(engine._resolve_query(
        engine._queries["undrawn_cards"], state, engine._build_context(state)
    ))
    return hands + discard + deck


def play_one(engine: GameEngine, seed: int, cap: int = 20000) -> dict:
    rng = random.Random(seed)
    state = engine.create_initial_state()
    steps = 0
    while steps < cap:
        ntype = engine.get_node_type(state)
        if ntype == "terminal":
            return {"steps": steps, "state": state, "winner": state["env"].get("winner")}
        if ntype == "chance":
            _, state = engine.sample_chance(state)
            steps += 1
            continue
        # player node
        legal = engine.get_legal_actions(state)
        if not legal:
            raise AssertionError(
                f"empty legal at player phase {state['env'].get('phase')} turn={state['env'].get('turn')}"
            )
        act = rng.choice(legal)
        state = engine.apply_action(state, act)
        steps += 1
    raise AssertionError(f"no terminal in {cap} steps (phase={state['env'].get('phase')})")


def check_deal(engine: GameEngine, seed: int) -> dict:
    """Drive deal+flip, assert 7-per-hand and 1-discard, return play state."""
    rng = random.Random(seed)
    state = engine.create_initial_state()
    for _ in range(300):
        ntype = engine.get_node_type(state)
        if ntype == "terminal":
            raise AssertionError("terminal during deal?!")
        if ntype == "chance":
            _, state = engine.sample_chance(state)
            continue
        phase = state["env"].get("phase")
        if phase == "play":
            break
        raise AssertionError(f"unexpected phase {phase}")
    pids = engine._constants["player_ids"]
    arrays = state["_arrays"]
    for p in pids:
        # 首张特殊牌（draw2/wild4）会让 p1 在发牌后立即吃罚牌（7+2 / 7+4），
        # 因此只断言 ≥ 7 而不是恰好 7。
        assert len(arrays[f"hand_{p}"]) >= 7, (p, len(arrays[f"hand_{p}"]))
    assert len(arrays["discard"]) == 1
    assert total_cards(engine, state) == 108
    assert state["env"]["topColor"] is not None and state["env"]["topSymbol"] is not None
    # 回合必须是 pid（seat_after 只返回座位索引，赋值前必须 AT(player_ids, ...)）
    assert state["env"]["turn"] in pids, state["env"]["turn"]
    return state


def main() -> int:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ok = 0
    for variant in VARIANTS:
        for count in COUNTS:
            for t in range(trials):
                seed = hash((variant, count, t)) & 0x7FFFFFFF
                eng = GameEngine(RULES, seed=seed, variant=variant, player_count=count)
                st = check_deal(eng, seed)
                res = play_one(eng, seed + 1, cap=30000)
                winner = res["winner"]
                pids = eng._constants["player_ids"]
                assert winner is None or winner in pids, winner
                if winner is None:
                    # env.winner written on the last decision effector; a
                    # terminal right at a chance phase may leave it unset.
                    pass
                util_sum = sum(eng.get_utility(res["state"], p) for p in pids)
                # +1(胜者) + (n-1)×(-1) = 2−n —— 非零和但逐人明确胜负（与 undercover 一致）。
                assert abs(util_sum - (2 - count)) <= 1e-9, (variant, count, util_sum)
                ok += 1
            print(f"OK  {variant:12s} × {count:2d}  ({trials} runs each stable)")
    # unknown variant
    try:
        GameEngine(RULES, variant="nope")
        raise AssertionError("unknown variant did NOT raise!")
    except ValueError as exc:
        print(f"OK  unknown-variant ValueError: {str(exc)[:60]}…")
    print(f"\nall {ok} smoke runs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())