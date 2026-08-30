#!/usr/bin/env python3
"""Probe: do natural mahjong games end (wall draw / win) within N steps? (temp)"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train-cli"))

from layer2_engine.core.engine import GameEngine  # noqa: E402


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "guangdong"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    rules = json.loads((ROOT / "rules" / "mahjong.json").read_text(encoding="utf-8"))
    engine = GameEngine(rules, seed=1, variant=variant, player_count=4)
    for ep in range(3):
        rng = random.Random(100 + ep)
        state = engine.create_initial_state()
        steps = 0
        t0 = time.perf_counter()
        trunc = False
        while not engine.is_terminal(state) and steps < cap:
            node = engine.get_node_type(state)
            if node == "chance":
                outs = engine.get_chance_outcomes(state)
                if not outs:
                    break
                probs = [float(getattr(o, "probability", 0.0) or 0.0) for o in outs]
                state = (
                    engine.apply_chance(state, rng.choices(outs, weights=probs, k=1)[0])
                    if sum(probs) > 0
                    else engine.apply_chance(state, rng.choice(outs))
                )
                steps += 1
                continue
            if node != "player":
                break
            legal = engine.get_legal_actions(state)
            if not legal:
                break
            action = legal[0] if len(legal) == 1 else rng.choice(legal)
            state = engine.apply_action(state, action)
            steps += 1
        if not engine.is_terminal(state):
            trunc = steps >= cap
        utils = [float(engine.get_utility(state, p)) for p in ("p0", "p1", "p2", "p3")]
        print(
            f"{variant} ep{ep} steps={steps} terminal={engine.is_terminal(state)} "
            f"truncated={trunc} utils={utils} wall={state.get('env', {}).get('wall_count')} "
            f"el={time.perf_counter() - t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()