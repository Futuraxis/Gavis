#!/usr/bin/env python3
"""Scratch probe: obs/action dims per mahjong variant + checkpoint load compat (temp)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train-cli"))

from layer2_engine.core.engine import GameEngine  # noqa: E402
from layer3_solvers.marl.action_space import ActionSpace  # noqa: E402
from layer3_solvers.marl.encoders import GameEncoder  # noqa: E402
from layer3_solvers.marl.maac import MAACConfig, MAACSolver  # noqa: E402

VARIANTS = ("guangdong", "hongzhong", "sichuan", "blood", "changsha", "taiwan")


def _player_state(engine: GameEngine):
    state = engine.create_initial_state()
    while engine.get_node_type(state) == "chance":
        outs = engine.get_chance_outcomes(state)
        if not outs:
            break
        state = engine.apply_chance(state, outs[0])
    return state


def main() -> None:
    rules = json.loads((ROOT / "rules" / "mahjong.json").read_text(encoding="utf-8"))
    players = ["p0", "p1", "p2", "p3"]
    dims = {}
    for v in VARIANTS:
        engine = GameEngine(rules, seed=42, variant=v, player_count=4)
        enc = GameEncoder.build_from_adapter(engine, players)
        asp = ActionSpace.build_from_adapter(engine)
        dims[v] = (enc.obs_dim, asp.dim, len(engine._constants["tile_ids"]))
        print(f"{v}: obs_dim={enc.obs_dim} action_dim={asp.dim} tiles={len(engine._constants['tile_ids'])}")

    ckpt = ROOT / "models" / "train" / "mahjong_guangdong" / "maac.pt"
    print("checkpoint exists:", ckpt.exists())
    if ckpt.exists():
        for v in ("guangdong", "hongzhong", "taiwan"):
            engine = GameEngine(rules, seed=42, variant=v, player_count=4)
            solver = MAACSolver(engine, MAACConfig(seed=42, device="cpu"))
            try:
                solver.load(str(ckpt))
                state = _player_state(engine)
                action = solver.select_action(state)
                legal_keys = {a.canonical_key for a in engine.get_legal_actions(state)}
                valid = action is not None and action.canonical_key in legal_keys
                print(f"load {v}: OK  valid_action={valid}  key={action.canonical_key if action else None}")
            except Exception as exc:  # noqa: BLE001
                print(f"load {v}: FAIL {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()