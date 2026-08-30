"""Botzone JSON adapter for Gavis.

Botzone runs a bot as one stdin read and one stdout write.  The platform
input is a JSON object containing prior ``requests`` / ``responses`` plus
optional per-match storage.  This adapter keeps that protocol at the
edge of Layer 4 and calls the existing data-driven runtime registry.

Supported current request schema (JSON object, or a JSON string holding
the object):

```
{
  "game_id": "moon_chess",
  "player_id": "p_black",
  "solver": "mcts",
  "seed": 42,
  "budget": 800,
  "state": { ... Gavis engine state ... }
}
```

If ``state`` is omitted, a fresh initial state is created and all initial
chance nodes are resolved.  This is enough for smoke testing and for
Botzone games whose judge sends the full current state each turn.  Games
with proprietary request formats can add a narrow request-to-state
translator here without touching Layers 1-3.
"""

from __future__ import annotations

import json
import sys
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance
from layer4_interface.botzone.mahjong_format import decide_mahjong_format, is_mahjong_format_payload
from layer4_interface.botzone.texas_holdem import decide_texas_holdem, is_texas_holdem_payload
from layer4_interface.frontend.engine_helpers import resolve_all_chance


class BotzoneError(Exception):
    """Invalid Botzone input or unsupported request shape."""


@dataclass(frozen=True)
class BotzoneDecision:
    """One decision plus compact debug text for Botzone logs."""

    response: Any
    debug: str
    data: str = ""
    globaldata: str = ""

    def to_envelope(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "debug": self.debug[:1024],
            "data": self.data,
            "globaldata": self.globaldata,
        }


def main(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    """CLI entrypoint used by ``python -m layer4_interface.botzone``."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    try:
        payload = _read_botzone_input(stdin)
        decision = decide(payload)
        print(json.dumps(decision.to_envelope(), ensure_ascii=False, separators=(",", ":")), file=stdout)
    except Exception as exc:  # Botzone requires a single output line even on failure.
        debug = f"{type(exc).__name__}: {exc}"
        if _debug_trace_enabled():
            debug = (debug + "\n" + traceback.format_exc())[:1024]
        envelope = {"response": None, "debug": debug[:1024], "data": "", "globaldata": ""}
        print(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")), file=stdout)


def decide(payload: dict[str, Any]) -> BotzoneDecision:
    """Return one solver decision from a Botzone JSON interaction object."""
    if is_mahjong_format_payload(payload):
        response, debug = decide_mahjong_format(payload)
        return BotzoneDecision(response=response, debug=debug)
    if is_texas_holdem_payload(payload):
        response, debug = decide_texas_holdem(payload)
        return BotzoneDecision(response=response, debug=debug)

    current = _current_request(payload)
    game_id = _required_str(current, "game_id")
    player_id = str(current.get("player_id") or current.get("player_pid") or "")
    solver_name = str(current.get("solver") or _default_solver(game_id))
    seed = int(current.get("seed", 42))
    budget = int(current.get("budget", _default_budget(game_id, solver_name)))

    engine, canonical_game_id = _build_engine(game_id, current, seed)
    state = _load_state(engine, current)
    if engine.get_node_type(state) == "chance":
        state = resolve_all_chance(engine, state)
    if engine.is_terminal(state):
        return BotzoneDecision(response=None, debug=f"{canonical_game_id}: terminal")

    from train_cli import default_provider

    solver = default_provider.create_solver(canonical_game_id, solver_name, engine, seed, budget)
    action = solver.select_action(state)
    if action is None:
        return BotzoneDecision(response=None, debug=f"{canonical_game_id}: no legal action")
    response = _serialize_action(action)
    if player_id:
        response["player_id"] = player_id
    return BotzoneDecision(response=response, debug=f"{canonical_game_id}/{solver.name}:{action.canonical_key}")


def _read_botzone_input(stdin: TextIO) -> dict[str, Any]:
    text = stdin.read().strip()
    if not text:
        raise BotzoneError("empty stdin")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = _read_simplified_input(text)
    if not isinstance(value, dict):
        raise BotzoneError("Botzone input must be a JSON object")
    return value


def _read_simplified_input(text: str) -> dict[str, Any]:
    """Best-effort support for Botzone's simplified line protocol."""
    lines = text.splitlines()
    if not lines:
        raise BotzoneError("empty simplified input")
    try:
        turn = int(lines[0].strip())
    except ValueError as exc:
        raise BotzoneError("stdin is neither JSON nor simplified Botzone input") from exc
    history = lines[1 : 1 + max(0, 2 * turn - 1)]
    requests = history[0::2]
    responses = history[1::2]
    data = lines[1 + max(0, 2 * turn - 1)] if len(lines) > 1 + max(0, 2 * turn - 1) else ""
    globaldata = "\n".join(lines[2 + max(0, 2 * turn - 1) :])
    return {"requests": requests, "responses": responses, "data": data, "globaldata": globaldata}


def _current_request(payload: dict[str, Any]) -> dict[str, Any]:
    requests = payload.get("requests")
    if isinstance(requests, list) and requests:
        raw = requests[-1]
    else:
        raw = payload.get("request", payload)
    value = _decode_maybe_json(raw)
    if not isinstance(value, dict):
        raise BotzoneError("current request must be a JSON object")
    return value


def _decode_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": value}


def _build_engine(game_id: str, request: dict[str, Any], seed: int) -> tuple[GameEngine, str]:
    """Build an engine through the runtime registry when possible."""
    from train_cli import GAMES

    canonical = _canonical_game_id(game_id, request)
    spec = GAMES.get(canonical)
    if spec is not None:
        rules = _load_rules_json(spec.engine.rules)
        kwargs: dict[str, Any] = {}
        if spec.engine.variant is not None:
            kwargs["variant"] = spec.engine.variant
        if spec.engine.player_count is not None:
            kwargs["player_count"] = spec.engine.player_count
        return GameEngine(rules, seed=seed, **kwargs), canonical

    # Last-mile convenience for raw rules users.
    rules = request.get("rules_json")
    if isinstance(rules, dict):
        return GameEngine(
            rules,
            seed=seed,
            variant=request.get("variant"),
            player_count=request.get("player_count"),
        ), canonical
    raise BotzoneError(f"unknown game_id: {game_id}")


def _load_rules_json(file_name: str) -> dict[str, Any]:
    """Load rules both from a source checkout and from a zipapp bundle."""
    source_root = Path(__file__).resolve().parents[2]
    source_path = source_root / "rules" / file_name
    if source_path.is_file():
        with open(source_path, encoding="utf-8") as f:
            return json.load(f)

    bundle = _bundle_path()
    if bundle is not None and bundle.is_file():
        with zipfile.ZipFile(bundle) as zf:
            with zf.open(f"rules/{file_name}") as f:
                return json.loads(f.read().decode("utf-8"))
    raise BotzoneError(f"rules file not found: {file_name}")


def _bundle_path() -> Path | None:
    file_name = str(__file__)
    marker = ".zip/"
    if marker in file_name:
        return Path(file_name.split(marker, 1)[0] + ".zip")
    candidate = Path(sys.path[0])
    return candidate if candidate.suffix == ".zip" else None


def _canonical_game_id(game_id: str, request: dict[str, Any]) -> str:
    if game_id == "mahjong":
        variant = request.get("variant", "guangdong")
        aliases = {
            "guangdong": "mahjong_guangdong",
            "hongzhong": "mahjong_hongzhong",
            "blood": "mahjong_blood",
            "international": "mahjong_international",
        }
        return aliases.get(str(variant), str(variant))
    return game_id


def _load_state(engine: GameEngine, request: dict[str, Any]) -> dict[str, Any]:
    raw_state = request.get("state")
    if raw_state is None:
        return engine.create_initial_state()
    state = _decode_maybe_json(raw_state)
    if not isinstance(state, dict):
        raise BotzoneError("state must be a JSON object")
    return engine.load_state(state)


def _serialize_action(action: ActionInstance) -> dict[str, Any]:
    return {
        "template_id": action.template_id,
        "type": action.type,
        "params": action.params,
        "canonical_key": action.canonical_key,
    }


def _required_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise BotzoneError(f"missing required field: {key}")
    return value


def _default_solver(game_id: str) -> str:
    if game_id.startswith("mahjong") or game_id == "mahjong":
        return "mahjong"
    if game_id == "texas_holdem":
        return "hybrid"
    return "mcts"


def _default_budget(game_id: str, solver_name: str) -> int:
    if solver_name in {"random", "mahjong", "ollama"}:
        return 1
    if game_id == "texas_holdem":
        return 300
    if game_id.startswith("stochastic_gomoku"):
        return 500
    return 300


def _debug_trace_enabled() -> bool:
    return "--trace" in sys.argv


if __name__ == "__main__":
    main()
