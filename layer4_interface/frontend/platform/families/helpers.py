"""Shared utilities for custom-game family modules.

Everything a family needs to wire a translated ``rules`` dict into a
``GameSpec`` lives here: bare engine construction (with the A1
``allow_codegen`` switch), generic chance resolution, grid cell-index
parsing, and player-id / player-count normalization from the rules
data.  Family modules (``grid.py``, later ``poker`` / ``mahjong`` /
``social``) implement the module protocol documented in
``families/__init__.py`` and reuse these helpers.

No solver imports here — solvers are only reached through
``SolverProvider`` (layer-4 contract, no ``layer3_solvers`` import).
"""

from __future__ import annotations

import re
from typing import Any

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance

#: ``cell_{r}_{c}`` id form used by grid templates (``derivedViews.cell``).
_CELL_ID_RE = re.compile(r"^cell_(\d+)_(\d+)$")


def engine_from_rules_dict(
    rules: dict[str, Any],
    seed: int | None = None,
    allow_codegen: bool = False,
    **kwargs: Any,
) -> GameEngine:
    """Build a bare ``GameEngine`` from a translated rules dict.

    Args:
        rules: v5 rules JSON dict.
        seed: Engine random seed.
        allow_codegen: When False, bypass ``RulesCompiler`` (pure
            interpreter path) per the A1 ``GameEngine`` switch; the
            default keeps translated custom rules on the interpreter
            path for reliability.
        **kwargs: Extra engine args (``variant`` / ``player_count``).

    Returns:
        The constructed engine.

    Notes:
        Before the A1 engine switch lands, ``allow_codegen`` is not an
        accepted keyword; a ``TypeError`` naming it falls back to the
        classic constructor signature so this helper works on both
        revisions.  Any other keyword ``TypeError`` is re-raised.
    """
    try:
        return GameEngine(rules, seed=seed, allow_codegen=allow_codegen, **kwargs)
    except TypeError as exc:
        if "allow_codegen" not in str(exc):
            raise
        return GameEngine(rules, seed=seed, **kwargs)


def resolve_all_chance(engine: GameEngine, state: dict) -> dict:
    """Advance through all pending chance nodes (generic engine protocol)."""
    while engine.get_node_type(state) == "chance":
        _, state = engine.sample_chance(state)
    return state


def action_cell_index(action: ActionInstance, cols: int) -> int:
    """Linear cell index of a grid action (``-1`` when unresolvable).

    Prefers the materialized ``cell._index`` field, then falls back to
    parsing the ``cell_{r}_{c}`` id; ``cols`` is the board width
    (``constants.board_size``) used for the row-major id fallback.
    """
    cell = action.params.get("cell", {})
    if not isinstance(cell, dict):
        return -1
    index = cell.get("_index")
    if isinstance(index, int) and index >= 0:
        return index
    match = _CELL_ID_RE.match(str(cell.get("id", "")))
    if match is None:
        return -1
    row, col = int(match.group(1)), int(match.group(2))
    return row * cols + col


def normalize_players(rules: dict[str, Any]) -> tuple[str, ...]:
    """Player ids from ``rules["players"]`` (str or ``{"id": ...}`` entries)."""
    out: list[str] = []
    for entry in rules.get("players", []):
        if isinstance(entry, str) and entry:
            out.append(entry)
        elif isinstance(entry, dict) and entry.get("id"):
            out.append(str(entry["id"]))
    return tuple(out)


def declared_player_counts(rules: dict[str, Any]) -> tuple[int, ...]:
    """Player-count options declared in ``rules["variants"]`` (``(2,)`` default)."""
    variants = rules.get("variants")
    if isinstance(variants, dict):
        count = variants.get("player_count")
        if isinstance(count, int) and count > 0:
            return (count,)
    return (2,)


def rules_board_size(rules: dict[str, Any]) -> int | None:
    """``constants.board_size`` when declared as a positive int, else ``None``."""
    value = rules.get("constants", {}).get("board_size")
    return int(value) if isinstance(value, int) and value > 0 else None
