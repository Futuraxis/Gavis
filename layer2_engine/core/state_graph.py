"""State representation for the v5.0 game engine.

Two-layer architecture:
  - Ground state: compact arrays + scalars, directly stored and cloned
  - Derived views: computed from ground state by rule (grid, enum, etc.)

Design principles:
  - No node/edge graph — use typed arrays + foreign keys
  - No entity enumeration — use derivation rules (grid, enumerate, regex)
  - Cloning is shallow array copy — no deep graph traversal
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Literal

from .expr_eval import ExprEvaluator

_TEMPLATE_RE = re.compile(r"\{([^}]+)\}")

# ── Core data types (single source since v5.2) ────────────────────────
# Previously defined in ``interfaces/solver_adapter.py`` (the Layer 2↔3
# contract); moved here so the adapter Protocol becomes a pure typing
# artifact and solvers import their data types straight from the engine
# layer (``solver_adapter`` now re-exports them for back-compat).

NodeType = Literal["player", "chance", "terminal"]
"""Type of a game node."""

State = dict[str, Any]
"""Game state — a generic dict with ground arrays + env scalars.

Ground arrays are stored under ``_arrays``, environment scalars under ``env``.
Derived views are computed on-the-fly by the engine.
"""


@dataclass
class ActionInstance:
    """A concrete action generated from an action template at runtime."""

    template_id: str
    type: str
    actor_id: str
    params: dict
    canonical_key: str


@dataclass
class ChanceOutcome:
    """A single outcome of a chance node."""

    key: str
    probability: float
    effect_ref: str
    canonical_key: str


Obs = dict[str, Any]
"""An observation returned by ``project_observation``.

For perfect-information games this includes materialized derived views;
for imperfect-information games it is the player's partial view after
visibility projection.
"""


# ── Ground state operations ──────────────────────────────────────────


def create_initial_state(
    schema: dict,
    constants: dict | None = None,
    players: list[dict] | None = None,
) -> dict:
    """Build initial ground state from JSON schema.

    The ground state consists of:
      - _arrays: Typed ground arrays (compact, O(1) index)
      - env: Environment scalars (turn, phase, winner, …)
      - _schema: Schema reference for derived view computation
      - _players: Player list
    """
    ground_def = schema.get("groundState", {})
    arrays: dict[str, list] = {}
    env: dict[str, Any] = {}
    constants = constants or {}
    players = players or []

    for name, field_def in ground_def.items():
        ftype = field_def.get("type")
        if ftype == "array":
            length = field_def.get("length")
            if isinstance(length, dict):
                length = _eval_length_expr(length, constants)
            length = length or 0
            if field_def.get("mutable"):
                arrays[name] = []
            else:
                arrays[name] = [None] * length

        elif ftype == "env":
            sub_fields = field_def.get("fields", {})
            for fname, sdef in sub_fields.items():
                env[fname] = deepcopy(sdef.get("initial"))

    # Default env values if schema didn't specify
    env.setdefault("phase", "playing")
    env.setdefault("winner", None)

    return {
        "_schema": schema,
        "_arrays": arrays,
        "_players": players,
        "_constants": constants,
        "env": env,
        "_pending_events": [],
        "_pending_effects": [],
    }


def _eval_length_expr(expr: dict, constants: dict) -> int:
    """Evaluate a simple length expression like {'expr': ...} or {'const': N}."""
    if "const" in expr:
        const_v = expr["const"]
        return int(const_v) if isinstance(const_v, (int, float)) else 0
    raw = expr.get("expr", "")
    # 常量替换：按名字长度降序 + 词边界匹配，避免短名子串污染长名
    # （如 "size" 先替换会破坏 "board_size" 内部），结果也不依赖 dict 遍历序。
    num_consts = {k: v for k, v in constants.items() if isinstance(v, (int, float))}
    if num_consts:
        names = sorted(num_consts, key=len, reverse=True)
        pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")
        raw = pattern.sub(lambda m: str(num_consts[m.group(1)]), raw)
    # Simple eval — the input is trusted static rules JSON; this is not a
    # security sandbox (only the arithmetic surface is intended).
    try:
        return int(eval(raw, {"__builtins__": {}}, {}))
    except Exception:
        return 0


def clone_state(state: dict) -> dict:
    """Fast shallow copy of ground state.

    Arrays are copied by list() — O(n).  Environment scalars
    are shallow-copied.  Schema reference is shared (read-only).
    """
    arrays = {}
    for k, v in state.get("_arrays", {}).items():
        if isinstance(v, list):
            arrays[k] = list(v)
        else:
            arrays[k] = v

    return {
        "_schema": state["_schema"],
        "_arrays": arrays,
        "_players": state.get("_players", []),
        "_constants": state.get("_constants", {}),
        "env": dict(state.get("env", {})),
        "_pending_events": [],
        "_pending_effects": [],
    }


# ── Derived View Engine ──────────────────────────────────────────────


class DerivedViewEngine:
    """Computes derived entity views from ground state.

    A derived view is a virtual table: entities computed from ground
    arrays via shape rules (grid, enum, literal, regex).

    Design: O(N) materialization on query, no caching between queries.
    Field defs are precompiled into closures at construction.

    Rule ``functions`` aliases are registered on the shared evaluator:
    they are pure expression definitions, so ``call`` works in view
    fields instead of raising "Unknown function" at materialization.
    """

    def __init__(self, schema: dict, functions: dict | None = None):
        self._views: dict = schema.get("derivedViews", {})
        self._evaluator = ExprEvaluator()
        self._evaluator.set_functions(functions or {})
        self._compiled_fields: dict[str, list[tuple[str, Callable[..., Any]]]] = {}
        for vname, vdef in self._views.items():
            fdefs = vdef.get("fields", {})
            self._compiled_fields[vname] = [(fname, self._evaluator.compile(fdef)) for fname, fdef in fdefs.items()]

    def materialize(self, state: dict, view_name: str) -> list[dict]:
        """Materialize a derived view from the current ground state."""
        view_def = self._views.get(view_name)
        if view_def is None:
            return []

        source = view_def.get("from", {})
        source_type = source.get("type", "literal")
        arrays = state.get("_arrays", {})
        players = state.get("_players", [])
        constants = state.get("_constants", {})
        env = state.get("env", {})
        field_fns = self._compiled_fields.get(view_name, [])

        # --- Grid derivation: 1D array → 2D entities indexed by (row, col) ---
        if source_type == "grid":
            arr = arrays.get(source.get("array", ""), [])
            cols_expr = source.get("cols", {})
            # Provide full context for _resolve_value including $constants key
            resolve_ctx = {"$constants": constants, "env": env, "$players": players}
            cols = _resolve_value(cols_expr, resolve_ctx)
            if not isinstance(cols, (int, float)) or cols <= 0:
                cols = 1
            cols = int(cols)
            rows = (len(arr) + cols - 1) // cols if cols > 0 else 0
            entities = []
            for r in range(rows):
                for c in range(cols):
                    idx = r * cols + c
                    if idx >= len(arr):
                        break
                    entity = {"_index": idx, "value": arr[idx], "_row": r, "_col": c}
                    _apply_compiled_fields(field_fns, entity, env, constants, players)
                    entities.append(entity)
            return entities

        # --- Enum derivation: array indexed by position ---
        if source_type == "enum":
            arr = arrays.get(source.get("array", ""), [])
            entities = []
            for i, val in enumerate(arr):
                entity = {"_index": i, "value": val, "_i": i}
                _apply_compiled_fields(field_fns, entity, env, constants, players)
                entities.append(entity)
            return entities

        # --- Literal derivation: from a static list ---
        if source_type == "literal":
            raw_list = source.get("list", [])
            # Resolve var references in the list
            if isinstance(raw_list, dict) and "var" in raw_list:
                raw_list = _resolve_value(
                    raw_list,
                    {
                        "$players": players,
                        "env": env,
                        **constants,
                    },
                )
            if not isinstance(raw_list, list):
                raw_list = []
            entities = []
            for i, val in enumerate(raw_list):
                entity = {"_index": i, "value": val, "_i": i}
                _apply_compiled_fields(field_fns, entity, env, constants, players)
                entities.append(entity)
            return entities

        return []


# ── Field evaluation helpers ──────────────────────────────────────────


def _apply_compiled_fields(
    field_fns: list[tuple[str, Callable[..., Any]]],
    entity: dict,
    env: dict,
    constants: dict,
    players: list,
) -> None:
    """Evaluate precompiled field closures into ``entity`` (in place)."""
    ctx = {
        "$self": entity,
        "$row": entity.get("_row", 0),
        "$col": entity.get("_col", 0),
        "$i": entity.get("_i", 0),
        "$env": env,
        "$constants": constants,
        "$players": players,
    }
    for fname, fn in field_fns:
        entity[fname] = fn(ctx)


def _resolve_value(expr: Any, ctx: dict) -> Any:
    """Resolve a simple expression value from context.

    Supports:
      - const: literal
      - var: context variable access
      - get: field access on a context value
      - template: string interpolation
      - switch: pattern match
      - raw string/number: pass through
    """
    if not isinstance(expr, dict):
        return expr

    if "const" in expr:
        return expr["const"]

    if "var" in expr:
        raw = expr["var"]
        path = raw.lstrip("$")
        parts = path.split(".")
        obj = ctx
        for p in parts:
            if isinstance(obj, dict):
                # Try direct key first, then with $ prefix
                if p in obj:
                    obj = obj.get(p)
                elif f"${p}" in obj:
                    obj = obj.get(f"${p}")
                else:
                    return None
            else:
                return None
        return obj

    if "get" in expr:
        target_path = expr["get"][0]
        field = expr["get"][1]
        # Resolve the target from ctx
        if isinstance(target_path, dict):
            target = _resolve_value(target_path, ctx)
        else:
            clean = target_path.lstrip("$")
            target = ctx.get(clean)
            if target is None:
                target = ctx.get(f"${clean}")
        if isinstance(target, dict):
            return target.get(field)
        return None

    if "template" in expr:
        tmpl = expr["template"]

        def _replacer(m):
            inner = m.group(1).strip()
            # If the inner doesn't start with $, try adding it for ctx lookup
            var_expr = inner if inner.startswith("$") else f"${inner}"
            val = _resolve_value({"var": var_expr}, ctx)
            return str(val) if val is not None else ""

        return re.sub(r"\{([^}]+)\}", _replacer, tmpl)

    if "switch" in expr:
        switch = expr["switch"]
        input_val = _resolve_value(expr["input"], ctx) if "input" in expr else None
        for case in switch:
            if case.get("case") == input_val:
                return case.get("then")
        # default
        default = [c for c in switch if "case" not in c]
        return default[0].get("then") if default else None

    # Should not reach here for field values
    return str(expr)
