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
from typing import Any

from .poker_utils import (
    card_rank,
    contains,
    poker_call_to,
    poker_hand_name,
    poker_hand_value,
    poker_min_raise_to,
    poker_payoff,
    poker_pot,
    poker_round_over,
    poker_winner,
)

_TEMPLATE_RE = re.compile(r'\{([^}]+)\}')


# ── Core data types (shared with SolverAdapter Protocol) ────────────────

@dataclass
class ActionInstance:
    """A concrete action generated from an ActionTemplate at runtime."""
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
    ground_def = schema.get('groundState', {})
    arrays: dict[str, list] = {}
    env: dict[str, Any] = {}
    constants = constants or {}
    players = players or []

    for name, field_def in ground_def.items():
        ftype = field_def.get('type')
        if ftype == 'array':
            length = field_def.get('length')
            if isinstance(length, dict):
                length = _eval_length_expr(length, constants)
            length = length or 0
            if field_def.get('mutable'):
                arrays[name] = []
            else:
                arrays[name] = [None] * length

        elif ftype == 'env':
            sub_fields = field_def.get('fields', {})
            for fname, sdef in sub_fields.items():
                env[fname] = deepcopy(sdef.get('initial'))

    # Default env values if schema didn't specify
    env.setdefault('phase', 'playing')
    env.setdefault('winner', None)

    return {
        '_schema': schema,
        '_arrays': arrays,
        '_players': players,
        '_constants': constants,
        'env': env,
        '_pending_events': [],
        '_pending_effects': [],
    }


def _eval_length_expr(expr: dict, constants: dict) -> int:
    """Evaluate a simple length expression like {'expr': 'board_size * board_size'}."""
    raw = expr.get('expr', '')
    context = constants.copy()
    # Try basic arithmetic: ONLY multiplication and subtraction for security
    try:
        # Replace constants
        for k, v in context.items():
            if isinstance(v, (int, float)):
                raw = raw.replace(k, str(v))
        # Simple eval — only allow numbers and basic ops
        result = eval(raw, {'__builtins__': {}}, {})
        return int(result)
    except Exception:
        return 0


def clone_state(state: dict) -> dict:
    """Fast shallow copy of ground state.

    Arrays are copied by list() — O(n).  Environment scalars
    are shallow-copied.  Schema reference is shared (read-only).
    """
    arrays = {}
    for k, v in state.get('_arrays', {}).items():
        if isinstance(v, list):
            arrays[k] = list(v)
        else:
            arrays[k] = v

    return {
        '_schema': state['_schema'],
        '_arrays': arrays,
        '_players': state.get('_players', []),
        '_constants': state.get('_constants', {}),
        'env': dict(state.get('env', {})),
        '_pending_events': [],
        '_pending_effects': [],
    }


# ── Expression precompilation ─────────────────────────────────────────

def compile_expr(spec: Any) -> Any:
    """Precompile a static expression spec into a closure over ctx.

    Returns either the literal value (non-dict specs) or ``fn(ctx)`` with
    semantics identical to ``_resolve_value(spec, ctx)``.  Fast-path types
    (const/var/get/template/switch) are inlined as closures; everything
    else falls back to ``str(spec)`` exactly like ``_resolve_value``.

    Used for derived-view field defs, which are evaluated ~700k times per
    MCTS move — compiling them eliminates ~4M dict-walking calls.
    """
    if not isinstance(spec, dict):
        return spec

    if 'const' in spec:
        value = spec['const']
        return lambda ctx: value

    if 'var' in spec:
        parts = spec['var'].lstrip('$').split('.')

        def _var(ctx: dict, parts: list = parts) -> Any:
            obj = ctx
            for p in parts:
                if isinstance(obj, dict):
                    if p in obj:
                        obj = obj[p]
                    elif f'${p}' in obj:
                        obj = obj[f'${p}']
                    else:
                        return None
                else:
                    return None
            return obj

        return _var

    if 'get' in spec:
        target_path, field = spec['get']
        if isinstance(target_path, dict):
            tgt_fn = compile_expr(target_path)
        else:
            clean = target_path.lstrip('$')

            def _tgt(ctx: dict, clean: str = clean) -> Any:
                value = ctx.get(clean)
                if value is None:
                    value = ctx.get(f'${clean}')
                return value

            tgt_fn = _tgt

        def _get(ctx: dict, tgt_fn: callable = tgt_fn, field: str = field) -> Any:
            target = tgt_fn(ctx)
            if isinstance(target, dict):
                return target.get(field)
            return None

        return _get

    if 'template' in spec:
        # Split template into literal parts + compiled var closures once.
        tmpl = spec['template']
        parts: list = []
        last = 0
        for m in _TEMPLATE_RE.finditer(tmpl):
            if m.start() > last:
                parts.append(tmpl[last:m.start()])
            inner = m.group(1).strip()
            parts.append(compile_expr({'var': inner if inner.startswith('$') else f'${inner}'}))
            last = m.end()
        if last < len(tmpl):
            parts.append(tmpl[last:])

        def _template(ctx: dict, parts: list = parts) -> str:
            out = []
            for p in parts:
                if callable(p):
                    value = p(ctx)
                    out.append(str(value) if value is not None else '')
                else:
                    out.append(p)
            return ''.join(out)

        return _template

    if 'switch' in spec:
        cases = []
        default_fn = None
        for case in spec['switch']:
            if 'case' in case:
                cases.append((case['case'], compile_expr(case['then'])))
            else:
                default_fn = compile_expr(case.get('then'))
        input_fn = compile_expr(spec.get('input', {'var': '$input'}))

        def _switch(
            ctx: dict,
            cases: list = cases,
            default_fn: callable | None = default_fn,
            input_fn: callable = input_fn,
        ) -> Any:
            value = input_fn(ctx)
            for case_val, then_fn in cases:
                if case_val == value:
                    return then_fn(ctx)
            return default_fn(ctx) if default_fn is not None else None

        return _switch

    # Fallback mirrors _resolve_value: str(expr)
    return lambda ctx, spec=spec: str(spec)


# ── Derived View Engine ──────────────────────────────────────────────

class DerivedViewEngine:
    """Computes derived entity views from ground state.

    A derived view is a virtual table: entities computed from ground
    arrays via shape rules (grid, enum, literal, regex).

    Design: O(N) materialization on query, no caching between queries.
    Field defs are precompiled into closures at construction.
    """

    def __init__(self, schema: dict):
        self._views: dict = schema.get('derivedViews', {})
        self._compiled_fields: dict[str, list[tuple[str, callable]]] = {}
        for vname, vdef in self._views.items():
            fdefs = vdef.get('fields', {})
            self._compiled_fields[vname] = [
                (fname, compile_expr(fdef)) for fname, fdef in fdefs.items()
            ]

    def materialize(self, state: dict, view_name: str) -> list[dict]:
        """Materialize a derived view from the current ground state."""
        view_def = self._views.get(view_name)
        if view_def is None:
            return []

        source = view_def.get('from', {})
        source_type = source.get('type', 'literal')
        arrays = state.get('_arrays', {})
        players = state.get('_players', [])
        constants = state.get('_constants', {})
        env = state.get('env', {})
        field_fns = self._compiled_fields.get(view_name, [])

        # --- Grid derivation: 1D array → 2D entities indexed by (row, col) ---
        if source_type == 'grid':
            arr = arrays.get(source.get('array', ''), [])
            cols_expr = source.get('cols', {})
            # Provide full context for _resolve_value including $constants key
            resolve_ctx = {'$constants': constants, 'env': env, '$players': players}
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
                    entity = {'_index': idx, 'value': arr[idx], '_row': r, '_col': c}
                    _apply_compiled_fields(field_fns, entity, env, constants, players)
                    entities.append(entity)
            return entities

        # --- Enum derivation: array indexed by position ---
        if source_type == 'enum':
            arr = arrays.get(source.get('array', ''), [])
            entities = []
            for i, val in enumerate(arr):
                entity = {'_index': i, 'value': val, '_i': i}
                _apply_compiled_fields(field_fns, entity, env, constants, players)
                entities.append(entity)
            return entities

        # --- Literal derivation: from a static list ---
        if source_type == 'literal':
            raw_list = source.get('list', [])
            # Resolve var references in the list
            if isinstance(raw_list, dict) and 'var' in raw_list:
                raw_list = _resolve_value(raw_list, {
                    '$players': players,
                    'env': env,
                    **constants,
                })
            if not isinstance(raw_list, list):
                raw_list = []
            entities = []
            for i, val in enumerate(raw_list):
                entity = {'_index': i, 'value': val, '_i': i}
                _apply_compiled_fields(field_fns, entity, env, constants, players)
                entities.append(entity)
            return entities

        return []


# ── Field evaluation helpers ──────────────────────────────────────────

def _apply_compiled_fields(
    field_fns: list[tuple[str, callable]],
    entity: dict,
    env: dict,
    constants: dict,
    players: list,
) -> None:
    """Evaluate precompiled field closures into ``entity`` (in place)."""
    ctx = {
        '$self': entity,
        '$row': entity.get('_row', 0),
        '$col': entity.get('_col', 0),
        '$i': entity.get('_i', 0),
        '$env': env,
        '$constants': constants,
        '$players': players,
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

    if 'const' in expr:
        return expr['const']

    if 'var' in expr:
        raw = expr['var']
        path = raw.lstrip('$')
        parts = path.split('.')
        obj = ctx
        for p in parts:
            if isinstance(obj, dict):
                # Try direct key first, then with $ prefix
                if p in obj:
                    obj = obj.get(p)
                elif f'${p}' in obj:
                    obj = obj.get(f'${p}')
                else:
                    return None
            else:
                return None
        return obj

    if 'get' in expr:
        target_path = expr['get'][0]
        field = expr['get'][1]
        # Resolve the target from ctx
        if isinstance(target_path, dict):
            target = _resolve_value(target_path, ctx)
        else:
            clean = target_path.lstrip('$')
            target = ctx.get(clean)
            if target is None:
                target = ctx.get(f'${clean}')
        if isinstance(target, dict):
            return target.get(field)
        return None

    if 'template' in expr:
        tmpl = expr['template']
        import re
        def _replacer(m):
            inner = m.group(1).strip()
            # If the inner doesn't start with $, try adding it for ctx lookup
            var_expr = inner if inner.startswith('$') else f'${inner}'
            val = _resolve_value({'var': var_expr}, ctx)
            return str(val) if val is not None else ''
        return re.sub(r'\{([^}]+)\}', _replacer, tmpl)

    if 'switch' in expr:
        switch = expr['switch']
        input_val = _resolve_value(expr['input'], ctx) if 'input' in expr else None
        for case in switch:
            if case.get('case') == input_val:
                return case.get('then')
        # default
        default = [c for c in switch if 'case' not in c]
        return default[0].get('then') if default else None

    # Should not reach here for field values
    return str(expr)


# ── Win check (built-in engine function) ──────────────────────────────

def check_line(state: dict, player_id: str, win_length: int) -> bool:
    """Check whether ``player_id`` has ``win_length`` consecutive pieces.

    Built-in engine function (not a registered external function).
    Works with both v4.1 (``_board``) and v5.0 (``_arrays.board``) states.
    """
    board = state.get('_board')
    if board is None:
        arr = state.get('_arrays', {})
        board = arr.get('board', [])
    if board is None or not board:
        return False

    bs = _get_board_size(state)
    players = state.get('_players', [])
    player_ids = [p['id'] for p in players] if players and isinstance(players[0], dict) else ['p_black', 'p_white']

    if player_id not in player_ids:
        return False

    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for idx in range(len(board)):
        if board[idx] != player_id:
            continue
        x, y = idx % bs, idx // bs
        for dx, dy in directions:
            count = 1
            px, py = x + dx, y + dy
            while 0 <= px < bs and 0 <= py < bs and board[py * bs + px] == player_id:
                count += 1
                px += dx
                py += dy
            nx, ny = x - dx, y - dy
            while 0 <= nx < bs and 0 <= ny < bs and board[ny * bs + nx] == player_id:
                count += 1
                nx -= dx
                ny -= dy
            if count >= win_length:
                return True
    return False


def _get_board_size(state: dict) -> int:
    """Extract board size from various state formats."""
    bs = state.get('board_size')
    if bs:
        return bs
    arr = state.get('_arrays', {})
    board = arr.get('board', [])
    if not board:
        return 3
    return int(len(board) ** 0.5)


# ── Built-in function registry ────────────────────────────────────────

BUILTIN_FUNCTIONS: dict[str, callable] = {
    'check_line': check_line,
    'debug_print': lambda msg: print(f"[debug] {msg}"),
    # Texas Hold'em (rules/texas_holdem.json)
    'contains': contains,
    'card_rank': card_rank,
    'poker_hand_value': poker_hand_value,
    'poker_call_to': poker_call_to,
    'poker_min_raise_to': poker_min_raise_to,
    'poker_round_over': poker_round_over,
    'poker_winner': poker_winner,
    'poker_payoff': poker_payoff,
    'poker_pot': poker_pot,
    'poker_hand_name': poker_hand_name,
}
