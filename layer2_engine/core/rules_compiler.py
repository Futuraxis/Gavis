"""Rules compiler — compiles a v5.0 rules JSON into native Python functions.

Pipeline: rules → analysis → Python source generation → ``compile()`` → exec.

The generated functions implement the SolverAdapter hot paths (terminal
checks, legal actions, chance outcomes, view materialization) directly
over ground arrays, replacing per-call interpreter evaluation.  Every
artifact is probe-validated against the interpreter at engine load;
on any mismatch the artifact is disabled and the engine falls back to
interpretation (which itself is compiled-closure-fast).

Supported shapes (v1):
  - Views: grid / enum / literal materialization with const / var / get /
    template / switch fields.
  - Query predicates: eq / neq / gt / gte / lt / lte / and / or / not /
    const / get over ``$node`` fields; ``count(query(...))`` aggregates.
  - Terminal conditions: any combination of the above.
  - Action templates: single-parameter with query domain, const ``legal``,
    template canonical keys.
  - Chance templates: explicit probability tables.

Anything outside these shapes stays on the interpreter path.
"""

from __future__ import annotations

import re
from typing import Any

from .state_graph import ActionInstance, ChanceOutcome, BUILTIN_FUNCTIONS

_TEMPLATE_RE = re.compile(r'\{([^}]+)\}')
_ARITH_PREFIX_RE = re.compile(r'\$([a-zA-Z_][a-zA-Z0-9_.]*)')
_ARITH_BARE_RE = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_.]*)\b')
_IDENT_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

_CMP_OPS = {
    'eq': '==', 'neq': '!=', 'gt': '>', 'gte': '>=', 'lt': '<', 'lte': '<=',
}
_NODE_KEYS = ('node', 'self', 'cell')
_NODE_VALUE_FIELDS = ('occupant', 'value')


class UnsupportedShape(Exception):
    """Construct outside the codegen-supported subset; caller falls back."""


class _Gen:
    """Expression → Python source generator for one compilation context.

    ``binder`` maps context names to Python fragments (e.g. ``env`` →
    ``"env"``, ``cell`` → ``"node"``, ``col`` → ``"_c"``).
    ``view_fns`` maps view name → generated ``_materialize_<view>`` fn name.
    ``view_defs`` maps view name → raw view def (for source-array lookup).
    ``node_scan`` binds ``$node.<occupant|value>`` to the raw array value.
    """

    __slots__ = ('constants', 'view_fns', 'view_defs', 'binder', 'node_scan')

    def __init__(self, constants: dict, view_fns: dict, view_defs: dict,
                 binder: dict | None = None, node_scan: bool = False):
        self.constants = constants
        self.view_fns = view_fns
        self.view_defs = view_defs
        self.binder = binder or {}
        self.node_scan = node_scan

    # ── Expression → Python ──────────────────────────────────────────

    def expr(self, spec: Any) -> str:
        """Compile a static expression spec into a Python expression string."""
        if not isinstance(spec, dict):
            return repr(spec)

        if 'const' in spec:
            return repr(spec['const'])

        if 'var' in spec:
            return self._path(spec['var'])

        if 'get' in spec:
            target, field = spec['get']
            head = (self.expr(target) if isinstance(target, dict)
                    else self._path(target))
            return f'{head}[{field!r}]'

        if 'template' in spec:
            parts: list[str] = []
            last = 0
            for m in _TEMPLATE_RE.finditer(spec['template']):
                if m.start() > last:
                    parts.append(repr(spec['template'][last:m.start()]))
                inner = m.group(1).strip()
                var_py = self._path(inner if inner.startswith('$') else f'${inner}')
                parts.append(f'_s({var_py})')
                last = m.end()
            if last < len(spec['template']):
                parts.append(repr(spec['template'][last:]))
            return '(' + ' + '.join(parts) + ')'

        if 'switch' in spec:
            input_py = self.expr(spec.get('input', {'var': '$input'}))
            branches = [
                f'(({input_py}) == {repr(case["case"])} and {self.expr(case["then"])})'
                for case in spec['switch'] if 'case' in case
            ]
            defaults = [case for case in spec['switch'] if 'case' not in case]
            if defaults:
                branches.append(f'({self.expr(defaults[0].get("then"))})')
            else:
                branches.append('(None)')
            return '(' + ' or '.join(branches) + ')'

        for op, pyop in _CMP_OPS.items():
            if op in spec:
                return f'({self.expr(spec[op][0])} {pyop} {self.expr(spec[op][1])})'

        if 'and' in spec:
            return '(' + ' and '.join(f'bool({self.expr(sub)})' for sub in spec['and']) + ')'

        if 'or' in spec:
            return '(' + ' or '.join(f'bool({self.expr(sub)})' for sub in spec['or']) + ')'

        if 'not' in spec:
            return f'(not bool({self.expr(spec["not"])}))'

        if 'expr' in spec:
            return self._arith(spec['expr'])

        if 'count' in spec:
            arg = spec['count']
            if isinstance(arg, dict) and 'query' in arg:
                arg = arg['query']
            if isinstance(arg, dict) and arg.get('view', '') in self.view_fns:
                return self._count_py(arg)
            return f'len({self.expr(arg)})'

        if 'query' in spec:
            return self._query_py(spec['query'])

        raise UnsupportedShape(spec)

    def _path(self, path: str | list) -> str:
        """Compile a ``var``/``get`` path into a Python access expression."""
        if isinstance(path, list):
            first, *rest = path
            head = self.expr(first) if isinstance(first, dict) else self._path_str(first)
            for r in rest:
                head = self._field_get(head, r)
            return head
        return self._path_str(path)

    def _path_str(self, path: str) -> str:
        """Compile a dotted path string into a Python access expression."""
        raw = path.lstrip('$').lstrip('.')
        parts = raw.split('.')
        if not parts:
            raise UnsupportedShape(path)
        first, rest = parts[0], parts[1:]

        # Inline constants at compile time.
        if first == 'constants':
            node: Any = self.constants
            for r in rest:
                if not isinstance(node, dict) or r not in node:
                    raise UnsupportedShape(path)
                node = node[r]
            return repr(node)

        if first in _NODE_KEYS and self.node_scan:
            if len(rest) == 1 and rest[0] in _NODE_VALUE_FIELDS:
                return self.binder['_val']
            raise UnsupportedShape(path)

        head = self.binder.get(first)
        if head is None:
            raise UnsupportedShape(path)
        for r in rest:
            head = self._field_get(head, r)
        return head

    @staticmethod
    def _field_get(head: str, part: str) -> str:
        """Append one path segment to a Python access expression."""
        if part.lstrip('-').isdigit():
            return f'{head}[{int(part)}]'
        return f'{head}[{part!r}]'

    def _arith(self, raw: str) -> str:
        """Compile a compact arithmetic string into a Python expression."""
        def _repl_prefixed(m: re.Match) -> str:
            return self._path_str(f'${m.group(1)}')

        s = _ARITH_PREFIX_RE.sub(_repl_prefixed, raw)

        def _repl_bare(m: re.Match) -> str:
            name = m.group(1)
            val = self.constants.get(name)
            if isinstance(val, (int, float)):
                return repr(val)
            raise UnsupportedShape(f'arith:{name}')

        return _ARITH_BARE_RE.sub(_repl_bare, s)

    # ── Query / aggregate specialisation ─────────────────────────────

    def _query_py(self, qexpr: dict) -> str:
        """Compile a query into a Python expression yielding entity dicts."""
        view = qexpr.get('view', '')
        fn_name = self.view_fns.get(view)
        if fn_name is None:
            raise UnsupportedShape(f'query:{view}')
        where = qexpr.get('filter') or qexpr.get('where')
        if where is None:
            return f'{fn_name}(state)'
        try:
            pred = self._scan_predicate(where)
            arr = self._view_array_expr(qexpr)
            return f'[{fn_name}(_i, _v) for _i, _v in enumerate({arr}) if {pred}]'
        except UnsupportedShape:
            entity_gen = _Gen(self.constants, self.view_fns, self.view_defs,
                              {**self.binder, 'node': '_n'})
            return f'[_n for _n in {fn_name}(state) if {entity_gen.expr(where)}]'

    def _count_py(self, qexpr: dict) -> str:
        """Compile ``count(query)`` into a Python aggregate expression."""
        try:
            pred = self._scan_predicate(qexpr.get('filter') or qexpr.get('where'))
            arr = self._view_array_expr(qexpr)
            return f'sum(1 for _v in {arr} if {pred})'
        except UnsupportedShape:
            return f'len({self._query_py(qexpr)})'

    def _scan_predicate(self, where: dict | None) -> str:
        """Compile a query predicate against the raw array (occupant-only)."""
        if where is None:
            return 'True'
        gen = _Gen(self.constants, self.view_fns, self.view_defs,
                   {**self.binder, '_val': '_v'}, node_scan=True)
        return gen.expr(where)

    def _view_array_expr(self, qexpr: dict) -> str:
        """Python expression for a view's source array, if it has one."""
        view_def = self.view_defs.get(qexpr.get('view', ''), {})
        arr_name = view_def.get('from', {}).get('array')
        if not arr_name:
            raise UnsupportedShape('view without array source')
        return f"state['_arrays'][{arr_name!r}]"


# ── View materialization codegen ─────────────────────────────────────

def _gen_view(vname: str, vdef: dict, constants: dict) -> str:
    """Generate ``_ent_<view>(_i, _v)`` + ``_materialize_<view>(state)``.

    Returns the source text or raises UnsupportedShape.
    """
    if not _IDENT_RE.match(vname):
        raise UnsupportedShape(f'view name: {vname}')
    source = vdef.get('from', {})
    stype = source.get('type', 'literal')
    fields = vdef.get('fields', {})
    binder = {'node': 'node', 'self': 'node', 'cell': 'node', 'i': '_i'}
    if stype == 'grid':
        binder.update(row='_r', col='_c')
    else:
        # Enum/literal entities have no _row/_col; interpreter defaults to 0.
        binder.update(row='0', col='0')

    if stype == 'grid':
        cols = _resolve_cols(source.get('cols', {}), constants)
        ent_body = [
            f'    _bs = {cols}',
            '    _r = _i // _bs',
            '    _c = _i % _bs',
        ]
        mat_body = [
            f'    arr = state["_arrays"][{source.get("array", "")!r}]',
            '    _out = []',
            '    for _i, _v in enumerate(arr):',
            f'        _out.append(_ent_{vname}(_i, _v))',
            '    return _out',
        ]
    elif stype == 'enum':
        ent_body = []
        mat_body = [
            f'    arr = state["_arrays"][{source.get("array", "")!r}]',
            '    _out = []',
            '    for _i, _v in enumerate(arr):',
            f'        _out.append(_ent_{vname}(_i, _v))',
            '    return _out',
        ]
    elif stype == 'literal':
        raw_list = source.get('list', [])
        if isinstance(raw_list, dict) and raw_list.get('var', '').lstrip('$') == 'players':
            list_src = 'state["_players"]'
        elif isinstance(raw_list, list):
            list_src = repr(raw_list)
        else:
            raise UnsupportedShape('literal list')
        ent_body = []
        mat_body = [
            f'    _src = {list_src}',
            '    _out = []',
            '    for _i, _v in enumerate(_src):',
            f'        _out.append(_ent_{vname}(_i, _v))',
            '    return _out',
        ]
    else:
        raise UnsupportedShape(f'view type: {stype}')

    if stype == 'grid':
        ent_body.append("    node = {'_index': _i, 'value': _v, '_row': _r, '_col': _c}")
    else:
        ent_body.append("    node = {'_index': _i, 'value': _v, '_i': _i}")

    gen = _Gen(constants, {}, {}, binder)
    for fname, fdef in fields.items():
        if not _IDENT_RE.match(fname):
            raise UnsupportedShape(f'field name: {fname}')
        ent_body.append(f'    node[{fname!r}] = {gen.expr(fdef)}')
    ent_body.append('    return node')

    return (
        f'def _ent_{vname}(_i, _v):\n' + '\n'.join(ent_body) + '\n\n'
        f'def _materialize_{vname}(state):\n' + '\n'.join(mat_body) + '\n'
    )


def _resolve_cols(cols_expr: dict, constants: dict) -> int:
    """Resolve a grid ``cols`` spec to a compile-time integer."""
    try:
        gen = _Gen(constants, {}, {})
        val = _safe_eval(gen.expr(cols_expr))
    except Exception as exc:  # noqa: BLE001 — any resolution failure → fallback
        raise UnsupportedShape('cols') from exc
    if not isinstance(val, (int, float)) or val <= 0:
        return 1
    return int(val)


def _safe_eval(py_expr: str) -> Any:
    """Evaluate a generated constant expression safely."""
    return eval(py_expr, {'__builtins__': {}}, {})  # noqa: S307 — consts only


# ── Terminal / actions / chance codegen ──────────────────────────────

def _gen_is_terminal(terminal: list[dict], constants: dict,
                     view_fns: dict, view_defs: dict) -> str:
    """Generate ``is_terminal(state)`` covering all terminal rules."""
    if not terminal:
        raise UnsupportedShape('no terminal rules')
    gen = _Gen(constants, view_fns, view_defs,
               {'env': 'env', 'state': 'state', 'players': 'players', 'node': 'node'})
    lines = [
        'def is_terminal(state):',
        '    env = state["env"]',
    ]
    for rule in terminal:
        cond = rule.get('condition')
        if not isinstance(cond, dict):
            raise UnsupportedShape(f'terminal: {rule.get("id")}')
        lines.append(f'    if {gen.expr(cond)}:')
        lines.append('        return True')
    lines.append('    return False')
    return '\n'.join(lines) + '\n'


def _gen_legal_actions(actions: list[dict], queries: dict, constants: dict,
                       view_fns: dict, view_defs: dict) -> str:
    """Generate ``legal_actions(state)`` for the supported template shape."""
    if not actions:
        raise UnsupportedShape('no actions')
    lines = [
        'def legal_actions(state):',
        '    env = state["env"]',
        '    _out = []',
    ]
    for tmpl in actions:
        params = tmpl.get('params', {})
        if len(params) != 1:
            raise UnsupportedShape(f'action: {tmpl["id"]} params != 1')
        if tmpl.get('legal', {'const': True}) != {'const': True}:
            raise UnsupportedShape(f'action: {tmpl["id"]} legal not const true')
        phases = tmpl.get('phases', [])
        if not phases or not all(isinstance(p, str) for p in phases):
            raise UnsupportedShape(f'action: {tmpl["id"]} phases')

        pname, pdef = next(iter(params.items()))
        domain = pdef.get('domain', [])
        if not (isinstance(domain, dict) and 'ref' in domain):
            raise UnsupportedShape(f'action: {tmpl["id"]} domain not query ref')
        qname = domain['ref']
        qdef = queries.get(qname)
        if qdef is None:
            raise UnsupportedShape(f'action: {tmpl["id"]} unknown query {qname}')

        actor = tmpl.get('actor', {'var': '$env.turn'})
        ck = tmpl.get('canonicalKey', {'template': tmpl['id']})
        bind = {'env': 'env', 'state': 'state', 'players': 'players',
                'node': 'node', 'cell': 'node', pname: 'node'}
        gen = _Gen(constants, view_fns, view_defs, bind)
        try:
            actor_py = gen.expr(actor)
            ck_py = gen.expr(ck)
            qpy = gen._query_py({'view': qdef.get('view', ''),  # noqa: SLF001
                                 'filter': qdef.get('filter') or qdef.get('where')})
        except UnsupportedShape as exc:
            raise UnsupportedShape(f'action: {tmpl["id"]}: {exc}') from exc

        phase_guard = ' or '.join(f"env['phase'] == {p!r}" for p in phases)
        lines.append(f'    if {phase_guard}:')
        lines.append(f'        for node in {qpy}:')
        lines.append(f"            _out.append(ActionInstance({tmpl['id']!r}, "
                     f"{tmpl.get('type', 'action')!r}, {actor_py}, "
                     f"{{{pname!r}: node}}, {ck_py}))")
    lines.append('    return _out')
    return '\n'.join(lines) + '\n'


def _gen_chance(chance: list[dict], constants: dict,
                view_fns: dict, view_defs: dict) -> str:
    """Generate ``chance_outcomes(state)`` for explicit probability tables."""
    if not chance:
        raise UnsupportedShape('no chance templates')
    lines = [
        'def chance_outcomes(state):',
        '    env = state["env"]',
        '    _out = []',
    ]
    for ct in chance:
        phases = ct.get('phases', [])
        prob = ct.get('probability', {})
        explicit = prob.get('explicit') if isinstance(prob, dict) else None
        if not phases or not explicit:
            raise UnsupportedShape('chance: uniform/nonexplicit not supported')
        effect_map = ct.get('effectMap', {})
        ck_tmpl = ct.get('canonicalKey', {'template': 'chance:{outcome}'})
        phase_guard = ' or '.join(f"env['phase'] == {p!r}" for p in phases)
        lines.append(f'    if {phase_guard}:')
        for entry in explicit:
            outcome_val = entry.get('outcome', entry.get('value'))
            prob_expr = entry.get('prob', entry.get('probability'))
            if not isinstance(prob_expr, (int, float)):
                raise UnsupportedShape('chance: non-const probability')
            gen = _Gen(constants, view_fns, view_defs,
                       {'env': 'env', 'state': 'state', 'outcome': repr(str(outcome_val))})
            ck_py = gen.expr(ck_tmpl)
            effect_ref = effect_map.get(str(outcome_val), f'do_{outcome_val}')
            lines.append(f'        _out.append(ChanceOutcome('
                         f'{str(outcome_val)!r}, {float(prob_expr)}, '
                         f'{effect_ref!r}, {ck_py}))')
    lines.append('    return _out')
    return '\n'.join(lines) + '\n'


# ── Top-level compiler ───────────────────────────────────────────────

class CompiledArtifacts:
    """Compiled rule artifacts; probe-validated against the interpreter."""

    __slots__ = ('materialize', 'is_terminal', 'legal_actions', 'chance_outcomes',
                 '_views', '_view_defs')

    def __init__(self) -> None:
        self.materialize: callable | None = None
        self.is_terminal: callable | None = None
        self.legal_actions: callable | None = None
        self.chance_outcomes: callable | None = None
        self._views: dict[str, callable] = {}
        self._view_defs: dict[str, dict] = {}

    def validate(self, engine) -> None:
        """Probe-compare generated artifacts against the interpreter.

        The engine must not yet have ``_compiled`` assigned, so its public
        methods run the interpreter path.  Any artifact that disagrees on
        a probe is disabled (falls back to interpretation).
        """
        state = engine.create_initial_state()
        probes = [state]
        for _ in range(3):
            nt = engine.get_node_type(state)
            if nt == 'player':
                actions = engine.get_legal_actions(state)
                if not actions:
                    break
                state = engine.apply_action(state, actions[0])
                while engine.get_node_type(state) == 'chance':
                    _, state = engine.sample_chance(state)
            elif nt == 'chance':
                _, state = engine.sample_chance(state)
            else:
                break
            probes.append(state)

        try:
            for p in probes:
                if self.is_terminal is not None and self.is_terminal(p) != engine.is_terminal(p):
                    self.is_terminal = None
                if self.legal_actions is not None:
                    mine = [a.canonical_key for a in self.legal_actions(p)]
                    theirs = [a.canonical_key for a in engine.get_legal_actions(p)]
                    if mine != theirs:
                        self.legal_actions = None
                if self.chance_outcomes is not None:
                    mine = [(o.key, o.probability, o.effect_ref, o.canonical_key)
                            for o in self.chance_outcomes(p)]
                    theirs = [(o.key, o.probability, o.effect_ref, o.canonical_key)
                              for o in engine.get_chance_outcomes(p)]
                    if mine != theirs:
                        self.chance_outcomes = None
                for vname, fn in list(self._views.items()):
                    if fn(p) != engine._view_engine.materialize(p, vname):  # noqa: SLF001
                        self._views.pop(vname)
        except Exception:
            self.is_terminal = None
            self.legal_actions = None
            self.chance_outcomes = None
            self._views = {}

        if self._views:
            def _dispatch(state: dict, view_name: str):
                fn = self._views.get(view_name)
                return fn(state) if fn is not None else None
            self.materialize = _dispatch
        else:
            self.materialize = None


class RulesCompiler:
    """Compiles a rules dict into probe-validated native functions."""

    def compile(self, rules: dict) -> CompiledArtifacts:
        constants = rules.get('constants', {})
        artifacts = CompiledArtifacts()
        src_parts = ['def _s(x):\n    return "" if x is None else str(x)\n']
        view_fns: dict[str, str] = {}

        for vname, vdef in rules.get('derivedViews', {}).items():
            try:
                src = _gen_view(vname, vdef, constants)
            except UnsupportedShape:
                continue
            artifacts._view_defs[vname] = vdef
            view_fns[vname] = f'_materialize_{vname}'
            src_parts.append(src)

        try:
            src_parts.append(_gen_is_terminal(rules.get('terminal', []),
                                              constants, view_fns, artifacts._view_defs))
        except UnsupportedShape:
            pass
        try:
            src_parts.append(_gen_legal_actions(rules.get('actions', []),
                                                rules.get('queries', {}),
                                                constants, view_fns, artifacts._view_defs))
        except UnsupportedShape:
            pass
        try:
            src_parts.append(_gen_chance(rules.get('chance', []),
                                         constants, view_fns, artifacts._view_defs))
        except UnsupportedShape:
            pass

        namespace: dict[str, Any] = {
            'ActionInstance': ActionInstance,
            'ChanceOutcome': ChanceOutcome,
            **BUILTIN_FUNCTIONS,
        }
        source = '\n'.join(src_parts)
        exec(compile(source, '<rules_codegen>', 'exec'), namespace)  # noqa: S102 — generated
        for vname, fn_name in view_fns.items():
            artifacts._views[vname] = namespace[fn_name]
        artifacts.is_terminal = namespace.get('is_terminal')
        artifacts.legal_actions = namespace.get('legal_actions')
        artifacts.chance_outcomes = namespace.get('chance_outcomes')
        return artifacts
