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

from .state_graph import ActionInstance, ChanceOutcome

_TEMPLATE_RE = re.compile(r"\{([^}]+)\}")
# Single-pass token matcher for arithmetic strings: ``$``-prefixed paths
# (group 1) and bare identifiers (group 2) in one scan, so substituted
# paths are never re-matched as bare names by the second alternative.
_ARITH_TOKEN_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_.]*)|([a-zA-Z_][a-zA-Z0-9_.]*)\b")
_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_CMP_OPS = {
    "eq": "==",
    "neq": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}
_NODE_KEYS = ("node", "self", "cell")
_NODE_VALUE_FIELDS = ("occupant", "value")


class UnsupportedShapeError(Exception):
    """Construct outside the codegen-supported subset; caller falls back."""


class _Gen:
    """Expression → Python source generator for one compilation context.

    ``binder`` maps context names to Python fragments (e.g. ``env`` →
    ``"env"``, ``cell`` → ``"node"``, ``col`` → ``"_c"``).
    ``view_fns`` maps view name → generated ``_materialize_<view>`` fn name.
    ``view_defs`` maps view name → raw view def (for source-array lookup).
    ``node_scan`` binds ``$node.<occupant|value>`` to the raw array value.
    ``functions`` are the rules' alias definitions, inlined for ``call``.
    """

    __slots__ = ("constants", "view_fns", "view_defs", "binder", "node_scan", "functions")

    def __init__(
        self,
        constants: dict,
        view_fns: dict,
        view_defs: dict,
        binder: dict | None = None,
        node_scan: bool = False,
        functions: dict | None = None,
    ):
        self.constants = constants
        self.view_fns = view_fns
        self.view_defs = view_defs
        self.binder = binder or {}
        self.node_scan = node_scan
        self.functions = functions or {}

    # ── Expression → Python ──────────────────────────────────────────

    def _list_guard(self, list_py: str) -> str:
        """Wrap a list expression: None → [] (mirrors the interpreter's
        ``not isinstance(items, list) → []`` behaviour for switch misses).

        list/tuple/range 均视为可迭代（解释器对 range 也迭代）。
        """
        return f"(({list_py}) if isinstance(({list_py}), (list, tuple, range)) else [])"

    def _gen_expr(self, spec: Any, as_var: str) -> str:
        """Compile ``spec`` with ``as_var`` bound to the loop item ``_it``."""
        sub = _Gen(
            self.constants,
            self.view_fns,
            self.view_defs,
            {**self.binder, as_var.lstrip("$"): "_it"},
            functions=self.functions,
        )
        return sub.expr(spec)

    def _expr_rest(self, spec: Any) -> str:
        """Remaining expression types (list primitives + aliases)."""
        if "filter" in spec:
            fspec = spec["filter"]
            list_py = self._list_guard(self.expr(fspec["list"]))
            pred = self._gen_expr(fspec["where"], fspec.get("as", "$node"))
            return f"[_it for _it in {list_py} if {pred}]"

        if "concat" in spec:
            return "(" + " + ".join(self.expr(s) for s in spec["concat"]) + ")"

        if "contains" in spec:
            # [list, item] → item in list
            return f"({self.expr(spec['contains'][1])} in {self._list_guard(self.expr(spec['contains'][0]))})"

        if "map" in spec:
            mspec = spec["map"]
            list_py = self._list_guard(self.expr(mspec["list"]))
            body = self._gen_expr(mspec["expr"], mspec.get("as", "$node"))
            return f"[{body} for _it in {list_py}]"

        if "range" in spec:
            rspec = spec["range"]
            return f"range({self.expr(rspec['from'])}, {self.expr(rspec['to'])})"

        if "any" in spec or "all" in spec:
            key = "any" if "any" in spec else "all"
            aspec = spec[key]
            list_py = self._list_guard(self.expr(aspec["list"]))
            pred = self._gen_expr(aspec["where"], aspec.get("as", "$node"))
            return f"({key}(bool({pred}) for _it in {list_py}))"

        if "group" in spec:
            # {'group': {'list': ..., 'by': ...}} → [{'key','count','items'}...]
            # （保序：dict.fromkeys 去重保持首次出现顺序）
            gspec = spec["group"]
            list_py = self.expr(gspec["list"])
            by_py: str | None = None
            if "by" in gspec:
                by_py = self._gen_expr(gspec["by"], "$item")
            if by_py is None:
                keys_py = f"dict.fromkeys({self._list_guard(list_py)})"
                pred_py = "_x == _k"
            else:
                keys_py = f"dict.fromkeys(({by_py} for _it in {self._list_guard(list_py)}))"
                pred_py = f"({by_py}) == _k"
            return (
                f'[{{"key": _k, '
                f'"count": sum(1 for _x in {self._list_guard(list_py)} if {pred_py}), '
                f'"items": [_x for _x in {self._list_guard(list_py)} if {pred_py}]}} '
                f"for _k in {keys_py}]"
            )

        if "at" in spec:
            arr_py = self.expr(spec["at"][0])
            idx_py = self.expr(spec["at"][1])
            return f"(({arr_py})[{idx_py}] if isinstance({idx_py}, int) and 0 <= {idx_py} < len({arr_py}) else None)"

        if "call" in spec:
            fn_name = spec["call"][0]
            defn = self.functions.get(fn_name)
            if defn is None or not isinstance(defn, dict):
                raise UnsupportedShapeError(f"call:{fn_name}")
            params = defn.get("params", [])
            args = spec["call"][1:]
            if len(args) != len(params):
                raise UnsupportedShapeError(f"call:{fn_name} arity")
            # 内联别名：参数表达式编译后绑定进别名 expr 的 binder
            sub = _Gen(self.constants, self.view_fns, self.view_defs, dict(self.binder), functions=self.functions)
            for pname, arg in zip(params, args):
                sub.binder[pname] = f"({self.expr(arg)})"
            return sub.expr(defn["expr"])

        raise UnsupportedShapeError(spec)

    def expr(self, spec: Any) -> str:
        """Compile a static expression spec into a Python expression string."""
        if isinstance(spec, list):
            # 字面量列表：逐元素编译（expr_eval 对 list 逐元素求值）
            return "[" + ", ".join(self.expr(s) for s in spec) + "]"
        if not isinstance(spec, dict):
            return repr(spec)

        if "const" in spec:
            return repr(spec["const"])

        if "var" in spec:
            return self._path(spec["var"])

        if "get" in spec:
            target, field = spec["get"]
            head = self.expr(target) if isinstance(target, dict) else self._path(target)
            # dict 守卫：非 dict 目标返回 None（与解释器一致）
            return f"(({head})[{field!r}] if isinstance(({head}), dict) else None)"

        if "template" in spec:
            parts: list[str] = []
            last = 0
            for m in _TEMPLATE_RE.finditer(spec["template"]):
                if m.start() > last:
                    parts.append(repr(spec["template"][last : m.start()]))
                inner = m.group(1).strip()
                var_py = self._path(inner if inner.startswith("$") else f"${inner}")
                parts.append(f"_s({var_py})")
                last = m.end()
            if last < len(spec["template"]):
                parts.append(repr(spec["template"][last:]))
            return "(" + " + ".join(parts) + ")"

        if "switch" in spec:
            # Single-evaluation if/elif/else chain: the input expression is
            # bound once via the walrus operator in the first branch, and a
            # matched branch's ``then`` is returned even when it evaluates to
            # a falsy value (0/""/[]/None) -- the old ``or``-chain let falsy
            # branch values fall through to the next case.  Default semantics
            # mirror the interpreter (expr_eval.py): a case with a "default"
            # key, or any case without a "case" key that has a "then".
            input_py = self.expr(spec.get("input", {"var": "$input"}))
            cases = [case for case in spec["switch"] if "case" in case]
            defaults = [case for case in spec["switch"] if "default" in case or ("case" not in case and "then" in case)]
            out = f"({self.expr(defaults[0]['then'])})" if defaults else "None"
            for case in reversed(cases[1:]):
                out = f"({self.expr(case['then'])} if _sw_in == {repr(case['case'])} else {out})"
            if cases:
                first = cases[0]
                out = f"({self.expr(first['then'])} if (_sw_in := {input_py}) == {repr(first['case'])} else {out})"
            return out

        for op, pyop in _CMP_OPS.items():
            if op in spec:
                return f"({self.expr(spec[op][0])} {pyop} {self.expr(spec[op][1])})"

        if "and" in spec:
            return "(" + " and ".join(f"bool({self.expr(sub)})" for sub in spec["and"]) + ")"

        if "or" in spec:
            return "(" + " or ".join(f"bool({self.expr(sub)})" for sub in spec["or"]) + ")"

        if "not" in spec:
            return f"(not bool({self.expr(spec['not'])}))"

        if "expr" in spec:
            return self._arith(spec["expr"])

        if "count" in spec:
            arg = spec["count"]
            if isinstance(arg, dict) and "query" in arg:
                arg = arg["query"]
            if isinstance(arg, dict) and arg.get("view", "") in self.view_fns:
                return self._count_py(arg)
            # 与解释器一致（expr_eval: len(list|dict)，其余 0）——dict 也是计数对象
            arg_py = self.expr(arg)
            return f"(len({arg_py}) if isinstance({arg_py}, (list, dict)) else 0)"

        if "query" in spec:
            return self._query_py(spec["query"])

        return self._expr_rest(spec)

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
        raw = path.lstrip("$").lstrip(".")
        parts = raw.split(".")
        if not parts:
            raise UnsupportedShapeError(path)
        first, rest = parts[0], parts[1:]

        # Inline constants at compile time.
        if first == "constants":
            node: Any = self.constants
            for r in rest:
                if not isinstance(node, dict) or r not in node:
                    raise UnsupportedShapeError(path)
                node = node[r]
            return repr(node)

        if first in _NODE_KEYS and self.node_scan:
            if len(rest) == 1 and rest[0] in _NODE_VALUE_FIELDS:
                return self.binder["_val"]
            raise UnsupportedShapeError(path)

        head = self.binder.get(first)
        if head is None:
            # 未知名 → 引擎 ctx 的数组平铺引用（$hand_p0 → state['_arrays']['hand_p0']）；
            # probe 验证兜底：编译产物与解释器不一致会被禁用
            head = f"state['_arrays'].get({first!r}, [])"
        for r in rest:
            head = self._field_get(head, r)
        return head

    @staticmethod
    def _field_get(head: str, part: str) -> str:
        """Append one path segment to a Python access expression."""
        if part.lstrip("-").isdigit():
            return f"{head}[{int(part)}]"
        return f"{head}[{part!r}]"

    def _arith(self, raw: str) -> str:
        """Compile a compact arithmetic string into a Python expression.

        ``$``-prefixed paths resolve through the binder (env/node/...);
        bare identifiers resolve to numeric constants.  Anything else is
        left to the interpreter (UnsupportedShapeError), matching its
        ``_eval_arithmetic`` fallback semantics.
        """

        def _repl(m: re.Match) -> str:
            if m.group(1) is not None:
                # $name[.rest]: compile only when resolvable in this context
                head = m.group(1).split(".")[0]
                if head == "constants" or head in self.binder:
                    return self._path_str(f"${m.group(1)}")
                raise UnsupportedShapeError(f"arith:${m.group(1)}")
            name = m.group(2)
            val = self.constants.get(name)
            if isinstance(val, (int, float)):
                return repr(val)
            raise UnsupportedShapeError(f"arith:{name}")

        return _ARITH_TOKEN_RE.sub(_repl, raw)

    # ── Query / aggregate specialisation ─────────────────────────────

    def _query_py(self, qexpr: dict) -> str:
        """Compile a query into a Python expression yielding entity dicts."""
        view = qexpr.get("view", "")
        fn_name = self.view_fns.get(view)
        if fn_name is None:
            raise UnsupportedShapeError(f"query:{view}")
        where = qexpr.get("filter") or qexpr.get("where")
        if where is None:
            return f"{fn_name}(state)"
        try:
            pred = self._scan_predicate(where)
            arr = self._view_array_expr(qexpr)
            return f"[{fn_name}(_i, _v) for _i, _v in enumerate({arr}) if {pred}]"
        except UnsupportedShapeError:
            entity_gen = _Gen(self.constants, self.view_fns, self.view_defs, {**self.binder, "node": "_n"})
            return f"[_n for _n in {fn_name}(state) if {entity_gen.expr(where)}]"

    def _count_py(self, qexpr: dict) -> str:
        """Compile ``count(query)`` into a Python aggregate expression."""
        try:
            pred = self._scan_predicate(qexpr.get("filter") or qexpr.get("where"))
            arr = self._view_array_expr(qexpr)
            return f"sum(1 for _v in {arr} if {pred})"
        except UnsupportedShapeError:
            return f"len({self._query_py(qexpr)})"

    def _scan_predicate(self, where: dict | None) -> str:
        """Compile a query predicate against the raw array (occupant-only)."""
        if where is None:
            return "True"
        gen = _Gen(self.constants, self.view_fns, self.view_defs, {**self.binder, "_val": "_v"}, node_scan=True)
        return gen.expr(where)

    def _view_array_expr(self, qexpr: dict) -> str:
        """Python expression for a view's source array, if it has one."""
        view_def = self.view_defs.get(qexpr.get("view", ""), {})
        arr_name = view_def.get("from", {}).get("array")
        if not arr_name:
            raise UnsupportedShapeError("view without array source")
        return f"state['_arrays'][{arr_name!r}]"


# ── View materialization codegen ─────────────────────────────────────


def _gen_view(vname: str, vdef: dict, constants: dict) -> str:
    """Generate ``_ent_<view>(_i, _v)`` + ``_materialize_<view>(state)``.

    Returns the source text or raises UnsupportedShapeError.
    """
    if not _IDENT_RE.match(vname):
        raise UnsupportedShapeError(f"view name: {vname}")
    source = vdef.get("from", {})
    stype = source.get("type", "literal")
    fields = vdef.get("fields", {})
    binder = {"node": "node", "self": "node", "cell": "node", "i": "_i"}
    if stype == "grid":
        binder.update(row="_r", col="_c")
    else:
        # Enum/literal entities have no _row/_col; interpreter defaults to 0.
        binder.update(row="0", col="0")

    if stype == "grid":
        cols = _resolve_cols(source.get("cols", {}), constants)
        ent_body = [
            f"    _bs = {cols}",
            "    _r = _i // _bs",
            "    _c = _i % _bs",
        ]
        mat_body = [
            f'    arr = state["_arrays"][{source.get("array", "")!r}]',
            "    _out = []",
            "    for _i, _v in enumerate(arr):",
            f"        _out.append(_ent_{vname}(_i, _v))",
            "    return _out",
        ]
    elif stype == "enum":
        ent_body = []
        mat_body = [
            f'    arr = state["_arrays"][{source.get("array", "")!r}]',
            "    _out = []",
            "    for _i, _v in enumerate(arr):",
            f"        _out.append(_ent_{vname}(_i, _v))",
            "    return _out",
        ]
    elif stype == "literal":
        raw_list = source.get("list", [])
        if isinstance(raw_list, dict) and raw_list.get("var", "").lstrip("$") == "players":
            list_src = 'state["_players"]'
        elif isinstance(raw_list, list):
            list_src = repr(raw_list)
        else:
            raise UnsupportedShapeError("literal list")
        ent_body = []
        mat_body = [
            f"    _src = {list_src}",
            "    _out = []",
            "    for _i, _v in enumerate(_src):",
            f"        _out.append(_ent_{vname}(_i, _v))",
            "    return _out",
        ]
    else:
        raise UnsupportedShapeError(f"view type: {stype}")

    if stype == "grid":
        ent_body.append("    node = {'_index': _i, 'value': _v, '_row': _r, '_col': _c}")
    else:
        ent_body.append("    node = {'_index': _i, 'value': _v, '_i': _i}")

    gen = _Gen(constants, {}, {}, binder)
    for fname, fdef in fields.items():
        if not _IDENT_RE.match(fname):
            raise UnsupportedShapeError(f"field name: {fname}")
        ent_body.append(f"    node[{fname!r}] = {gen.expr(fdef)}")
    ent_body.append("    return node")

    return (
        f"def _ent_{vname}(_i, _v):\n" + "\n".join(ent_body) + "\n\n"
        f"def _materialize_{vname}(state):\n" + "\n".join(mat_body) + "\n"
    )


def _resolve_cols(cols_expr: dict, constants: dict) -> int:
    """Resolve a grid ``cols`` spec to a compile-time integer."""
    try:
        gen = _Gen(constants, {}, {})
        val = _safe_eval(gen.expr(cols_expr))
    except Exception as exc:  # noqa: BLE001 — any resolution failure → fallback
        raise UnsupportedShapeError("cols") from exc
    if not isinstance(val, (int, float)) or val <= 0:
        return 1
    return int(val)


def _safe_eval(py_expr: str) -> Any:
    """Evaluate a generated constant expression.

    ``__builtins__`` is removed to keep the surface minimal, but this is
    NOT a security sandbox (literal attribute access can escape) — inputs
    are trusted static rules JSON, so the risk is acceptable.
    """
    return eval(py_expr, {"__builtins__": {}}, {})  # noqa: S307 — trusted consts only


# ── Terminal / actions / chance codegen ──────────────────────────────


def _gen_is_terminal(terminal: list[dict], constants: dict, view_fns: dict, view_defs: dict) -> str:
    """Generate ``is_terminal(state)`` covering all terminal rules."""
    if not terminal:
        raise UnsupportedShapeError("no terminal rules")
    gen = _Gen(constants, view_fns, view_defs, {"env": "env", "state": "state", "players": "players", "node": "node"})
    lines = [
        "def is_terminal(state):",
        '    env = state["env"]',
    ]
    for rule in terminal:
        cond = rule.get("condition")
        if not isinstance(cond, dict):
            raise UnsupportedShapeError(f"terminal: {rule.get('id')}")
        lines.append(f"    if {gen.expr(cond)}:")
        lines.append("        return True")
    lines.append("    return False")
    return "\n".join(lines) + "\n"


def _gen_legal_actions(
    actions: list[dict],
    queries: dict,
    constants: dict,
    view_fns: dict,
    view_defs: dict,
    functions: dict | None = None,
    engine_ref: object | None = None,
) -> str:
    """Generate ``legal_actions(state)`` for the supported template shapes.

    Per-template compilation: templates outside the supported subset are
    skipped and re-expanded at runtime by the engine's interpreter
    (``_engine._expand_missing``), so one unsupported action no longer
    disables the compiler for the whole ruleset.  Supported shapes:
      - params: 0 or 1 parameter; domain is a query ``ref`` or a dynamic
        ``array`` expression
      - ``legal``: ``const true`` or any compilable expression
      - expressions: const/var/get/template/switch/comparisons/boolean/
        arith/count/query/filter/at + inlined rule-alias ``call``

    ``engine_ref`` must be provided when any template is skipped: the
    generated code calls ``_engine._expand_missing``, so compiling without
    an engine would emit code that NameErrors at runtime.
    """
    if not actions:
        raise UnsupportedShapeError("no actions")
    lines = [
        "def legal_actions(state):",
        '    env = state["env"]',
        "    _out = []",
    ]
    skipped: list[str] = []
    for tmpl in actions:
        try:
            lines.extend(_gen_one_template(tmpl, queries, constants, view_fns, view_defs, functions))
        except UnsupportedShapeError:
            skipped.append(tmpl["id"])
    if not skipped and len(lines) == 3:
        raise UnsupportedShapeError("no supported action templates")
    if skipped:
        if engine_ref is None:
            raise UnsupportedShapeError(f"skipped action templates need engine fallback: {skipped}")
        lines.append(f"    _out.extend(_engine._expand_missing({skipped!r}, state))")
    lines.append("    return _out")
    return "\n".join(lines) + "\n"


def _gen_one_template(
    tmpl: dict, queries: dict, constants: dict, view_fns: dict, view_defs: dict, functions: dict | None = None
) -> list[str]:
    """Generate the body lines for one action template (raises UnsupportedShapeError)."""
    params = tmpl.get("params", {})
    if len(params) > 1:
        raise UnsupportedShapeError(f"action: {tmpl['id']} params > 1")
    phases = tmpl.get("phases", [])
    if not phases or not all(isinstance(p, str) for p in phases):
        raise UnsupportedShapeError(f"action: {tmpl['id']} phases")

    actor = tmpl.get("actor", {"var": "$env.turn"})
    ck = tmpl.get("canonicalKey", {"template": tmpl["id"]})
    legal = tmpl.get("legal", {"const": True})
    bind = {"env": "env", "state": "state", "players": "players", "node": "node", "cell": "node"}
    for pname in params:
        bind[pname] = "node"

    gen = _Gen(constants, view_fns, view_defs, bind, functions=functions)
    try:
        actor_py = gen.expr(actor)
        ck_py = gen.expr(ck)
        legal_py = gen.expr(legal) if legal != {"const": True} else None
    except UnsupportedShapeError as exc:
        raise UnsupportedShapeError(f"action: {tmpl['id']}: {exc}") from exc

    # 参数 domain：query ref（视图实体）/ 动态 array 表达式 / 任意表达式
    param_py: str | None = None
    param_name: str | None = None
    pre_lines: list[str] = []
    if params:
        pname, pdef = next(iter(params.items()))
        domain = pdef.get("domain", [])
        if isinstance(domain, dict) and "ref" in domain:
            qname = domain["ref"]
            qdef = queries.get(qname)
            if qdef is None:
                raise UnsupportedShapeError(f"action: {tmpl['id']} unknown query {qname}")
            try:
                param_py = gen._query_py(
                    {
                        "view": qdef.get("view", ""),  # noqa: SLF001
                        "filter": qdef.get("filter") or qdef.get("where"),
                    }
                )
            except UnsupportedShapeError as exc:
                raise UnsupportedShapeError(f"action: {tmpl['id']}: {exc}") from exc
        elif isinstance(domain, dict) and "array" in domain:
            try:
                arr_py = gen.expr(domain["array"])
            except UnsupportedShapeError as exc:
                raise UnsupportedShapeError(f"action: {tmpl['id']}: {exc}") from exc
            param_py = f'state["_arrays"].get({arr_py}) or []'
        elif isinstance(domain, dict) and "expr" in domain:
            try:
                expr_py = gen.expr(domain["expr"])
            except UnsupportedShapeError as exc:
                raise UnsupportedShapeError(f"action: {tmpl['id']}: {exc}") from exc
            pre_lines.append(f"        _dom = {expr_py}")
            param_py = "_dom if isinstance(_dom, list) else ([_dom] if _dom is not None else [])"
        else:
            raise UnsupportedShapeError(f"action: {tmpl['id']} domain not query ref / array / expr")
        param_name = pname

    phase_guard = " or ".join(f"env['phase'] == {p!r}" for p in phases)
    lines = [f"    if {phase_guard}:"]
    if param_name is None:
        if legal_py is not None:
            lines.append(f"        if {legal_py}:")
        lines.append(
            f"            _out.append(ActionInstance({tmpl['id']!r}, "
            f"{tmpl.get('type', 'action')!r}, {actor_py}, {{}}, {ck_py}))"
        )
        return lines
    lines.extend(pre_lines)
    lines.append(f"        for node in {param_py}:")
    if legal_py is not None:
        lines.append(f"            if {legal_py}:")
    lines.append(
        f"                _out.append(ActionInstance({tmpl['id']!r}, "
        f"{tmpl.get('type', 'action')!r}, {actor_py}, "
        f"{{{param_name!r}: node}}, {ck_py}))"
    )
    return lines


def _gen_chance(chance: list[dict], constants: dict, view_fns: dict, view_defs: dict) -> str:
    """Generate ``chance_outcomes(state)`` for explicit probability tables."""
    if not chance:
        raise UnsupportedShapeError("no chance templates")
    lines = [
        "def chance_outcomes(state):",
        '    env = state["env"]',
        "    _out = []",
    ]
    first = True
    for ct in chance:
        phases = ct.get("phases", [])
        prob = ct.get("probability", {})
        explicit = prob.get("explicit") if isinstance(prob, dict) else None
        if not phases or not explicit:
            raise UnsupportedShapeError("chance: uniform/nonexplicit not supported")
        effect_map = ct.get("effectMap", {})
        ck_tmpl = ct.get("canonicalKey", {"template": "chance:{outcome}"})
        phase_guard = " or ".join(f"env['phase'] == {p!r}" for p in phases)
        # First matching template wins: an if/elif chain mirrors the
        # interpreter's first-match semantics (engine._interp_chance_outcomes);
        # the old independent ``if`` blocks emitted a union when templates
        # shared a phase.
        lines.append(f"    {'if' if first else 'elif'} {phase_guard}:")
        first = False
        for entry in explicit:
            outcome_val = entry.get("outcome", entry.get("value"))
            prob_expr = entry.get("prob", entry.get("probability"))
            if not isinstance(prob_expr, (int, float)):
                raise UnsupportedShapeError("chance: non-const probability")
            # ``$outcome`` 绑定原始值（与解释器 ctx["outcome"]=outcome_val 一致）；
            # 字符串化会让数值 outcome 的 ``$outcome == 3`` 类 canonicalKey 错配。
            gen = _Gen(constants, view_fns, view_defs, {"env": "env", "state": "state", "outcome": repr(outcome_val)})
            ck_py = gen.expr(ck_tmpl)
            effect_ref = effect_map.get(str(outcome_val), f"do_{outcome_val}")
            lines.append(
                f"        _out.append(ChanceOutcome({str(outcome_val)!r}, {float(prob_expr)}, {effect_ref!r}, {ck_py}))"
            )
    lines.append("    return _out")
    return "\n".join(lines) + "\n"


# ── Top-level compiler ───────────────────────────────────────────────


class CompiledArtifacts:
    """Compiled rule artifacts; probe-validated against the interpreter."""

    __slots__ = ("materialize", "is_terminal", "legal_actions", "chance_outcomes", "_views", "_view_defs")

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
            if nt == "player":
                actions = engine.get_legal_actions(state)
                if not actions:
                    break
                state = engine.apply_action(state, actions[0])
                while engine.get_node_type(state) == "chance":
                    _, state = engine.sample_chance(state)
            elif nt == "chance":
                _, state = engine.sample_chance(state)
            else:
                break
            probes.append(state)

        try:
            for p in probes:
                if self.is_terminal is not None and self.is_terminal(p) != engine.is_terminal(p):
                    self.is_terminal = None
                if self.legal_actions is not None:
                    # 集合语义对比：部分编译时 fallback 展开的顺序可能与
                    # 解释器不同，但动作集合必须完全一致
                    mine = sorted(a.canonical_key for a in self.legal_actions(p))
                    theirs = sorted(a.canonical_key for a in engine.get_legal_actions(p))
                    if mine != theirs:
                        self.legal_actions = None
                if self.chance_outcomes is not None:
                    mine = [(o.key, o.probability, o.effect_ref, o.canonical_key) for o in self.chance_outcomes(p)]
                    theirs = [
                        (o.key, o.probability, o.effect_ref, o.canonical_key) for o in engine.get_chance_outcomes(p)
                    ]
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

    def compile(self, rules: dict, engine=None) -> CompiledArtifacts:
        constants = rules.get("constants", {})
        artifacts = CompiledArtifacts()
        src_parts = ['def _s(x):\n    return "" if x is None else str(x)\n']
        view_fns: dict[str, str] = {}
        # 供部分编译的运行时 fallback 使用（_expand_missing）
        self._engine_ref = engine

        for vname, vdef in rules.get("derivedViews", {}).items():
            try:
                src = _gen_view(vname, vdef, constants)
            except UnsupportedShapeError:
                continue
            artifacts._view_defs[vname] = vdef
            view_fns[vname] = f"_materialize_{vname}"
            src_parts.append(src)

        try:
            src_parts.append(_gen_is_terminal(rules.get("terminal", []), constants, view_fns, artifacts._view_defs))
        except UnsupportedShapeError:
            pass
        try:
            src_parts.append(
                _gen_legal_actions(
                    rules.get("actions", []),
                    rules.get("queries", {}),
                    constants,
                    view_fns,
                    artifacts._view_defs,
                    rules.get("functions", {}),
                    engine_ref=engine,
                )
            )
        except UnsupportedShapeError:
            pass
        try:
            src_parts.append(_gen_chance(rules.get("chance", []), constants, view_fns, artifacts._view_defs))
        except UnsupportedShapeError:
            pass

        namespace: dict[str, Any] = {
            "ActionInstance": ActionInstance,
            "ChanceOutcome": ChanceOutcome,
        }
        if self._engine_ref is not None:
            namespace["_engine"] = self._engine_ref
        source = "\n".join(src_parts)
        exec(compile(source, "<rules_codegen>", "exec"), namespace)  # noqa: S102 — generated
        for vname, fn_name in view_fns.items():
            artifacts._views[vname] = namespace[fn_name]
        artifacts.is_terminal = namespace.get("is_terminal")
        artifacts.legal_actions = namespace.get("legal_actions")
        artifacts.chance_outcomes = namespace.get("chance_outcomes")
        return artifacts
