"""Expression evaluator for the v5.1 rules engine.

Evaluates a JSON-based expression tree against a context dict.
Expanded from v5.0 with pure-mathematical primitives (no game-specific
builtins): choose/range/sort/group/distinct/contains/sum/max/min/at,
plus alias ``call`` (rules ``functions`` become real inline definitions).

Core expression types:
  const, var, get, eq, and, or, not, gt, lt
  if, switch, call, query, count, template, concat
  filter, any, all, map
  choose, range, sort, group, distinct, contains, sum, max, min, at
  expr — compact arithmetic string "y * board_size + x"
"""

from __future__ import annotations

import re
from typing import Any, Callable

# Precompiled regexes (hot path: called millions of times per MCTS move).
_ARITH_PREFIX_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_.]*)")
_ARITH_BARE_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_.]*)\b")
_TEMPLATE_RE = re.compile(r"\{([^}]+)\}")

# Alias bodies may not recurse; this caps any (accidentally) deep chain.
_MAX_CALL_DEPTH = 32


class ExprEvaluator:
    """Evaluates expression trees against a context dictionary."""

    def __init__(self):
        self._functions: dict[str, Any] = {}
        self._cyclic: set[str] = set()
        self._call_depth = 0

    def register_function(self, name: str, fn: Callable):
        """Register a plain callable that expressions can call."""
        self._functions[name] = fn

    def set_functions(self, defs: dict) -> None:
        """Populate the registry from a rules ``functions`` block.

        Each entry is either a callable (engine-internal) or an alias
        definition ``{"params": [...], "expr": {...}}``.  Aliases are
        bound by parameter name at call time; recursion cycles among
        aliases are detected statically and rejected with a
        ``RecursionError`` if ever invoked.
        """
        self._functions = {}
        self._cyclic = set()
        for name, defn in defs.items():
            if isinstance(defn, dict):
                params = defn.get("params")
                body = defn.get("expr")
                if not isinstance(params, list) or not isinstance(body, dict):
                    # v5.0-style declaration stub (metadata only) — skip it;
                    # a call to it later fails with "Unknown function".
                    continue
            self._functions[name] = defn

        # Static recursion-cycle detection (Tarjan-lite DFS over the
        # call graph).  Cyclic aliases keep a guard that raises on call;
        # the interpreter call path also enforces ``_MAX_CALL_DEPTH``.
        visiting: set[str] = set()
        done: set[str] = set()

        def _visit(name: str):
            if name in done:
                return
            if name in visiting:
                raise RecursionError(f"Alias recursion cycle involving {name!r}")
            defn = self._functions.get(name)
            if isinstance(defn, dict):
                visiting.add(name)
                for callee in _collect_call_names(defn.get("expr", {})):
                    _visit(callee)
                visiting.remove(name)
            done.add(name)

        for name in list(self._functions):
            try:
                _visit(name)
            except RecursionError:
                # Only the members of the cycle are unusable; others keep working.
                for cname in visiting:
                    self._cyclic.add(cname)
                visiting.clear()

    def _alias_call(self, fn_name: str, args: list, ctx: dict) -> Any:
        """Evaluate an alias definition with ``args`` bound to its params."""
        defn = self._functions[fn_name]
        if fn_name in self._cyclic:
            raise RecursionError(f"Alias recursion cycle involving {fn_name!r}")
        params = defn["params"]
        if len(args) != len(params):
            raise ValueError(f"Alias {fn_name!r}: expected {len(params)} args, got {len(args)}")
        self._call_depth += 1
        try:
            if self._call_depth > _MAX_CALL_DEPTH:
                raise RecursionError(f"Alias call depth exceeded: {fn_name!r}")
            sub = dict(ctx)
            for pname, pval in zip(params, args):
                sub[pname] = pval
                sub[f"${pname}"] = pval
            return self.eval(defn["expr"], sub)
        finally:
            self._call_depth -= 1

    def compile(self, spec: Any) -> callable:
        """Precompile a static expression tree into ``fn(ctx)``.

        Hot expression types (const/var/get/template/switch/eq/neq/gt/
        gte/lt/lte/and/or/not/if) are inlined as closures with identical
        semantics to :meth:`eval`; everything else falls back to the
        interpreter, so behavior is preserved by construction.
        """
        if not isinstance(spec, dict):
            return lambda ctx: spec

        # ── Literals ────────────────────────────────────────────────────
        if "const" in spec:
            value = spec["const"]
            return lambda ctx: value

        if "var" in spec:
            return self._compile_path(spec["var"])

        if "get" in spec:
            path = spec["get"]
            return self._compile_path(path)

        # ── String ops ──────────────────────────────────────────────────
        if "template" in spec:
            tmpl = spec["template"]
            parts: list = []
            last = 0
            for m in _TEMPLATE_RE.finditer(tmpl):
                if m.start() > last:
                    parts.append(tmpl[last : m.start()])
                parts.append(self._compile_path(m.group(1).strip()))
                last = m.end()
            if last < len(tmpl):
                parts.append(tmpl[last:])

            def _template(ctx: dict, parts: list = parts) -> str:
                out = []
                for p in parts:
                    if callable(p):
                        value = p(ctx)
                        out.append(str(value) if value is not None else "")
                    else:
                        out.append(p)
                return "".join(out)

            return _template

        if "concat" in spec:
            children = [self.compile(item) for item in spec["concat"]]

            def _concat(ctx: dict, children: list = children) -> Any:
                items = [fn(ctx) for fn in children]
                return _concat_values(items)

            return _concat

        # ── Arithmetic shorthand ────────────────────────────────────────
        if "expr" in spec:

            def _arith(ctx: dict, spec: dict = spec) -> Any:
                return self._eval_arithmetic(spec["expr"], ctx)

            return _arith

        # ── Conditionals ────────────────────────────────────────────────
        if "if" in spec:
            if_body = spec["if"]
            if isinstance(if_body, dict) and "cond" in if_body:
                cond_fn = self.compile(if_body["cond"])
                then_fn = self.compile(if_body.get("then", True))
                else_fn = self.compile(if_body.get("else", None))
            else:
                cond_fn = self.compile(if_body)
                then_fn = self.compile(spec.get("then", True))
                else_fn = self.compile(spec.get("else", None))

            def _if(ctx: dict, cond_fn=cond_fn, then_fn=then_fn, else_fn=else_fn) -> Any:
                return then_fn(ctx) if cond_fn(ctx) else else_fn(ctx)

            return _if

        if "switch" in spec:
            cases = []
            default_fn = None
            for case in spec["switch"]:
                if "case" in case:
                    cases.append((case["case"], self.compile(case["then"])))
                else:
                    default_fn = self.compile(case.get("then"))
            input_fn = self.compile(spec.get("input", {"var": "$input"}))

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

        # ── Arithmetic nodes (pure math ops over evaluated operands) ────
        if "add" in spec or "sub" in spec or "mul" in spec or "div" in spec:
            op = next(k for k in ("add", "sub", "mul", "div") if k in spec)
            left_fn = self.compile(spec[op][0])
            right_fn = self.compile(spec[op][1])

            def _arith_node(ctx: dict, left_fn=left_fn, right_fn=right_fn, op=op) -> Any:
                a, b = left_fn(ctx), right_fn(ctx)
                try:
                    if op == "add":
                        return a + b
                    if op == "sub":
                        return a - b
                    if op == "mul":
                        return a * b
                    if op == "div":
                        return int(a) // int(b)
                except (TypeError, ValueError, ZeroDivisionError):
                    return None
                return None

            return _arith_node

        # ── Logical / comparison ops ────────────────────────────────────
        if "eq" in spec or "neq" in spec or "gt" in spec or "gte" in spec or "lt" in spec or "lte" in spec:
            op = next(k for k in ("eq", "neq", "gt", "gte", "lt", "lte") if k in spec)
            left_fn = self.compile(spec[op][0])
            right_fn = self.compile(spec[op][1])
            cmp_map = {
                "eq": lambda a, b: a == b,
                "neq": lambda a, b: a != b,
                "gt": lambda a, b: a > b,
                "gte": lambda a, b: a >= b,
                "lt": lambda a, b: a < b,
                "lte": lambda a, b: a <= b,
            }
            cmp_fn = cmp_map[op]

            def _cmp(ctx: dict, left_fn=left_fn, right_fn=right_fn, cmp_fn=cmp_fn) -> bool:
                return cmp_fn(left_fn(ctx), right_fn(ctx))

            return _cmp

        if "and" in spec:
            children = [self.compile(sub) for sub in spec["and"]]

            def _and(ctx: dict, children: list = children) -> bool:
                return all(fn(ctx) for fn in children)

            return _and

        if "or" in spec:
            children = [self.compile(sub) for sub in spec["or"]]

            def _or(ctx: dict, children: list = children) -> bool:
                return any(fn(ctx) for fn in children)

            return _or

        if "not" in spec:
            inner_fn = self.compile(spec["not"])

            def _not(ctx: dict, inner_fn=inner_fn) -> bool:
                return not inner_fn(ctx)

            return _not

        # ── Collection ops (compiled closures) ──────────────────────────
        if "filter" in spec:
            fspec = spec["filter"]
            list_fn = self.compile(fspec["list"])
            as_var = fspec.get("as", "$node")
            where_fn = self.compile(fspec["where"])

            def _filter(ctx: dict, list_fn=list_fn, where_fn=where_fn, as_var=as_var) -> list:
                items = list_fn(ctx)
                if not isinstance(items, list):
                    return []
                out = []
                for item in items:
                    if where_fn({**ctx, as_var: item}):
                        out.append(item)
                return out

            return _filter

        if "any" in spec:
            aspec = spec["any"]
            list_fn = self.compile(aspec["list"])
            as_var = aspec.get("as", "$node")
            where_fn = self.compile(aspec["where"])

            def _any(ctx: dict, list_fn=list_fn, where_fn=where_fn, as_var=as_var) -> bool:
                items = list_fn(ctx)
                if not isinstance(items, list):
                    return False
                for item in items:
                    if where_fn({**ctx, as_var: item}):
                        return True
                return False

            return _any

        if "all" in spec:
            aspec = spec["all"]
            list_fn = self.compile(aspec["list"])
            as_var = aspec.get("as", "$node")
            where_fn = self.compile(aspec["where"])

            def _all(ctx: dict, list_fn=list_fn, where_fn=where_fn, as_var=as_var) -> bool:
                items = list_fn(ctx)
                if not isinstance(items, list):
                    return True
                for item in items:
                    if not where_fn({**ctx, as_var: item}):
                        return False
                return True

            return _all

        if "map" in spec:
            mspec = spec["map"]
            list_fn = self.compile(mspec["list"])
            as_var = mspec.get("as", "$node")
            map_fn = self.compile(mspec["expr"])

            def _map(ctx: dict, list_fn=list_fn, map_fn=map_fn, as_var=as_var) -> list:
                items = list_fn(ctx)
                if not isinstance(items, list):
                    return []
                return [map_fn({**ctx, as_var: item}) for item in items]

            return _map

        if "count" in spec:
            inner_fn = self.compile(spec["count"])

            def _count(ctx: dict, inner_fn=inner_fn) -> int:
                value = inner_fn(ctx)
                return len(value) if isinstance(value, (list, dict)) else 0

            return _count

        if "distinct" in spec:
            inner_fn = self.compile(spec["distinct"])

            def _distinct(ctx: dict, inner_fn=inner_fn) -> list:
                value = inner_fn(ctx)
                if not isinstance(value, list):
                    return []
                out: list = []
                seen: set = set()
                for item in value:
                    try:
                        if item in seen:
                            continue
                        seen.add(item)
                    except TypeError:
                        pass  # unhashable item — fall back to linear scan
                    if item not in out:
                        out.append(item)
                return out

            return _distinct

        if "contains" in spec:
            cont_fn = self.compile(spec["contains"][0])
            item_fn = self.compile(spec["contains"][1])

            def _contains(ctx: dict, cont_fn=cont_fn, item_fn=item_fn) -> bool:
                container = cont_fn(ctx)
                if not isinstance(container, list):
                    return False
                return item_fn(ctx) in container

            return _contains

        for _op in ("sum", "max", "min"):
            if _op in spec:
                inner_fn = self.compile(spec[_op])
                op = _op

                def _agg(ctx: dict, inner_fn=inner_fn, op=op) -> Any:
                    value = inner_fn(ctx)
                    if not isinstance(value, list):
                        return 0 if op == "sum" else None
                    if op == "sum":
                        return sum(value)
                    if not value:
                        return None
                    try:
                        return max(value) if op == "max" else min(value)
                    except TypeError:
                        return None

                return _agg

        if "at" in spec:
            cont_fn = self.compile(spec["at"][0])
            idx_fn = self.compile(spec["at"][1])

            def _at(ctx: dict, cont_fn=cont_fn, idx_fn=idx_fn) -> Any:
                container = cont_fn(ctx)
                idx = idx_fn(ctx)
                if isinstance(container, dict):
                    return container.get(idx)
                if isinstance(container, (list, str)) and isinstance(idx, (int, float)) and not isinstance(idx, bool):
                    i = int(idx)
                    # Negative indices are out of bounds: win-check windows
                    # rely on ``at`` returning None past the array edges.
                    if 0 <= i < len(container):
                        return container[i]
                return None

            return _at

        if "range" in spec:
            rspec = spec["range"]
            from_fn = self.compile(rspec.get("from", {"const": 0}))
            to_fn = self.compile(rspec["to"])
            step_fn = self.compile(rspec.get("step", {"const": 1}))

            def _range(ctx: dict, from_fn=from_fn, to_fn=to_fn, step_fn=step_fn) -> list:
                try:
                    a, b, s = int(from_fn(ctx)), int(to_fn(ctx)), int(step_fn(ctx))
                except (TypeError, ValueError):
                    return []
                if s == 0:
                    return []
                return list(range(a, b, s))

            return _range

        if "sort" in spec:
            sspec = spec["sort"]
            list_fn = self.compile(sspec["list"])
            by_fn = self.compile(sspec["by"]) if "by" in sspec else None
            reverse = bool(sspec.get("reverse", False))

            def _sort(ctx: dict, list_fn=list_fn, by_fn=by_fn, reverse=reverse) -> list:
                items = list_fn(ctx)
                if not isinstance(items, list):
                    return []
                if by_fn is None:
                    try:
                        return sorted(items, reverse=reverse)
                    except TypeError:
                        return list(items)
                keyed = [(by_fn({**ctx, "$node": item, "$item": item}), item) for item in items]
                keyed.sort(key=lambda kv: kv[0], reverse=reverse)
                return [item for _, item in keyed]

            return _sort

        if "group" in spec:
            gspec = spec["group"]
            list_fn = self.compile(gspec["list"])
            by_fn = self.compile(gspec["by"]) if "by" in gspec else None

            def _group(ctx: dict, list_fn=list_fn, by_fn=by_fn) -> list:
                items = list_fn(ctx)
                if not isinstance(items, list):
                    return []
                buckets: dict = {}
                order: list = []
                for item in items:
                    key = item if by_fn is None else by_fn({**ctx, "$node": item, "$item": item})
                    if key not in buckets:
                        buckets[key] = {"key": key, "count": 0, "items": []}
                        order.append(key)
                    buckets[key]["count"] += 1
                    buckets[key]["items"].append(item)
                return [buckets[k] for k in order]

            return _group

        if "choose" in spec:
            return self._compile_choose(spec["choose"])

        if "call" in spec:
            fn_name = spec["call"][0]
            defn = self._functions.get(fn_name)
            if isinstance(defn, dict) and "expr" in defn and fn_name not in self._cyclic:
                params = defn["params"]
                arg_fns = [self.compile(arg) for arg in spec["call"][1:]]
                body_fn = self.compile(defn["expr"])

                def _alias(ctx: dict, arg_fns=arg_fns, body_fn=body_fn, params=params) -> Any:
                    sub = dict(ctx)
                    for pname, argfn in zip(params, arg_fns):
                        value = argfn(ctx)
                        sub[pname] = value
                        sub[f"${pname}"] = value
                    return body_fn(sub)

                return _alias

            def _runtime_call(ctx: dict, spec: dict = spec) -> Any:
                return self.eval(spec, ctx)

            return _runtime_call

        # ── Fallback: interpreter for exotic types ──────────────────────
        def _fallback(ctx: dict, spec: dict = spec) -> Any:
            return self.eval(spec, ctx)

        return _fallback

    def _compile_choose(self, cspec: dict) -> Callable:
        """Compile a ``choose`` spec into a backtracking closure.

        Semantics (mirrors the interpreter): the input list is sorted by
        value and deduplicated, then all k-combinations are enumerated.
        ``prefix`` prunes partial combinations (monotone contract: if a
        partial combination fails ``prefix``, every extension fails too);
        ``where`` filters full combinations.  With ``then`` + ``agg`` the
        best (max/min) ``then`` value over satisfying combinations is
        returned; without ``then``, existence (bool) is returned.
        """
        items_fn = self.compile(cspec["items"])
        k_fn = self.compile(cspec["k"])
        as_var = cspec.get("as", "$c")
        prefix_fn = self.compile(cspec["prefix"]) if "prefix" in cspec else None
        where_fn = self.compile(cspec["where"]) if "where" in cspec else None
        then_fn = self.compile(cspec["then"]) if "then" in cspec else None
        maximize = cspec.get("agg", "max") != "min"

        def _choose(
            ctx: dict,
            items_fn=items_fn,
            k_fn=k_fn,
            prefix_fn=prefix_fn,
            where_fn=where_fn,
            then_fn=then_fn,
            as_var=as_var,
            maximize=maximize,
        ) -> Any:
            items = items_fn(ctx)
            if not isinstance(items, list):
                return None if then_fn is not None else False
            try:
                items = sorted(items)
            except TypeError:
                pass  # unorderable items — keep input order
            uniq: list = []
            seen: set = set()
            for item in items:
                try:
                    if item in seen:
                        continue
                    seen.add(item)
                except TypeError:
                    pass
                if item not in uniq:
                    uniq.append(item)
            try:
                k = int(k_fn(ctx))
            except (TypeError, ValueError):
                return None if then_fn is not None else False
            k = max(0, min(k, len(uniq)))

            best: Any = None
            found = False
            combo: list = []

            def _rec(i: int):
                nonlocal best, found
                if len(combo) == k:
                    sub = {**ctx, as_var: list(combo), "$node": list(combo)}
                    if where_fn is not None and not where_fn(sub):
                        return
                    found = True
                    if then_fn is not None:
                        value = then_fn(sub)
                        if best is None or (value > best if maximize else value < best):
                            best = value
                    return
                if i >= len(uniq):
                    return
                # Branch 1: include uniq[i].  Prefix gates *deeper* search
                # only — a full-length combination is judged by ``where``.
                combo.append(uniq[i])
                if len(combo) == k or prefix_fn is None or prefix_fn({**ctx, as_var: list(combo)}):
                    _rec(i + 1)
                combo.pop()
                # Branch 2: skip uniq[i].
                _rec(i + 1)

            _rec(0)
            if then_fn is not None:
                return best
            return found

        return _choose

    def _compile_path(self, path: str | list) -> callable:
        """Compile a ``var``/``get`` path spec into a ctx closure."""
        if isinstance(path, list):
            raw = path[0]
            rest = path[1:]
            if isinstance(raw, dict):
                head_fn = self.compile(raw)
            else:
                head_fn = self._compile_path(raw)

            def _get(ctx: dict, head_fn=head_fn, rest=rest) -> Any:
                obj = head_fn(ctx)
                for p in rest:
                    if isinstance(obj, dict):
                        obj = obj.get(p)
                    elif isinstance(obj, (list, tuple)) and p.lstrip("-").isdigit():
                        idx = int(p)
                        obj = obj[idx] if -len(obj) <= idx < len(obj) else None
                    else:
                        return None
                return obj

            return _get
        parts = path.lstrip("$").lstrip(".").split(".")

        def _path(ctx: dict, parts: list = parts) -> Any:
            obj = ctx
            for p in parts:
                if isinstance(obj, dict):
                    if p in obj:
                        obj = obj[p]
                    elif f"${p}" in obj:
                        obj = obj[f"${p}"]
                    else:
                        return None
                elif isinstance(obj, (list, tuple)) and p.lstrip("-").isdigit():
                    idx = int(p)
                    obj = obj[idx] if -len(obj) <= idx < len(obj) else None
                else:
                    return None
            return obj

        return _path

    def eval(self, expr: Any, ctx: dict) -> Any:
        """Evaluate an expression tree against the given context.

        ``expr`` is read-only here (no mutation anywhere in eval), so the
        shallow copy is unnecessary — skipping it saves ~1M dict copies
        per MCTS move.
        """
        if not isinstance(expr, dict):
            return expr

        # ── Literals ────────────────────────────────────────────────────

        if "const" in expr:
            return expr["const"]

        if "var" in expr:
            return self._resolve_path(ctx, expr["var"])

        if "get" in expr:
            return self._resolve_path(ctx, expr["get"])

        # ── String ops ──────────────────────────────────────────────────

        if "template" in expr:
            template = expr["template"]
            parts = []
            last_end = 0
            for m in _TEMPLATE_RE.finditer(template):
                parts.append(template[last_end : m.start()])
                val = self._resolve_path(ctx, m.group(1).strip())
                parts.append(str(val) if val is not None else "")
                last_end = m.end()
            parts.append(template[last_end:])
            return "".join(parts)

        if "concat" in expr:
            items = [self.eval(item, ctx) for item in expr["concat"]]
            return _concat_values(items)

        # ── Arithmetic shorthand ────────────────────────────────────────

        if "expr" in expr:
            return self._eval_arithmetic(expr["expr"], ctx)

        # ── Conditionals ────────────────────────────────────────────────

        if "if" in expr:
            if_body = expr["if"]
            if isinstance(if_body, dict) and "cond" in if_body:
                cond = self.eval(if_body["cond"], ctx)
                then_val = if_body.get("then", True)
                else_val = if_body.get("else", None)
            else:
                cond = self.eval(if_body, ctx)
                then_val = expr.get("then", True)
                else_val = expr.get("else", None)
            return self.eval(then_val, ctx) if cond else (self.eval(else_val, ctx) if else_val is not None else None)

        if "switch" in expr:
            input_val = self.eval(expr.get("input", {"var": "$input"}), ctx)
            for case in expr["switch"]:
                if case.get("case") == input_val:
                    return self.eval(case["then"], ctx)
            # else/default branch
            for case in expr["switch"]:
                if "default" in case or ("case" not in case and "then" in case):
                    return self.eval(case["then"], ctx)
            return None

        # ── Arithmetic nodes (pure math ops over evaluated operands) ────

        if "add" in expr:
            return self.eval(expr["add"][0], ctx) + self.eval(expr["add"][1], ctx)

        if "sub" in expr:
            return self.eval(expr["sub"][0], ctx) - self.eval(expr["sub"][1], ctx)

        if "mul" in expr:
            return self.eval(expr["mul"][0], ctx) * self.eval(expr["mul"][1], ctx)

        if "div" in expr:
            try:
                return int(self.eval(expr["div"][0], ctx)) // int(self.eval(expr["div"][1], ctx))
            except (TypeError, ValueError, ZeroDivisionError):
                return None

        # ── Logical ops ─────────────────────────────────────────────────

        if "eq" in expr:
            return self.eval(expr["eq"][0], ctx) == self.eval(expr["eq"][1], ctx)

        if "neq" in expr:
            return self.eval(expr["neq"][0], ctx) != self.eval(expr["neq"][1], ctx)

        if "gt" in expr:
            return self.eval(expr["gt"][0], ctx) > self.eval(expr["gt"][1], ctx)

        if "gte" in expr:
            return self.eval(expr["gte"][0], ctx) >= self.eval(expr["gte"][1], ctx)

        if "lt" in expr:
            return self.eval(expr["lt"][0], ctx) < self.eval(expr["lt"][1], ctx)

        if "lte" in expr:
            return self.eval(expr["lte"][0], ctx) <= self.eval(expr["lte"][1], ctx)

        if "and" in expr:
            return all(self.eval(sub, ctx) for sub in expr["and"])

        if "or" in expr:
            return any(self.eval(sub, ctx) for sub in expr["or"])

        if "not" in expr:
            return not self.eval(expr["not"], ctx)

        # ── Collection ops ──────────────────────────────────────────────

        if "filter" in expr:
            items = self.eval(expr["filter"]["list"], ctx)
            as_var = expr["filter"].get("as", "$node")
            where = expr["filter"]["where"]
            if not isinstance(items, list):
                return []
            results = []
            for item in items:
                item_ctx = {**ctx, as_var: item}
                if self.eval(where, item_ctx):
                    results.append(item)
            return results

        if "any" in expr:
            items = self.eval(expr["any"]["list"], ctx)
            as_var = expr["any"].get("as", "$node")
            where = expr["any"]["where"]
            if not isinstance(items, list):
                return False
            for item in items:
                item_ctx = {**ctx, as_var: item}
                if self.eval(where, item_ctx):
                    return True
            return False

        if "all" in expr:
            items = self.eval(expr["all"]["list"], ctx)
            as_var = expr["all"].get("as", "$node")
            where = expr["all"]["where"]
            if not isinstance(items, list):
                return True
            for item in items:
                item_ctx = {**ctx, as_var: item}
                if not self.eval(where, item_ctx):
                    return False
            return True

        if "map" in expr:
            items = self.eval(expr["map"]["list"], ctx)
            as_var = expr["map"].get("as", "$node")
            map_expr = expr["map"]["expr"]
            if not isinstance(items, list):
                return []
            results = []
            for item in items:
                item_ctx = {**ctx, as_var: item}
                results.append(self.eval(map_expr, item_ctx))
            return results

        # ── v5.1 collection primitives (pure math ops) ──────────────────

        if "distinct" in expr:
            value = self.eval(expr["distinct"], ctx)
            if not isinstance(value, list):
                return []
            out = []
            for item in value:
                if item not in out:
                    out.append(item)
            return out

        if "contains" in expr:
            container = self.eval(expr["contains"][0], ctx)
            if not isinstance(container, list):
                return False
            return self.eval(expr["contains"][1], ctx) in container

        if "sum" in expr:
            value = self.eval(expr["sum"], ctx)
            return sum(value) if isinstance(value, list) else 0

        if "max" in expr:
            value = self.eval(expr["max"], ctx)
            if not isinstance(value, list) or not value:
                return None
            try:
                return max(value)
            except TypeError:
                return None

        if "min" in expr:
            value = self.eval(expr["min"], ctx)
            if not isinstance(value, list) or not value:
                return None
            try:
                return min(value)
            except TypeError:
                return None

        if "at" in expr:
            container = self.eval(expr["at"][0], ctx)
            idx = self.eval(expr["at"][1], ctx)
            if isinstance(container, dict):
                return container.get(idx)
            if isinstance(container, (list, str)) and isinstance(idx, (int, float)) and not isinstance(idx, bool):
                i = int(idx)
                # Negative indices are out of bounds (see compiled ``at``).
                if 0 <= i < len(container):
                    return container[i]
            return None

        if "range" in expr:
            rspec = expr["range"]
            try:
                a = int(self.eval(rspec.get("from", {"const": 0}), ctx))
                b = int(self.eval(rspec["to"], ctx))
                s = int(self.eval(rspec.get("step", {"const": 1}), ctx))
            except (TypeError, ValueError):
                return []
            if s == 0:
                return []
            return list(range(a, b, s))

        if "sort" in expr:
            sspec = expr["sort"]
            items = self.eval(sspec["list"], ctx)
            if not isinstance(items, list):
                return []
            reverse = bool(sspec.get("reverse", False))
            by = sspec.get("by")
            if by is None:
                try:
                    return sorted(items, reverse=reverse)
                except TypeError:
                    return list(items)
            keyed = [(self.eval(by, {**ctx, "$node": item, "$item": item}), item) for item in items]
            keyed.sort(key=lambda kv: kv[0], reverse=reverse)
            return [item for _, item in keyed]

        if "group" in expr:
            gspec = expr["group"]
            items = self.eval(gspec["list"], ctx)
            if not isinstance(items, list):
                return []
            by = gspec.get("by")
            buckets: dict = {}
            order: list = []
            for item in items:
                key = item if by is None else self.eval(by, {**ctx, "$node": item, "$item": item})
                if key not in buckets:
                    buckets[key] = {"key": key, "count": 0, "items": []}
                    order.append(key)
                buckets[key]["count"] += 1
                buckets[key]["items"].append(item)
            return [buckets[k] for k in order]

        if "choose" in expr:
            return self._eval_choose(expr["choose"], ctx)

        # ── Query / Aggregate ───────────────────────────────────────────

        if "query" in expr:
            qfn = ctx.get("_query_fn")
            if qfn is not None:
                return qfn(expr, ctx)
            return []

        if "count" in expr:
            arg = expr["count"]
            items = self.eval(arg, ctx)
            return len(items) if isinstance(items, (list, dict)) else 0

        # ── Function call ───────────────────────────────────────────────

        if "call" in expr:
            fn_name = expr["call"][0]
            fn = self._functions.get(fn_name)
            if fn is None:
                raise ValueError(f"Unknown function: {fn_name}")
            args_list = [self.eval(arg, ctx) for arg in expr["call"][1:]]
            if not isinstance(fn, dict):
                return fn(*args_list)
            return self._alias_call(fn_name, args_list, ctx)

        raise ValueError(f"Unknown expression type: {list(expr.keys())}")

    def _eval_choose(self, cspec: dict, ctx: dict) -> Any:
        """Interpreter for ``choose`` (see :meth:`_compile_choose`)."""
        items = self.eval(cspec["items"], ctx)
        if not isinstance(items, list):
            return None if "then" in cspec else False
        try:
            items = sorted(items)
        except TypeError:
            pass
        uniq: list = []
        for item in items:
            if item not in uniq:
                uniq.append(item)
        try:
            k = int(self.eval(cspec["k"], ctx))
        except (TypeError, ValueError):
            return None if "then" in cspec else False
        k = max(0, min(k, len(uniq)))

        as_var = cspec.get("as", "$c")
        prefix = cspec.get("prefix")
        where = cspec.get("where")
        then = cspec.get("then")
        maximize = cspec.get("agg", "max") != "min"
        best: Any = None
        found = False
        combo: list = []

        def _rec(i: int):
            nonlocal best, found
            if len(combo) == k:
                sub = {**ctx, as_var: list(combo), "$node": list(combo)}
                if where is not None and not self.eval(where, sub):
                    return
                found = True
                if then is not None:
                    value = self.eval(then, sub)
                    if best is None or (value > best if maximize else value < best):
                        best = value
                return
            if i >= len(uniq):
                return
            combo.append(uniq[i])
            if len(combo) == k or prefix is None or self.eval(prefix, {**ctx, as_var: list(combo)}):
                _rec(i + 1)
            combo.pop()
            _rec(i + 1)

        _rec(0)
        if then is not None:
            return best
        return found

    def _resolve_path(self, ctx: dict, path: str | list) -> Any:
        """Resolve a dotted path like ``$env.turn`` against context."""
        if isinstance(path, list):
            # ["$env.turn", "currentPlayerId"] or ["$node", "props.occupant"]
            raw = path[0]
            if isinstance(raw, dict):
                start = self.eval(raw, ctx)
            else:
                start = ctx.get(raw)
                if start is None:
                    clean = raw.lstrip("$").lstrip(".")
                    p_parts = clean.split(".")
                    start = ctx.get(p_parts[0])
                    if start is None:
                        start = ctx.get(f"${p_parts[0]}")
                    if start is not None:
                        for p_part in p_parts[1:]:
                            if isinstance(start, dict):
                                start = start.get(p_part)
                            else:
                                start = None
                                break
            if start is None:
                return None
            parts = path[1:] if isinstance(path[1:], list) else [path[1]]
        else:
            clean = path.lstrip("$").lstrip(".")
            parts = clean.split(".")
            start = ctx.get(parts[0])
            if start is None:
                start = ctx.get(f"${parts[0]}")
            if start is None:
                return None
            parts = parts[1:]

        obj = start
        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part)
            elif isinstance(obj, (list, tuple)) and isinstance(part, str) and part.lstrip("-").isdigit():
                obj = obj[int(part)]
            else:
                return None
        return obj

    def _eval_arithmetic(self, raw: str, ctx: dict) -> int | float | None:
        """Evaluate a compact arithmetic string like ``y * board_size + x``.

        Only supports: +, -, *, /, %, parentheses, variable substitution.
        No function calls, no builtins access.
        """

        # Step 1: Replace $prefixed.variables (e.g. $cell.x → resolved value)
        def _replace_prefixed(m):
            name = m.group(1)
            val = self._resolve_path(ctx, f"${name}")
            return str(val) if isinstance(val, (int, float)) else "0"

        s = _ARITH_PREFIX_RE.sub(_replace_prefixed, raw)

        # Step 2: Replace bare variable names (board_size, win_length, etc.)
        def _replace_bare(m):
            name = m.group(1)
            if name.replace(".", "", 1).lstrip("-").isdigit():
                return name
            val = self._resolve_path(ctx, name)
            if isinstance(val, (int, float)):
                return str(val)
            val = ctx.get(name)
            if isinstance(val, (int, float)):
                return str(val)
            return "0"

        s = _ARITH_BARE_RE.sub(_replace_bare, s)

        # Step 3: Evaluate safely
        try:
            return int(eval(s, {"__builtins__": {}}, {}))
        except Exception:
            try:
                return float(eval(s, {"__builtins__": {}}, {}))
            except Exception:
                return None


def _concat_values(items: list) -> Any:
    """Join evaluated concat items: all lists → flattened list; else string.

    ``None`` items are dropped in both modes — partial combinations
    (e.g. ``at`` of a not-yet-full array) must not poison the result.
    """
    live = [item for item in items if item is not None]
    if live and all(isinstance(item, list) for item in live):
        out = []
        for item in live:
            out.extend(item)
        return out
    return "".join(str(item) for item in live)


def _collect_call_names(expr: Any, out: set[str] | None = None) -> set[str]:
    """Collect every alias name invoked by a ``call`` inside ``expr``."""
    if out is None:
        out = set()
    if isinstance(expr, dict):
        if "call" in expr and isinstance(expr["call"], list) and expr["call"]:
            out.add(expr["call"][0])
        for value in expr.values():
            _collect_call_names(value, out)
    elif isinstance(expr, list):
        for value in expr:
            _collect_call_names(value, out)
    return out
