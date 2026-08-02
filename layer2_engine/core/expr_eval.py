"""Expression evaluator for the v5.0 rules engine.

Evaluates a JSON-based expression tree against a context dict.
Expanded from v4.1 to support switch/case, filter/map/any/all
collection operations, and simple arithmetic expressions.

Core expression types:
  const, var, get, eq, and, or, not, gt, lt
  if, switch, call, query, count, template, concat
  filter, any, all, map
  expr — compact arithmetic string "y * board_size + x"
"""

from __future__ import annotations

import re
from typing import Any

# Precompiled regexes (hot path: called millions of times per MCTS move).
_ARITH_PREFIX_RE = re.compile(r'\$([a-zA-Z_][a-zA-Z0-9_.]*)')
_ARITH_BARE_RE = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_.]*)\b')
_TEMPLATE_RE = re.compile(r'\{([^}]+)\}')


class ExprEvaluator:
    """Evaluates expression trees against a context dictionary."""

    def __init__(self):
        self._functions: dict[str, callable] = {}

    def register_function(self, name: str, fn: callable):
        """Register a function that can be called from expressions."""
        self._functions[name] = fn

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
        if 'const' in spec:
            value = spec['const']
            return lambda ctx: value

        if 'var' in spec:
            return self._compile_path(spec['var'])

        if 'get' in spec:
            path = spec['get']
            return self._compile_path(path)

        # ── String ops ──────────────────────────────────────────────────
        if 'template' in spec:
            tmpl = spec['template']
            parts: list = []
            last = 0
            for m in _TEMPLATE_RE.finditer(tmpl):
                if m.start() > last:
                    parts.append(tmpl[last:m.start()])
                parts.append(self._compile_path(m.group(1).strip()))
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

        if 'concat' in spec:
            children = [self.compile(item) for item in spec['concat']]

            def _concat(ctx: dict, children: list = children) -> str:
                return ''.join(str(fn(ctx)) for fn in children)

            return _concat

        # ── Arithmetic shorthand ────────────────────────────────────────
        if 'expr' in spec:
            def _arith(ctx: dict, spec: dict = spec) -> Any:
                return self._eval_arithmetic(spec['expr'], ctx)

            return _arith

        # ── Conditionals ────────────────────────────────────────────────
        if 'if' in spec:
            if_body = spec['if']
            if isinstance(if_body, dict) and 'cond' in if_body:
                cond_fn = self.compile(if_body['cond'])
                then_fn = self.compile(if_body.get('then', True))
                else_fn = self.compile(if_body.get('else', None))
            else:
                cond_fn = self.compile(if_body)
                then_fn = self.compile(spec.get('then', True))
                else_fn = self.compile(spec.get('else', None))

            def _if(ctx: dict, cond_fn=cond_fn, then_fn=then_fn, else_fn=else_fn) -> Any:
                return then_fn(ctx) if cond_fn(ctx) else else_fn(ctx)

            return _if

        if 'switch' in spec:
            cases = []
            default_fn = None
            for case in spec['switch']:
                if 'case' in case:
                    cases.append((case['case'], self.compile(case['then'])))
                else:
                    default_fn = self.compile(case.get('then'))
            input_fn = self.compile(spec.get('input', {'var': '$input'}))

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

        # ── Logical / comparison ops ────────────────────────────────────
        if 'eq' in spec or 'neq' in spec or 'gt' in spec or 'gte' in spec or 'lt' in spec or 'lte' in spec:
            op = next(k for k in ('eq', 'neq', 'gt', 'gte', 'lt', 'lte') if k in spec)
            left_fn = self.compile(spec[op][0])
            right_fn = self.compile(spec[op][1])
            cmp_map = {
                'eq': lambda a, b: a == b,
                'neq': lambda a, b: a != b,
                'gt': lambda a, b: a > b,
                'gte': lambda a, b: a >= b,
                'lt': lambda a, b: a < b,
                'lte': lambda a, b: a <= b,
            }
            cmp_fn = cmp_map[op]

            def _cmp(ctx: dict, left_fn=left_fn, right_fn=right_fn, cmp_fn=cmp_fn) -> bool:
                return cmp_fn(left_fn(ctx), right_fn(ctx))

            return _cmp

        if 'and' in spec:
            children = [self.compile(sub) for sub in spec['and']]

            def _and(ctx: dict, children: list = children) -> bool:
                return all(fn(ctx) for fn in children)

            return _and

        if 'or' in spec:
            children = [self.compile(sub) for sub in spec['or']]

            def _or(ctx: dict, children: list = children) -> bool:
                return any(fn(ctx) for fn in children)

            return _or

        if 'not' in spec:
            inner_fn = self.compile(spec['not'])

            def _not(ctx: dict, inner_fn=inner_fn) -> bool:
                return not inner_fn(ctx)

            return _not

        # ── Fallback: interpreter for exotic types ──────────────────────
        def _fallback(ctx: dict, spec: dict = spec) -> Any:
            return self.eval(spec, ctx)

        return _fallback

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
                    elif isinstance(obj, (list, tuple)) and p.lstrip('-').isdigit():
                        idx = int(p)
                        obj = obj[idx] if -len(obj) <= idx < len(obj) else None
                    else:
                        return None
                return obj

            return _get
        parts = path.lstrip('$').lstrip('.').split('.')

        def _path(ctx: dict, parts: list = parts) -> Any:
            obj = ctx
            for p in parts:
                if isinstance(obj, dict):
                    if p in obj:
                        obj = obj[p]
                    elif f'${p}' in obj:
                        obj = obj[f'${p}']
                    else:
                        return None
                elif isinstance(obj, (list, tuple)) and p.lstrip('-').isdigit():
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

        if 'const' in expr:
            return expr['const']

        if 'var' in expr:
            return self._resolve_path(ctx, expr['var'])

        if 'get' in expr:
            return self._resolve_path(ctx, expr['get'])

        # ── String ops ──────────────────────────────────────────────────

        if 'template' in expr:
            template = expr['template']
            parts = []
            last_end = 0
            for m in _TEMPLATE_RE.finditer(template):
                parts.append(template[last_end:m.start()])
                val = self._resolve_path(ctx, m.group(1).strip())
                parts.append(str(val) if val is not None else '')
                last_end = m.end()
            parts.append(template[last_end:])
            return ''.join(parts)

        if 'concat' in expr:
            return ''.join(str(self.eval(item, ctx)) for item in expr['concat'])

        # ── Arithmetic shorthand ────────────────────────────────────────

        if 'expr' in expr:
            return self._eval_arithmetic(expr['expr'], ctx)

        # ── Conditionals ────────────────────────────────────────────────

        if 'if' in expr:
            if_body = expr['if']
            if isinstance(if_body, dict) and 'cond' in if_body:
                cond = self.eval(if_body['cond'], ctx)
                then_val = if_body.get('then', True)
                else_val = if_body.get('else', None)
            else:
                cond = self.eval(if_body, ctx)
                then_val = expr.get('then', True)
                else_val = expr.get('else', None)
            return self.eval(then_val, ctx) if cond else (
                self.eval(else_val, ctx) if else_val is not None else None
            )

        if 'switch' in expr:
            input_val = self.eval(expr.get('input', {'var': '$input'}), ctx)
            for case in expr['switch']:
                if case.get('case') == input_val:
                    return self.eval(case['then'], ctx)
            # else/default branch
            for case in expr['switch']:
                if 'default' in case or ('case' not in case and 'then' in case):
                    return self.eval(case['then'], ctx)
            return None

        # ── Logical ops ─────────────────────────────────────────────────

        if 'eq' in expr:
            return self.eval(expr['eq'][0], ctx) == self.eval(expr['eq'][1], ctx)

        if 'neq' in expr:
            return self.eval(expr['neq'][0], ctx) != self.eval(expr['neq'][1], ctx)

        if 'gt' in expr:
            return self.eval(expr['gt'][0], ctx) > self.eval(expr['gt'][1], ctx)

        if 'gte' in expr:
            return self.eval(expr['gte'][0], ctx) >= self.eval(expr['gte'][1], ctx)

        if 'lt' in expr:
            return self.eval(expr['lt'][0], ctx) < self.eval(expr['lt'][1], ctx)

        if 'lte' in expr:
            return self.eval(expr['lte'][0], ctx) <= self.eval(expr['lte'][1], ctx)

        if 'and' in expr:
            return all(self.eval(sub, ctx) for sub in expr['and'])

        if 'or' in expr:
            return any(self.eval(sub, ctx) for sub in expr['or'])

        if 'not' in expr:
            return not self.eval(expr['not'], ctx)

        # ── Collection ops ──────────────────────────────────────────────

        if 'filter' in expr:
            items = self.eval(expr['filter']['list'], ctx)
            as_var = expr['filter'].get('as', '$node')
            where = expr['filter']['where']
            if not isinstance(items, list):
                return []
            results = []
            for item in items:
                item_ctx = {**ctx, as_var: item}
                if self.eval(where, item_ctx):
                    results.append(item)
            return results

        if 'any' in expr:
            items = self.eval(expr['any']['list'], ctx)
            as_var = expr['any'].get('as', '$node')
            where = expr['any']['where']
            if not isinstance(items, list):
                return False
            for item in items:
                item_ctx = {**ctx, as_var: item}
                if self.eval(where, item_ctx):
                    return True
            return False

        if 'all' in expr:
            items = self.eval(expr['all']['list'], ctx)
            as_var = expr['all'].get('as', '$node')
            where = expr['all']['where']
            if not isinstance(items, list):
                return True
            for item in items:
                item_ctx = {**ctx, as_var: item}
                if not self.eval(where, item_ctx):
                    return False
            return True

        if 'map' in expr:
            items = self.eval(expr['map']['list'], ctx)
            as_var = expr['map'].get('as', '$node')
            map_expr = expr['map']['expr']
            if not isinstance(items, list):
                return []
            results = []
            for item in items:
                item_ctx = {**ctx, as_var: item}
                results.append(self.eval(map_expr, item_ctx))
            return results

        # ── Query / Aggregate ───────────────────────────────────────────

        if 'query' in expr:
            qfn = ctx.get('_query_fn')
            if qfn is not None:
                return qfn(expr, ctx)
            return []

        if 'count' in expr:
            arg = expr['count']
            items = self.eval(arg, ctx)
            return len(items) if isinstance(items, (list, dict)) else 0

        # ── Function call ───────────────────────────────────────────────

        if 'call' in expr:
            fn_name = expr['call'][0]
            args_list = [self.eval(arg, ctx) for arg in expr['call'][1:]]
            fn = self._functions.get(fn_name)
            if fn is None:
                raise ValueError(f"Unknown function: {fn_name}")
            return fn(*args_list)

        raise ValueError(f"Unknown expression type: {list(expr.keys())}")

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
                    clean = raw.lstrip('$').lstrip('.')
                    p_parts = clean.split('.')
                    start = ctx.get(p_parts[0])
                    if start is None:
                        start = ctx.get(f'${p_parts[0]}')
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
            clean = path.lstrip('$').lstrip('.')
            parts = clean.split('.')
            start = ctx.get(parts[0])
            if start is None:
                start = ctx.get(f'${parts[0]}')
            if start is None:
                return None
            parts = parts[1:]

        obj = start
        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part)
            elif isinstance(obj, (list, tuple)) and isinstance(part, str) and part.lstrip('-').isdigit():
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
            val = self._resolve_path(ctx, f'${name}')
            return str(val) if isinstance(val, (int, float)) else '0'

        s = _ARITH_PREFIX_RE.sub(_replace_prefixed, raw)

        # Step 2: Replace bare variable names (board_size, win_length, etc.)
        def _replace_bare(m):
            name = m.group(1)
            if name.replace('.', '', 1).lstrip('-').isdigit():
                return name
            val = self._resolve_path(ctx, name)
            if isinstance(val, (int, float)):
                return str(val)
            val = ctx.get(name)
            if isinstance(val, (int, float)):
                return str(val)
            return '0'

        s = _ARITH_BARE_RE.sub(_replace_bare, s)

        # Step 3: Evaluate safely
        try:
            return int(eval(s, {'__builtins__': {}}, {}))
        except Exception:
            try:
                return float(eval(s, {'__builtins__': {}}, {}))
            except Exception:
                return None
