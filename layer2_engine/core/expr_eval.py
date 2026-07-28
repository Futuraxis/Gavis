"""Expression evaluator for the v4.1 rules engine.

Evaluates a JSON-based expression tree against a context dict.
Supports the operators needed by ``rules.json``: const, get, eq,
and, or, not, gt, lt, count, call, template, concat.
"""

from __future__ import annotations

from typing import Any


class ExprEvaluator:
    """Evaluates expression trees against a context dictionary."""

    def __init__(self):
        self._functions: dict[str, callable] = {}

    def register_function(self, name: str, fn: callable):
        """Register a function that can be called from expressions."""
        self._functions[name] = fn

    def eval(self, expr: Any, ctx: dict) -> Any:
        """Evaluate an expression tree against the given context."""
        if not isinstance(expr, dict):
            return expr

        expr = dict(expr)  # shallow copy so we don't mutate the original

        if 'const' in expr:
            return expr['const']

        if 'var' in expr:
            return self._resolve_path(ctx, expr['var'])

        if 'get' in expr:
            return self._resolve_path(ctx, expr['get'])

        if 'eq' in expr:
            left = self.eval(expr['eq'][0], ctx)
            right = self.eval(expr['eq'][1], ctx)
            return left == right

        if 'gt' in expr:
            return self.eval(expr['gt'][0], ctx) > self.eval(expr['gt'][1], ctx)

        if 'lt' in expr:
            return self.eval(expr['lt'][0], ctx) < self.eval(expr['lt'][1], ctx)

        if 'and' in expr:
            return all(self.eval(sub, ctx) for sub in expr['and'])

        if 'or' in expr:
            return any(self.eval(sub, ctx) for sub in expr['or'])

        if 'not' in expr:
            return not self.eval(expr['not'], ctx)

        if 'query' in expr:
            # Query expression — resolves to a list of nodes via the context
            qfn = ctx.get('_query_fn')
            if qfn is not None:
                return qfn(expr, ctx)
            return []

        if 'count' in expr:
            arg = expr['count']
            # arg may be a query dict — eval it first
            items = self.eval(arg, ctx)
            return len(items) if isinstance(items, (list, dict)) else 0

        if 'call' in expr:
            fn_name = expr['call'][0]
            args = [self.eval(arg, ctx) for arg in expr['call'][1:]]
            fn = self._functions.get(fn_name)
            if fn is None:
                raise ValueError(f"Unknown function: {fn_name}")
            return fn(*args)

        if 'template' in expr:
            template = expr['template']
            parts = []
            last_end = 0
            import re
            for m in re.finditer(r'\{([^}]+)\}', template):
                parts.append(template[last_end:m.start()])
                val = self._resolve_path(ctx, m.group(1).strip())
                parts.append(str(val) if val is not None else 'null')
                last_end = m.end()
            parts.append(template[last_end:])
            return ''.join(parts)

        if 'concat' in expr:
            return ''.join(str(self.eval(item, ctx)) for item in expr['concat'])

        if 'if' in expr:
            if_body = expr['if']
            # Support both {"if": COND, "then": T, "else": E}
            # and {"if": {"cond": COND, "then": T, "else": E}}
            if isinstance(if_body, dict) and 'cond' in if_body:
                cond = self.eval(if_body['cond'], ctx)
                then_val = if_body.get('then', True)
                else_val = if_body.get('else', None)
            else:
                cond = self.eval(if_body, ctx)
                then_val = expr.get('then', True)
                else_val = expr.get('else', None)
            return self.eval(then_val, ctx) if cond else (self.eval(else_val, ctx) if else_val is not None else None)

        raise ValueError(f"Unknown expression type: {list(expr.keys())}")

    def _resolve_path(self, ctx: dict, path: str | list) -> Any:
        """Resolve a dotted path like ``$env.turn.currentPlayerId``."""
        if isinstance(path, list):
            # ["$env.turn", "currentPlayerId"] or ["$node", "props.occupant"]
            raw = path[0]
            if isinstance(raw, (dict, list)):
                start = self.eval(raw, ctx)
            else:
                # Try direct key, then dotted path
                start = ctx.get(raw)
                if start is None:
                    # Maybe it's a dotted path like "$env.turn"
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
            parts = path[1:]
        else:
            # "$env.turn.currentPlayerId"
            clean = path.lstrip('$').lstrip('.')
            parts = clean.split('.')
            # Try both with and without $ prefix (context may have "$env" or "env")
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
