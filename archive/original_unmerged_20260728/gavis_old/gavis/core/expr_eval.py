"""Expression evaluator for v4.1 JSON AST expressions.

Evaluates the declarative expression language used in game rules.
Supports the subset of Expr needed for stochastic gomoku.
"""

from __future__ import annotations
import re
from typing import Any, Callable, Optional


class ExprEvaluator:
    """Recursively evaluates JSON AST expressions against a runtime context."""

    def __init__(self, functions: Optional[dict] = None):
        self._functions = functions or {}

    def register_function(self, name: str, fn: Callable):
        self._functions[name] = fn

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def eval(self, expr: Any, context: dict) -> Any:
        """Evaluate an expression AST node within the given context."""
        if not isinstance(expr, dict):
            return expr  # literal

        if 'const' in expr:
            return expr['const']

        if 'var' in expr:
            return self._resolve_path(context, expr['var'])

        if 'get' in expr:
            target, path = expr['get']
            obj = self.eval(target, context) if isinstance(target, dict) else self._resolve_path(context, str(target))
            if obj is None:
                return None
            # Resolve dotted path on the object (e.g. "props.color" on a node dict)
            if isinstance(obj, dict):
                parts = path.split('.')
                val = obj
                for p in parts:
                    if isinstance(val, dict):
                        val = val.get(p)
                    else:
                        return None
                return val
            return None

        if 'eq' in expr:
            a, b = expr['eq']
            return self.eval(a, context) == self.eval(b, context)

        if 'neq' in expr:
            a, b = expr['neq']
            return self.eval(a, context) != self.eval(b, context)

        if 'gt' in expr:
            a, b = expr['gt']
            return self.eval(a, context) > self.eval(b, context)

        if 'lt' in expr:
            a, b = expr['lt']
            return self.eval(a, context) < self.eval(b, context)

        if 'gte' in expr:
            a, b = expr['gte']
            return self.eval(a, context) >= self.eval(b, context)

        if 'lte' in expr:
            a, b = expr['lte']
            return self.eval(a, context) <= self.eval(b, context)

        if 'and' in expr:
            return all(self.eval(e, context) for e in expr['and'])

        if 'or' in expr:
            return any(self.eval(e, context) for e in expr['or'])

        if 'not' in expr:
            return not self.eval(expr['not'], context)

        if 'count' in expr:
            result = self.eval(expr['count'], context)
            if isinstance(result, list):
                return len(result)
            if isinstance(result, dict):
                return len(result)
            return 0

        if 'add' in expr:
            return sum(self.eval(e, context) for e in expr['add'])

        if 'sub' in expr:
            a, b = expr['sub']
            return self.eval(a, context) - self.eval(b, context)

        if 'mul' in expr:
            result = 1
            for e in expr['mul']:
                result *= self.eval(e, context)
            return result

        if 'div' in expr:
            a, b = expr['div']
            return self.eval(a, context) / self.eval(b, context)

        if 'mod' in expr:
            a, b = expr['mod']
            return self.eval(a, context) % self.eval(b, context)

        if 'if' in expr:
            cond = self.eval(expr['if']['cond'], context)
            if cond:
                return self.eval(expr['if']['then'], context)
            return self.eval(expr['if']['else'], context)

        if 'template' in expr:
            return self._eval_template(expr['template'], context)

        if 'query' in expr:
            return self._eval_query(expr['query'], context)

        if 'call' in expr:
            fn_name = expr['call'][0]
            fn_args = [self.eval(a, context) for a in expr['call'][1:]]
            if fn_name not in self._functions:
                raise NameError(f"Unknown function: {fn_name}")
            return self._functions[fn_name](*fn_args)

        if 'ref' in expr:
            # References are resolved by the engine (queries/predicates)
            # For now, return the expression to be evaluated later
            return expr

        if 'list' in expr:
            return [self.eval(e, context) for e in expr['list']]

        if 'object' in expr:
            return {k: self.eval(v, context) for k, v in expr['object'].items()}

        # filter, map, any, all on lists
        if 'filter' in expr:
            return self._eval_filter(expr['filter'], context)

        if 'map' in expr:
            return self._eval_map(expr['map'], context)

        if 'any' in expr:
            return self._eval_any(expr['any'], context)

        if 'all' in expr:
            return self._eval_all(expr['all'], context)

        # Unknown expression type — return as-is
        return expr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, context: dict, path: str) -> Any:
        """Resolve a dotted path like '$env.turn.currentPlayerId'."""
        # Handle $var shorthand
        path = path.lstrip('$')
        parts = path.split('.')
        value = context
        for part in parts:
            if isinstance(value, dict):
                if part not in value:
                    return None
                value = value[part]
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None
        return value

    def _eval_template(self, template: str, context: dict) -> str:
        """Interpolate {path} placeholders in a template string.

        Example: "place:{cell.props.x},{cell.props.y}" → "place:5,7"
        """
        def replacer(match):
            inner = match.group(1).strip()
            result = self._resolve_path(context, inner)
            return str(result) if result is not None else 'null'
        return re.sub(r'\{([^}]+)\}', replacer, template)

    def _eval_query(self, query_expr: dict, context: dict) -> list:
        """Evaluate a node query.  Resolved by the engine through context."""
        # The engine pre-resolves queries and puts results in context['_query_cache']
        # or we delegate to context['_query_fn']
        query_fn = context.get('_query_fn')
        if query_fn:
            return query_fn(query_expr, context)
        return []

    def _eval_filter(self, filter_expr: dict, context: dict) -> list:
        lst = self.eval(filter_expr['list'], context)
        as_var = filter_expr['as']
        where = filter_expr['where']
        result = []
        for item in lst:
            ctx = {**context, as_var: item, '$node': item}
            if self.eval(where, ctx):
                result.append(item)
        return result

    def _eval_map(self, map_expr: dict, context: dict) -> list:
        lst = self.eval(map_expr['list'], context)
        as_var = map_expr['as']
        body = map_expr['expr']
        result = []
        for item in lst:
            ctx = {**context, as_var: item, '$node': item}
            result.append(self.eval(body, ctx))
        return result

    def _eval_any(self, any_expr: dict, context: dict) -> bool:
        lst = self.eval(any_expr['list'], context)
        as_var = any_expr['as']
        where = any_expr['where']
        for item in lst:
            ctx = {**context, as_var: item, '$node': item}
            if self.eval(where, ctx):
                return True
        return False

    def _eval_all(self, all_expr: dict, context: dict) -> bool:
        lst = self.eval(all_expr['list'], context)
        as_var = all_expr['as']
        where = all_expr['where']
        for item in lst:
            ctx = {**context, as_var: item, '$node': item}
            if not self.eval(where, ctx):
                return False
        return True
