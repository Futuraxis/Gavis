"""Game engine — ties state, rules, and expression evaluation together.

Implements the ``SolverAdapter`` Protocol: all solvers in Layer 3 interact
with the game exclusively through this engine.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Optional

from .state_graph import (
    ActionInstance,
    ChanceOutcome,
    clone_state,
    create_gomoku_state,
    check_five_in_row,
    cell_index,
    cell_xy,
)
from .expr_eval import ExprEvaluator


class GameEngine:
    """A game engine that interprets a v4.1 rules JSON.

    This is the canonical implementation of the ``SolverAdapter`` Protocol:
    all Layer 3 solvers consume the game through this class.
    """

    def __init__(self, rules: dict, seed: Optional[int] = None):
        self.rules = rules
        self.rng = random.Random(seed)
        self.expr = ExprEvaluator()
        self._register_functions()

        self._actions_by_id: dict[str, dict] = {
            a['id']: a for a in rules.get('actions', [])
        }
        self._effects_by_name: dict[str, dict] = {
            name: eff for name, eff in rules.get('effects', {}).items()
        }
        self._chance_templates: list[dict] = rules.get('chance', [])
        self._triggers: list[dict] = rules.get('triggers', [])
        self._phases: dict[str, dict] = {
            p['id']: p for p in rules.get('phases', [])
        }
        self._constants: dict = rules.get('constants', {})

    # ── SolverAdapter API ─────────────────────────────────────────

    def create_initial_state(self) -> dict:
        bs = self._constants.get('board_size', 9)
        state = create_gomoku_state(bs)
        state['_win_length'] = self._constants.get('win_length', 5)
        return state

    def get_node_type(self, state: dict) -> str:
        if self.is_terminal(state):
            return 'terminal'
        phase = state['env']['phase']
        for ct in self._chance_templates:
            if phase in ct.get('phases', []):
                return 'chance'
        phase_def = self._phases.get(phase, {})
        if phase_def.get('actions'):
            return 'player'
        return 'terminal'

    def get_current_player(self, state: dict) -> Optional[str]:
        if self.get_node_type(state) != 'player':
            return None
        return state['env']['turn']['currentPlayerId']

    def get_legal_actions(self, state: dict) -> list[ActionInstance]:
        if self.get_node_type(state) != 'player':
            return []
        phase = state['env']['phase']
        actions = []
        for tmpl in self.rules.get('actions', []):
            if phase not in tmpl.get('phases', []):
                continue
            actions.extend(self._expand_template(tmpl, state))
        return actions

    def apply_action(self, state: dict, action: ActionInstance) -> dict:
        new_state = clone_state(state)
        tmpl = self._actions_by_id.get(action.template_id)
        if tmpl is None:
            raise ValueError(f"Unknown action template: {action.template_id}")
        ctx = self._build_context(new_state, action)
        self._execute_effect(tmpl['effectRef'], ctx, new_state)
        self._run_triggers(new_state)
        return new_state

    def get_chance_outcomes(self, state: dict) -> list[ChanceOutcome]:
        phase = state['env']['phase']
        for ct in self._chance_templates:
            if phase in ct.get('phases', []):
                return self._expand_chance(ct, state)
        return []

    def apply_chance(self, state: dict, outcome: ChanceOutcome) -> dict:
        new_state = clone_state(state)
        ctx = self._build_context(new_state)
        ctx['outcome'] = outcome.key
        # Inject last-placed-cell node so vanish/keep effects can reference $cell
        last_cell_id = new_state['env'].get('lastPlacedCell')
        if last_cell_id:
            if last_cell_id in new_state['nodes']:
                cell_node = new_state['nodes'][last_cell_id]
            else:
                # Rebuild node from cell_id since clone_state clears nodes
                try:
                    parts = last_cell_id.split('_')
                    x, y = int(parts[-2]), int(parts[-1])
                    bs = new_state['board_size']
                    idx = y * bs + x
                    cell_node = {
                        'id': last_cell_id,
                        'type': 'board_cell',
                        'props': {
                            'x': x, 'y': y,
                            'occupant': new_state['_board'][idx],
                            'idx': idx,
                        },
                    }
                except (ValueError, IndexError):
                    cell_node = {'id': last_cell_id, 'type': 'board_cell', 'props': {}}
            ctx['cell'] = cell_node
            ctx['$cell'] = cell_node
        self._execute_effect(outcome.effect_ref, ctx, new_state)
        # Post-chance: switch turn unless game over
        if new_state['env']['phase'] != 'game_over':
            self._execute_effect('switch_turn', ctx, new_state)
        self._run_triggers(new_state)
        return new_state

    def sample_chance(self, state: dict) -> tuple[ChanceOutcome, dict]:
        """Convenience: sample one chance outcome and apply it."""
        outcomes = self.get_chance_outcomes(state)
        r = self.rng.random()
        cumsum = 0.0
        chosen = outcomes[-1]
        for o in outcomes:
            cumsum += o.probability
            if r < cumsum:
                chosen = o
                break
        return chosen, self.apply_chance(state, chosen)

    def is_terminal(self, state: dict) -> bool:
        ctx = self._build_context(state)
        for rule in self.rules.get('terminal', []):
            if self.expr.eval(rule['condition'], ctx):
                return True
        return False

    def get_observation(self, state: dict, player_id: str) -> dict:
        """Return the player's observation.

        For perfect-information games this is the full board state.
        """
        return {
            'board': list(state['_board']),
            'board_size': state['board_size'],
            'current_player': state['env']['turn']['currentPlayerId'],
            'phase': state['env']['phase'],
        }

    def get_info_set_key(self, state: dict, player_id: str) -> str:
        """Return a canonical info-set key for CFR.

        For perfect-information games this is just a board hash.
        """
        obs = self.get_observation(state, player_id)
        board_tuple = tuple(obs['board'])
        return f"{board_tuple}|{player_id}|{obs['phase']}"

    def get_utility(self, state: dict, player_id: str) -> float:
        ctx = self._build_context(state)
        for rule in self.rules.get('utility', []):
            rule_player = rule['player']
            if isinstance(rule_player, str):
                pid = rule_player
            else:
                pid = self.expr.eval(rule_player, ctx)
            if pid != player_id:
                continue
            when = rule.get('when')
            if when is not None and not self.expr.eval(when, ctx):
                continue
            return float(self.expr.eval(rule['value'], ctx))
        return 0.0

    def load_state(self, state: dict) -> dict:
        """Import an externally constructed state (e.g. from VLM).

        Validates that the state has the required structure and fills in
        any missing fields.
        """
        if '_board' not in state:
            raise ValueError("load_state: missing '_board'")
        if 'env' not in state:
            raise ValueError("load_state: missing 'env'")
        bs = state.get('board_size', self._constants.get('board_size', 9))
        if len(state['_board']) != bs * bs:
            raise ValueError(
                f"load_state: _board length {len(state['_board'])} "
                f"does not match board_size {bs}"
            )
        # Rebuild nodes dict to match _board
        nodes = {}
        for idx, occupant in enumerate(state['_board']):
            x, y = idx % bs, idx // bs
            nodes[f'cell_{x}_{y}'] = {
                'id': f'cell_{x}_{y}',
                'type': 'board_cell',
                'props': {'x': x, 'y': y, 'occupant': occupant, 'idx': idx},
            }
        state['nodes'] = nodes
        state['_win_length'] = state.get('_win_length', self._constants.get('win_length', 5))
        return state

    # ── Action / Chance expansion ─────────────────────────────────

    def _expand_template(self, tmpl: dict, state: dict) -> list[ActionInstance]:
        ctx = self._build_context(state)
        actor_expr = tmpl.get('actor', {'var': '$env.turn.currentPlayerId'})
        actor_id = self.expr.eval(actor_expr, ctx)

        param_domains = {}
        for pname, pdef in tmpl.get('params', {}).items():
            domain = self._eval_param_domain(pdef, state, ctx)
            if pdef.get('filter'):
                filtered = []
                for item in domain:
                    item_ctx = {**ctx, f'${pname}': item, '$cell': item, '$node': item}
                    item_ctx[pname] = item
                    if self.expr.eval(pdef['filter'], item_ctx):
                        filtered.append(item)
                domain = filtered
            param_domains[pname] = domain

        combinations = _cartesian_product(param_domains)
        actions = []

        for combo in combinations:
            action_ctx = {**ctx}
            for pname, pval in combo.items():
                action_ctx[pname] = pval
                action_ctx[f'${pname}'] = pval
                if pname == 'cell':
                    action_ctx['cell'] = pval
                    action_ctx['$cell'] = pval

            legal_expr = tmpl.get('legal', {'const': True})
            if not self.expr.eval(legal_expr, action_ctx):
                continue

            ck_expr = tmpl.get('canonicalKey', {'template': f"{tmpl['id']}"})
            canonical_key = self.expr.eval(ck_expr, action_ctx)

            actions.append(ActionInstance(
                template_id=tmpl['id'],
                type=tmpl.get('type', 'action'),
                actor_id=actor_id,
                params={k: v for k, v in combo.items()},
                canonical_key=canonical_key,
            ))

        return actions

    def _eval_param_domain(self, pdef: dict, state: dict, ctx: dict) -> list:
        domain = pdef.get('domain', [])
        if isinstance(domain, list):
            return list(domain)
        if isinstance(domain, dict):
            if 'ref' in domain:
                ref_name = domain['ref']
                queries = self.rules.get('queries', {})
                if ref_name in queries:
                    domain = queries[ref_name].get('expr', domain)
                else:
                    return []
            return self._eval_query_domain(domain, state, ctx)
        return []

    def _eval_query_domain(self, query_expr: dict, state: dict, ctx: dict) -> list:
        query = query_expr.get('query', query_expr)
        node_type = query.get('type')

        if node_type == 'board_cell':
            board = state['_board']
            bs = state['board_size']
            nodes = state['nodes']
            results = []
            for idx in range(bs * bs):
                if board[idx] is None:
                    x, y = idx % bs, idx // bs
                    cell_id = f'cell_{x}_{y}'
                    node = nodes.get(cell_id)
                    if node is None:
                        node = {
                            'id': cell_id,
                            'type': 'board_cell',
                            'props': {'x': x, 'y': y, 'occupant': None, 'idx': idx},
                        }
                    results.append(node)
            return results
        return []

    def _expand_chance(self, ct: dict, state: dict) -> list[ChanceOutcome]:
        ctx = self._build_context(state)
        prob_expr = ct['probability']
        effect_map = ct.get('effectMap', {})

        if 'explicit' in prob_expr:
            outcomes = []
            for entry in prob_expr['explicit']:
                val = self.expr.eval(entry['value'], ctx)
                prob = float(self.expr.eval(entry['probability'], ctx))
                ck = self.expr.eval(
                    ct.get('canonicalKey', {'template': f"chance:{val}"}),
                    {**ctx, 'outcome': val}
                )
                outcomes.append(ChanceOutcome(
                    key=str(val),
                    probability=prob,
                    effect_ref=effect_map.get(val, f"do_{val}"),
                    canonical_key=ck,
                ))
            return outcomes
        return []

    # ── Effect execution ──────────────────────────────────────────

    def _execute_effect(self, effect_name: str, ctx: dict, state: dict):
        if effect_name not in self._effects_by_name:
            return
        for op in self._effects_by_name[effect_name].get('ops', []):
            self._execute_op(op, ctx, state)

    def _execute_op(self, op: dict, ctx: dict, state: dict):
        op_type = op.get('op')

        if op_type == 'set':
            path = self._resolve_path_template(op['path'], ctx)
            value = self.expr.eval(op['value'], ctx)
            self._set_state_path(state, path, value)

        elif op_type == 'inc':
            path = self._resolve_path_template(op['path'], ctx)
            by = self.expr.eval(op['by'], ctx)
            current = self._get_state_path(state, path) or 0
            self._set_state_path(state, path, current + by)

        elif op_type == 'listAppend':
            path = self._resolve_path_template(op['path'], ctx)
            value = self.expr.eval(op['value'], ctx)
            lst: list = self._get_state_path(state, path) or []
            if not isinstance(lst, list):
                lst = [lst]
            lst.append(value)
            self._set_state_path(state, path, lst)

        elif op_type == 'listShift':
            path = self._resolve_path_template(op['path'], ctx)
            lst: list = self._get_state_path(state, path) or []
            if lst and isinstance(lst, list):
                shifted = lst.pop(0)
                if 'dest' in op:
                    dest_path = self._resolve_path_template(op['dest'], ctx)
                    self._set_state_path(state, dest_path, shifted)
                self._set_state_path(state, path, lst)

        elif op_type == 'trimQueue':
            """Trim a FIFO queue, evicting oldest entries beyond max length."""
            path = self._resolve_path_template(op['path'], ctx)
            max_len = int(self.expr.eval(op['max'], ctx)) if 'max' in op else 3
            queue: list = self._get_state_path(state, path) or []
            if not isinstance(queue, list):
                return
            while len(queue) > max_len:
                oldest = queue.pop(0)
                cell_id = None
                if isinstance(oldest, dict):
                    cell_id = oldest.get('cellId')
                elif isinstance(oldest, str):
                    cell_id = oldest
                if cell_id:
                    self._set_state_path(state, f"state.nodes.{cell_id}.props.occupant", None)
            self._set_state_path(state, path, queue)

        elif op_type == 'emit':
            event_name = op['event']
            payload = {k: self.expr.eval(v, ctx) for k, v in op.get('payload', {}).items()}
            state.setdefault('_pending_events', []).append((event_name, payload))

        elif op_type == 'callEffect':
            effect_ref = op['effectRef']
            sub_ctx = {**ctx}
            for k, v in op.get('args', {}).items():
                sub_ctx[k] = self.expr.eval(v, ctx)
            self._execute_effect(effect_ref, sub_ctx, state)

        elif op_type == 'branch':
            cond = self.expr.eval(op['if'], ctx)
            branch_ops = op.get('then') if cond else op.get('else', [])
            for sub_op in branch_ops:
                self._execute_op(sub_op, ctx, state)

        elif op_type == 'forEach':
            lst = self.expr.eval(op['list'], ctx)
            as_var = op['as']
            for item in lst:
                sub_ctx = {**ctx, as_var: item}
                for sub_op in op['do']:
                    self._execute_op(sub_op, sub_ctx, state)

        elif op_type == 'enqueueEffect':
            args = {k: self.expr.eval(v, ctx) for k, v in op.get('args', {}).items()}
            state.setdefault('_pending_effects', []).append((op['effectRef'], args))

    def _run_triggers(self, state: dict):
        pending = state.pop('_pending_events', [])
        pending_effects = state.pop('_pending_effects', [])

        for event_name, payload in pending:
            for trigger in self._triggers:
                if trigger['event'] != event_name:
                    continue
                cond = trigger.get('condition')
                if cond is not None:
                    t_ctx = self._build_context(state)
                    t_ctx['$event'] = payload
                    if not self.expr.eval(cond, t_ctx):
                        continue
                t_ctx = self._build_context(state)
                t_ctx['$event'] = payload
                t_ctx.update(payload)
                self._execute_effect(trigger['effectRef'], t_ctx, state)

        for effect_ref, args in pending_effects:
            e_ctx = self._build_context(state)
            e_ctx.update(args)
            self._execute_effect(effect_ref, e_ctx, state)

    # ── Context building ──────────────────────────────────────────

    def _build_context(self, state: dict, action: Optional[ActionInstance] = None) -> dict:
        ctx = {
            '$state': state,
            '$env': state['env'],
            'state': state,
            'env': state['env'],
            '_query_fn': lambda q, c: self._eval_query_domain(q, state, c),
        }

        if action is not None:
            pid = action.actor_id
            actor_node = {
                'id': pid,
                'props': {'color': 'black' if pid == 'p_black' else 'white'},
            }
            ctx['$actor'] = actor_node
            ctx['actor'] = actor_node
            ctx['$action'] = action
            ctx['action'] = action
            ctx['$params'] = action.params
            ctx['params'] = action.params
            for k, v in action.params.items():
                ctx[k] = v
                ctx[f'${k}'] = v
        else:
            pid = state['env']['turn'].get('currentPlayerId', 'p_black')
            actor_node = {
                'id': pid,
                'props': {'color': 'black' if pid == 'p_black' else 'white'},
            }
            ctx['$actor'] = actor_node
            ctx['actor'] = actor_node

        return ctx

    # ── State path access ─────────────────────────────────────────

    def _resolve_path_template(self, path: str, ctx: dict) -> str:
        def replacer(m):
            inner = m.group(1).strip()
            val = self.expr._resolve_path(ctx, inner)
            return str(val) if val is not None else 'null'
        return re.sub(r'\{([^}]+)\}', replacer, path)

    def _get_state_path(self, state: dict, path: str) -> any:
        parts = path.replace('state.', '').split('.')
        obj = state
        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return None
        return obj

    def _set_state_path(self, state: dict, path: str, value: any):
        clean = path.replace('state.', '')
        parts = clean.split('.')

        # Intercept nodes.*.props.occupant → sync _board (fast path)
        if (len(parts) == 4 and parts[0] == 'nodes'
                and parts[2] == 'props' and parts[3] == 'occupant'):
            cell_id = parts[1]
            try:
                segs = cell_id.split('_')
                x, y = int(segs[-2]), int(segs[-1])
                idx = y * state['board_size'] + x
            except (IndexError, ValueError):
                return
            state['_board'][idx] = value
            node = state['nodes'].get(cell_id)
            if node is not None:
                node['props']['occupant'] = value
            return

        # Generic dict path
        obj = state
        for part in parts[:-1]:
            if isinstance(obj, dict):
                if part not in obj:
                    obj[part] = {}
                obj = obj[part]
        if isinstance(obj, dict):
            obj[parts[-1]] = value

    # ── Functions ─────────────────────────────────────────────────

    def _register_functions(self):
        self.expr.register_function(
            'check_five_in_row',
            lambda s, cell_id: check_five_in_row(s, str(cell_id))
        )
        self.expr.register_function('debug_print', lambda msg: print(f"[debug] {msg}"))

    @classmethod
    def from_json(cls, path: str | Path, **kwargs) -> 'GameEngine':
        with open(path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        return cls(rules, **kwargs)


# ── Cartesian product helper ──────────────────────────────────────

def _cartesian_product(param_domains: dict[str, list]) -> list[dict]:
    if not param_domains:
        return [{}]
    keys = list(param_domains.keys())
    values = list(param_domains.values())
    results = []

    def _recurse(idx, current):
        if idx == len(keys):
            results.append(dict(current))
            return
        for val in values[idx]:
            current[keys[idx]] = val
            _recurse(idx + 1, current)

    _recurse(0, {})
    return results
