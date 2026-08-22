"""Game engine — interprets v5.0 rules JSON as a stochastic game model.

Implements the ``SolverAdapter`` Protocol: all solvers in Layer 3 interact
with the game exclusively through this engine.

Two-layer state architecture:
  - Ground state: compact arrays + env scalars
  - Derived views: computed on-the-fly via derivation rules (grid, enum, literal)
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Optional

from .expr_eval import ExprEvaluator
from .rules_compiler import RulesCompiler
from .state_graph import (
    ActionInstance,
    ChanceOutcome,
    DerivedViewEngine,
    clone_state,
    create_initial_state,
)


class GameEngine:
    """A game engine that interprets a v5.0 rules JSON.

    This is the canonical implementation of the ``SolverAdapter`` Protocol:
    all Layer 3 solvers consume the game through this class.
    """

    def __init__(self, rules: dict, seed: Optional[int] = None):
        self.rules = rules
        self.rng = random.Random(seed)
        self.expr = ExprEvaluator()
        # Compiled query filters, keyed by id() of the filter expr object
        # (rule dicts live for the engine's lifetime, so ids stay stable).
        self._compiled_filters: dict[int, callable] = {}
        # RulesCompiler artifacts; assigned at the end of __init__ so that
        # probe validation runs against the pure interpreter.
        self._compiled = None

        # Register rules functions (alias definitions, v5.1 — the
        # BUILTIN_FUNCTIONS registry is retired; rules JSON is self-sufficient).
        self.expr.set_functions(rules.get("functions", {}))

        # Parse schema
        self._constants: dict = rules.get("constants", {})
        self._players: list[dict] = self._parse_players(rules.get("players", []))
        self._schema = {
            "groundState": rules.get("groundState", {}),
            "derivedViews": rules.get("derivedViews", {}),
        }

        # Derived view engine (lazy)
        self._view_engine = DerivedViewEngine(self._schema)

        # Pre-parse effectors, actions, etc.
        self._actions: list[dict] = rules.get("actions", [])
        self._actions_by_id: dict[str, dict] = {a["id"]: a for a in self._actions}
        self._effectors: dict[str, dict] = rules.get("effectors", {})
        self._phases: dict[str, dict] = {p["id"]: p for p in rules.get("phases", [])}
        self._chance_templates: list[dict] = rules.get("chance", [])
        self._triggers: list[dict] = rules.get("triggers", [])
        self._queries: dict[str, dict] = rules.get("queries", {})
        self._terminal: list[dict] = rules.get("terminal", [])
        self._utility: list[dict] = rules.get("utility", [])
        self._visibility: dict = rules.get("visibility", {"default": "public"})

        # Compile the rules into native functions, probe-validated against
        # the interpreter.  Any artifact failing validation is disabled.
        # Validation samples chance nodes, so the rng stream is saved and
        # restored to keep engine construction side-effect free.
        try:
            artifacts = RulesCompiler().compile(rules, engine=self)
            rng_state = self.rng.getstate()
            artifacts.validate(self)
            self.rng.setstate(rng_state)
            self._compiled = artifacts
        except Exception:
            self._compiled = None

    def _expand_missing(self, template_ids: list[str], state: dict) -> list:
        """Interpreter fallback for action templates the compiler skipped.

        Phase filter mirrors ``_interp_legal_actions`` (the compiler's
        generated phase guards only cover compiled templates).
        """
        out = []
        phase = state["env"].get("phase", "")
        for tmpl in self._actions:
            if tmpl["id"] in template_ids and phase in tmpl.get("phases", []):
                out.extend(self._expand_template(tmpl, state))
        return out

    # ── Initial state ─────────────────────────────────────────────────

    def create_initial_state(self) -> dict:
        """Create initial ground state from schema."""
        return create_initial_state(self._schema, self._constants, self._players)

    # ── SolverAdapter API ─────────────────────────────────────────────

    def get_node_type(self, state: dict) -> str:
        """Return 'player', 'chance', or 'terminal'."""
        if self.is_terminal(state):
            return "terminal"
        phase = state["env"].get("phase", "")
        for ct in self._chance_templates:
            if phase in ct.get("phases", []):
                return "chance"
        phase_def = self._phases.get(phase, {})
        if phase_def.get("actions"):
            return "player"
        return "terminal"

    def get_current_player(self, state: dict) -> Optional[str]:
        """Return the current player ID, or None if not a player node."""
        if self.get_node_type(state) != "player":
            return None
        return state["env"].get("turn")

    def get_legal_actions(self, state: dict) -> list[ActionInstance]:
        """Expand action templates into concrete action instances."""
        if self._compiled is not None and self._compiled.legal_actions is not None and "_arrays" in state:
            return self._compiled.legal_actions(state)
        return self._interp_legal_actions(state)

    def _interp_legal_actions(self, state: dict) -> list[ActionInstance]:
        """Interpreter-path legal actions (used by compiler validation)."""
        if self.get_node_type(state) != "player":
            return []
        phase = state["env"].get("phase", "")
        actions = []
        for tmpl in self._actions:
            if phase not in tmpl.get("phases", []):
                continue
            actions.extend(self._expand_template(tmpl, state))
        return actions

    def apply_action(self, state: dict, action: ActionInstance) -> dict:
        """Apply an action to a clone of the state."""
        new_state = clone_state(state)
        tmpl = self._actions_by_id.get(action.template_id)
        if tmpl is None:
            raise ValueError(f"Unknown action template: {action.template_id}")
        ctx = self._build_context(new_state, action)
        self._execute_effector(tmpl["effectRef"], ctx, new_state)
        self._run_triggers(new_state)
        return new_state

    def get_chance_outcomes(self, state: dict) -> list[ChanceOutcome]:
        """Return all chance outcomes with probabilities."""
        if self._compiled is not None and self._compiled.chance_outcomes is not None and "_arrays" in state:
            return self._compiled.chance_outcomes(state)
        return self._interp_chance_outcomes(state)

    def _interp_chance_outcomes(self, state: dict) -> list[ChanceOutcome]:
        """Interpreter-path chance outcomes (used by compiler validation)."""
        phase = state["env"].get("phase", "")
        for ct in self._chance_templates:
            if phase in ct.get("phases", []):
                return self._expand_chance(ct, state)
        return []

    def apply_chance(self, state: dict, outcome: ChanceOutcome) -> dict:
        """Apply a chance outcome to a clone of the state."""
        new_state = clone_state(state)
        ctx = self._build_context(new_state)
        ctx["outcome"] = outcome.key
        self._execute_effector(outcome.effect_ref, ctx, new_state)
        self._run_triggers(new_state)
        return new_state

    def sample_chance(self, state: dict) -> tuple[ChanceOutcome, dict]:
        """Sample one chance outcome and apply it.

        Probabilities are normalized by their sum, so a non-normalized
        table samples without bias toward its last entry.  An empty (or
        all-zero) outcome table is a rules bug and raises ``ValueError``
        instead of crashing on ``outcomes[-1]``.
        """
        outcomes = self.get_chance_outcomes(state)
        if not outcomes:
            phase = state["env"].get("phase", "?")
            raise ValueError(f"no chance outcomes at phase {phase}")
        total = sum(o.probability for o in outcomes)
        if total <= 0:
            phase = state["env"].get("phase", "?")
            raise ValueError(f"non-positive total chance probability {total} at phase {phase}")
        r = self.rng.random() * total
        cumsum = 0.0
        chosen = outcomes[-1]  # float-cumsum fallback: r may land just below total
        for o in outcomes:
            cumsum += o.probability
            if r < cumsum:
                chosen = o
                break
        return chosen, self.apply_chance(state, chosen)

    def is_terminal(self, state: dict) -> bool:
        """Return True if any terminal condition is met."""
        if self._compiled is not None and self._compiled.is_terminal is not None and "_arrays" in state:
            return self._compiled.is_terminal(state)
        return self._interp_is_terminal(state)

    def _interp_is_terminal(self, state: dict) -> bool:
        """Interpreter-path terminal check (used by compiler validation)."""
        ctx = self._build_context(state)
        for rule in self._terminal:
            if self.expr.eval(rule["condition"], ctx):
                return True
        return False

    def get_observation(self, state: dict, player_id: str) -> dict:
        """Return the player's observation (projected via visibility rules)."""
        return self.project_observation(state, player_id)

    def project_observation(self, state: dict, viewer: str) -> dict:
        """Project state through visibility rules for ``viewer``.

        Materializes all derived views and applies field-level visibility.
        """
        obs = {}
        view_names = list(self._schema.get("derivedViews", {}).keys())
        visibility = self._visibility
        default_level = visibility.get("default", "public")
        rules = visibility.get("rules", [])

        for vname in view_names:
            entities = self._materialize_view(state, vname)
            if default_level == "public":
                # Perfect information: return all fields
                obs[vname] = entities
            else:
                # Partial information: filter fields per entity
                filtered = []
                for entity in entities:
                    entity_ctx = {**self._build_context(state), "$node": entity, "$viewer": viewer}
                    entity_obs = dict(entity)
                    for rule in rules:
                        if rule.get("view", "") != vname:
                            continue
                        rule_filter = rule.get("filter")
                        if rule_filter is not None:
                            if not self.expr.eval(rule_filter, entity_ctx):
                                continue
                        # Apply field visibility
                        hidden_any = False
                        for field, level in rule.get("fields", {}).items():
                            if level != "public":
                                entity_obs.pop(field, None)
                                hidden_any = True
                        # Strip the raw array entry too — enum/grid views expose
                        # it as ``value``, which would leak hidden data
                        # (e.g. an opponent's hole card id).
                        if hidden_any:
                            entity_obs.pop("value", None)
                    filtered.append(entity_obs)
                obs[vname] = filtered

        # Also include env
        obs["env"] = dict(state.get("env", {}))
        return obs

    def get_info_set_key(self, state: dict, player_id: str) -> str:
        """Return a canonical info-set key for CFR.

        Compact sha256 hash of the projected observation, serialized in
        schema order.  No ``sort_keys`` — ``project_observation`` builds
        the dict in deterministic view/env insertion order, so sorting
        only pays CPU.  The key stays 64 chars no matter how large the
        observation grows (e.g. long werewolf speech logs).

        Format note (2026-08-13): keys are hashes, NOT the full JSON —
        CFR strategy tables persisted under the old format (Hybrid's
        ``cfr_table_path`` JSON) are incompatible and must be retrained.
        """
        obs = self.project_observation(state, player_id)
        serialized = json.dumps(obs, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get_utility(self, state: dict, player_id: str) -> float:
        """Evaluate utility for ``player_id`` at a terminal state."""
        ctx = self._build_context(state)
        for rule in self._utility:
            rule_player = rule["player"]
            if isinstance(rule_player, dict):
                pid = self.expr.eval(rule_player, ctx)
            else:
                pid = rule_player
            if pid != player_id:
                continue
            when = rule.get("when")
            if when is not None and not self.expr.eval(when, ctx):
                continue
            value = rule.get("value")
            if isinstance(value, dict):
                return float(self.expr.eval(value, ctx))
            return float(value)
        return 0.0

    def load_state(self, state: dict) -> dict:
        """Import an externally constructed state (e.g. from VLM).

        Fills in any missing ground arrays/env fields from schema defaults.
        """
        schema_ground = self._schema.get("groundState", {})
        result = create_initial_state(self._schema, self._constants, self._players)

        # Merge arrays
        for arr_name, arr_def in schema_ground.items():
            if arr_def.get("type") != "array":
                continue
            ext_arr = state.get(arr_name) or state.get("_arrays", {}).get(arr_name)
            if ext_arr is not None:
                result["_arrays"][arr_name] = list(ext_arr)

        # Merge env
        ext_env = state.get("env", {})
        if ext_env:
            result["env"].update(ext_env)

        return result

    # ── Action expansion ─────────────────────────────────────────────

    def _expand_template(self, tmpl: dict, state: dict) -> list[ActionInstance]:
        ctx = self._build_context(state)
        actor_expr = tmpl.get("actor", {"var": "$env.turn"})
        actor_id = self.expr.eval(actor_expr, ctx)

        param_domains: dict[str, list] = {}
        for pname, pdef in tmpl.get("params", {}).items():
            if pdef.get("type") == "text":
                # 自由文本参数（预制能力）：不参与枚举，展开占位空串。
                # solver 在 apply_action 时把实际文本放入 ActionInstance.params，
                # effector 经 ctx['$text'] 读取（_build_context 自动平铺 params）。
                param_domains[pname] = [""]
                continue
            domain = self._resolve_param_domain(pdef, state, ctx)
            if pdef.get("filter"):
                filtered = []
                for item in domain:
                    item_ctx = {**ctx, f"${pname}": item, "$node": item}
                    if self.expr.eval(pdef["filter"], item_ctx):
                        filtered.append(item)
                domain = filtered
            param_domains[pname] = domain

        combinations = _cartesian_product(param_domains)
        actions = []

        for combo in combinations:
            action_ctx = {**ctx}
            for pname, pval in combo.items():
                action_ctx[pname] = pval
                action_ctx[f"${pname}"] = pval

            legal_expr = tmpl.get("legal", {"const": True})
            if not self.expr.eval(legal_expr, action_ctx):
                continue

            ck_expr = tmpl.get("canonicalKey", {"template": tmpl["id"]})
            canonical_key = self.expr.eval(ck_expr, action_ctx)

            actions.append(
                ActionInstance(
                    template_id=tmpl["id"],
                    type=tmpl.get("type", "action"),
                    actor_id=actor_id,
                    params={k: v for k, v in combo.items()},
                    canonical_key=canonical_key,
                )
            )

        return actions

    def _resolve_param_domain(self, pdef: dict, state: dict, ctx: dict) -> list[Any]:
        """Resolve a parameter's candidate domain.

        Supports:
          - [] — direct literal list
          - {"ref": "queryName"} — reference to a named query
          - {"array": <expr|name>} — dynamic ground array contents
          - {"expr": <expr>} — any expression; a non-list result is
            wrapped into a single-element list
        """
        domain = pdef.get("domain", [])
        if isinstance(domain, list):
            return list(domain)
        if isinstance(domain, dict):
            if "ref" in domain:
                query_name = domain["ref"]
                query_def = self._queries.get(query_name)
                if query_def is None:
                    return []
                return self._resolve_query(query_def, state, ctx)
            if "array" in domain:
                arr_spec = domain["array"]
                arr_name = self.expr.eval(arr_spec, ctx) if isinstance(arr_spec, dict) else arr_spec
                arr = state.get("_arrays", {}).get(arr_name)
                return list(arr) if isinstance(arr, list) else []
            if "expr" in domain:
                value = self.expr.eval(domain["expr"], ctx)
                if isinstance(value, list):
                    return list(value)
                return [value] if value is not None else []
        return []

    def _resolve_query(self, query_def: dict, state: dict, ctx: dict) -> list[dict]:
        """Resolve a query against a derived view."""
        view_name = query_def.get("view", "")
        filter_expr = query_def.get("filter")
        entities = self._materialize_view(state, view_name)

        if filter_expr is None:
            return entities

        filter_fn = self._get_compiled_filter(filter_expr)
        filtered = []
        for entity in entities:
            entity_ctx = {**ctx, "$node": entity, "$self": entity}
            if filter_fn(entity_ctx):
                filtered.append(entity)
        return filtered

    def _materialize_view(self, state: dict, view_name: str) -> list[dict]:
        """Materialize a derived view from current state."""
        if self._compiled is not None and self._compiled.materialize is not None and "_arrays" in state:
            result = self._compiled.materialize(state, view_name)
            if result is not None:
                return result
        return self._view_engine.materialize(state, view_name)

    # ── Chance expansion ─────────────────────────────────────────────

    def _expand_chance(self, ct: dict, state: dict) -> list[ChanceOutcome]:
        ctx = self._build_context(state)
        prob_expr = ct["probability"]
        effect_map = ct.get("effectMap", {})

        if "explicit" in prob_expr:
            outcomes = []
            for entry in prob_expr["explicit"]:
                outcome_val = entry.get("outcome", entry.get("value"))
                prob = entry.get("prob", entry.get("probability"))
                if isinstance(prob, dict):
                    prob = float(self.expr.eval(prob, ctx))
                else:
                    prob = float(prob)
                ck = self.expr.eval(
                    ct.get("canonicalKey", {"template": f"chance:{outcome_val}"}),
                    {**ctx, "outcome": outcome_val},
                )
                effect_ref = effect_map.get(str(outcome_val), f"do_{outcome_val}")
                outcomes.append(
                    ChanceOutcome(
                        key=str(outcome_val),
                        probability=prob,
                        effect_ref=effect_ref,
                        canonical_key=ck,
                    )
                )
            return outcomes

        if "uniform" in prob_expr:
            over = prob_expr["uniform"]["over"]
            param_def = ct.get("params", {}).get(over, {})
            candidates = self._resolve_param_domain(param_def, state, ctx)
            if not candidates:
                return []
            prob = 1.0 / len(candidates)
            outcomes = []
            for c in candidates:
                cid = c.get("id", str(c)) if isinstance(c, dict) else str(c)
                ck = self.expr.eval(
                    ct.get("canonicalKey", {"template": f"chance:{cid}"}),
                    {**ctx, "outcome": cid},
                )
                outcomes.append(
                    ChanceOutcome(
                        key=cid,
                        probability=prob,
                        effect_ref=effect_map.get(cid, ct.get("effectRef", "")),
                        canonical_key=ck,
                    )
                )
            return outcomes

        return []

    # ── Effector execution ───────────────────────────────────────────

    def _execute_effector(self, effect_name: str, ctx: dict, state: dict):
        """Execute a named effector's op sequence."""
        effector = self._effectors.get(effect_name)
        if effector is None:
            return
        for op in effector.get("ops", []):
            self._execute_op(op, ctx, state)

    def _execute_op(self, op: dict, ctx: dict, state: dict):
        """Execute a single effect operation."""
        op_type = op.get("op")

        # ── Ground array ops ──────────────────────────────────────────

        if op_type == "setIndex":
            arr_name = self._resolve_array_name(op, ctx)
            at = self.expr.eval(op["at"], ctx)
            value = self.expr.eval(op["value"], ctx)
            arr = state["_arrays"].get(arr_name)
            if arr is not None and isinstance(at, int) and 0 <= at < len(arr):
                arr[at] = value

        elif op_type == "append":
            arr_name = self._resolve_array_name(op, ctx)
            raw_value = op["value"]
            # Support both simple expressions and dict-of-expressions
            if isinstance(raw_value, dict):
                # Check if ALL keys are known expression types
                is_expr_dict = all(_is_expr_key(k) for k in raw_value)
                if is_expr_dict:
                    value = self.expr.eval(raw_value, ctx)
                else:
                    # Value dict: evaluate each field's expression
                    value = {}
                    for k, v in raw_value.items():
                        value[k] = self.expr.eval(v, ctx) if isinstance(v, dict) else v
            else:
                value = raw_value
            arr = state["_arrays"].get(arr_name)
            if arr is not None:
                arr.append(value)
            elif isinstance(arr_name, str) and isinstance(state["env"].get(arr_name), list):
                # Env lists are shared by reference across clones — rebind.
                state["env"][arr_name] = state["env"][arr_name] + [value]

        elif op_type == "remove":
            """Multiset difference A⊖B — remove up to ``count`` matches by value.

            Rebind (never mutate in place) so env lists — which are shared
            by reference across cloned states — are never corrupted.
            """
            arr_name = self._resolve_array_name(op, ctx)
            value = self.expr.eval(op["value"], ctx)
            count = int(self.expr.eval(op.get("count", {"const": 1}), ctx))
            arr = state["_arrays"].get(arr_name)
            if arr is None and isinstance(arr_name, str) and isinstance(state["env"].get(arr_name), list):
                state["env"][arr_name] = _remove_matches(state["env"][arr_name], value, count)
                return
            if arr is None:
                return
            state["_arrays"][arr_name] = _remove_matches(arr, value, count)

        elif op_type == "setArray":
            """Wholesale array replacement (e.g. rewriting a meld list).

            The value is a fresh list by construction, so env-list rebinding
            is safe against the clone-sharing trap.
            """
            arr_name = self._resolve_array_name(op, ctx)
            value = self.expr.eval(op["value"], ctx)
            fresh = list(value) if isinstance(value, list) else []
            if isinstance(arr_name, str) and arr_name in state["_arrays"]:
                state["_arrays"][arr_name] = fresh
            elif isinstance(arr_name, str) and arr_name in state["env"]:
                state["env"][arr_name] = fresh

        elif op_type == "trimByKey":
            arr_name = op["array"]
            max_val = int(self.expr.eval(op["max"], ctx))
            key = op["key"]
            value = self.expr.eval(op["value"], ctx)
            on_evict = op.get("onEvict", [])
            arr = state["_arrays"].get(arr_name)
            if arr is None or not isinstance(arr, list):
                return
            # Group by key, trim each group
            filtered = [x for x in arr if x.get(key) != value]
            group = [x for x in arr if x.get(key) == value]
            while len(group) > max_val:
                evicted = group.pop(0)
                if on_evict:
                    # Compute evicted board index from cell_id (e.g. "cell_1_2")
                    evicted_cell = evicted.get("cell_id", "")
                    evicted_idx = 0
                    try:
                        parts = evicted_cell.split("_")
                        board_arr = state["_arrays"].get("board", [])
                        bs = int(len(board_arr) ** 0.5) if board_arr else 3
                        if len(parts) >= 3:
                            # cell_id is "cell_{row}_{col}"; board index is
                            # row-major (row * bs + col), matching do_place.
                            evicted_idx = int(parts[-2]) * bs + int(parts[-1])
                    except (ValueError, IndexError):
                        pass
                    evict_ctx = {**ctx, "$evicted": evicted, "evicted_index": evicted_idx}
                    for eop in on_evict:
                        self._execute_op(eop, evict_ctx, state)
            state["_arrays"][arr_name] = filtered + group

        # ── Environment ops ───────────────────────────────────────────

        elif op_type == "setEnv":
            key = op["key"]
            value = self.expr.eval(op["value"], ctx)
            state["env"][key] = value

        elif op_type == "inc":
            key = op["key"]
            by = int(self.expr.eval(op["by"], ctx))
            current = state["env"].get(key, 0)
            if not isinstance(current, (int, float)):
                current = 0
            state["env"][key] = current + by

        # ── Control flow ──────────────────────────────────────────────

        elif op_type == "branch":
            cond = self.expr.eval(op["if"], ctx)
            branch_ops = op.get("then") if cond else op.get("else", [])
            for sub_op in branch_ops:
                self._execute_op(sub_op, ctx, state)

        elif op_type == "callEffect":
            effect_ref = op["effectRef"]
            sub_ctx = {**ctx}
            for k, v in op.get("args", {}).items():
                sub_ctx[k] = self.expr.eval(v, ctx)
            self._execute_effector(effect_ref, sub_ctx, state)

        elif op_type == "forEach":
            items = self.expr.eval(op["list"], ctx)
            as_var = op.get("as", "$item")
            for item in items:
                sub_ctx = {**ctx, as_var: item}
                for sub_op in op.get("do", []):
                    self._execute_op(sub_op, sub_ctx, state)

        # ── Events / triggers ─────────────────────────────────────────

        elif op_type == "emit":
            event_name = op["event"]
            payload = {k: self.expr.eval(v, ctx) for k, v in op.get("payload", {}).items()}
            state.setdefault("_pending_events", []).append((event_name, payload))

        elif op_type == "enqueueEffect":
            args = {k: self.expr.eval(v, ctx) for k, v in op.get("args", {}).items()}
            state.setdefault("_pending_effects", []).append((op["effectRef"], args))

    def _resolve_array_name(self, op: dict, ctx: dict) -> Any:
        """Resolve an effector's ``array`` field: literal name or expression."""
        raw = op.get("array")
        if isinstance(raw, dict):
            return self.expr.eval(raw, ctx)
        return raw

    # ── Triggers ──────────────────────────────────────────────────────

    def _run_triggers(self, state: dict):
        pending = state.pop("_pending_events", [])
        pending_effects = state.pop("_pending_effects", [])

        for event_name, payload in pending:
            for trigger in self._triggers:
                if trigger["event"] != event_name:
                    continue
                cond = trigger.get("condition")
                if cond is not None:
                    t_ctx = self._build_context(state)
                    t_ctx["$event"] = payload
                    if not self.expr.eval(cond, t_ctx):
                        continue
                t_ctx = self._build_context(state)
                t_ctx["$event"] = payload
                t_ctx.update(payload)
                self._execute_effector(trigger["effectRef"], t_ctx, state)

        for effect_ref, args in pending_effects:
            e_ctx = self._build_context(state)
            e_ctx.update(args)
            self._execute_effector(effect_ref, e_ctx, state)

    # ── Context building ──────────────────────────────────────────────

    def _build_context(self, state: dict, action: Optional[ActionInstance] = None) -> dict:
        """Build expression evaluation context from state."""
        constants = self._constants
        ctx: dict[str, Any] = {
            "$state": state,
            "$env": state.get("env", {}),
            "$constants": constants,
            "$players": self._players,
            "_query_fn": self._query_fn,
        }

        # Flatten constants into top-level context for easy access
        for k, v in constants.items():
            if isinstance(v, (int, float, str)):
                ctx[k] = v

        # Add ground arrays as context vars for convenience
        arrays = state.get("_arrays", {})
        for k, v in arrays.items():
            ctx[f"${k}"] = v

        if action is not None:
            ctx["$action"] = action
            ctx["$params"] = action.params
            for k, v in action.params.items():
                ctx[k] = v
                ctx[f"${k}"] = v

        return ctx

    def _query_fn(self, query_expr: dict, ctx: dict) -> list:
        """Query function for ExprEvaluator.

        Handles both:
          - Direct: {"view": "cell", "where": ...}  (from terminal/query refs)
          - Wrapped: {"query": {"view": "cell", "where": ...}}  (from count expr)
        """
        if "query" in query_expr:
            query_expr = query_expr["query"]
        view_name = query_expr.get("view", "")
        entities = self._materialize_view(ctx.get("$state", {}), view_name)
        filter_expr = query_expr.get("filter") or query_expr.get("where")
        if filter_expr:
            filter_fn = self._get_compiled_filter(filter_expr)
            filtered = []
            for ent in entities:
                ent_ctx = {**ctx, "$node": ent, "$self": ent}
                if filter_fn(ent_ctx):
                    filtered.append(ent)
            return filtered
        return entities

    def _get_compiled_filter(self, filter_expr: dict) -> callable:
        """Return a compiled closure for ``filter_expr`` (cached by id)."""
        fid = id(filter_expr)
        fn = self._compiled_filters.get(fid)
        if fn is None:
            fn = self.expr.compile(filter_expr)
            self._compiled_filters[fid] = fn
        return fn

    @classmethod
    def from_json(cls, path: str | Path, **kwargs) -> "GameEngine":
        """Load engine from a v5.0 JSON rules file."""
        with open(path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        return cls(rules, **kwargs)

    # ── Player parser ─────────────────────────────────────────────────

    @staticmethod
    def _parse_players(raw: list | None) -> list[dict]:
        """Accept both ['p_black', 'p_white'] and [{'id': 'p_black'}, ...]."""
        if not raw:
            return [{"id": "p_black"}, {"id": "p_white"}]
        if isinstance(raw[0], str):
            return [{"id": pid} for pid in raw]
        return raw


# ── Helpers ────────────────────────────────────────────────────────────

_EXPR_KEYS = frozenset(
    {
        "const",
        "var",
        "get",
        "eq",
        "neq",
        "gt",
        "gte",
        "lt",
        "lte",
        "and",
        "or",
        "not",
        "if",
        "switch",
        "call",
        "template",
        "concat",
        "expr",
        "query",
        "count",
        "filter",
        "any",
        "all",
        "map",
        "choose",
        "range",
        "sort",
        "group",
        "distinct",
        "contains",
        "sum",
        "max",
        "min",
        "at",
        "add",
        "sub",
        "mul",
        "div",
        "ref",
    }
)


def _is_expr_key(key: str) -> bool:
    """Check if a dict key is a known expression type, not a field name.

    Used to distinguish ``{"cell_id": ..., "player_id": ...}`` (value dict)
    from ``{"get": [...]}`` (expression).
    """
    return key in _EXPR_KEYS


def _remove_matches(arr: list, value: Any, count: int) -> list:
    """Return ``arr`` with up to ``count`` occurrences of ``value`` removed."""
    out = []
    removed = 0
    for item in arr:
        if removed < count and item == value:
            removed += 1
        else:
            out.append(item)
    return out


def _cartesian_product(param_domains: dict[str, list]) -> list[dict]:
    """Cartesian product of multiple parameter domains."""
    if not param_domains:
        return [{}]
    keys = list(param_domains.keys())
    values = list(param_domains.values())
    results: list[dict] = []

    def _recurse(idx: int, current: dict):
        if idx == len(keys):
            results.append(dict(current))
            return
        for val in values[idx]:
            current[keys[idx]] = val
            _recurse(idx + 1, current)

    _recurse(0, {})
    return results
