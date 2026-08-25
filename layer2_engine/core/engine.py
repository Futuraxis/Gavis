"""Game engine — interprets v5.0 rules JSON as a stochastic game model.

GameEngine — the single solver-facing contract (Layer 2 to Layer 3): all solvers interact
with the game exclusively through this engine.

Two-layer state architecture:
  - Ground state: compact arrays + env scalars
  - Derived views: computed on-the-fly via derivation rules (grid, enum, literal)
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Callable

from .expr_eval import ExprEvaluator
from .rules_compiler import RulesCompiler
from .state_graph import (
    ActionInstance,
    ChanceOutcome,
    DerivedViewEngine,
    clone_state,
    create_initial_state,
)

logger = logging.getLogger(__name__)

# Action-expansion guard: parameter-domain cartesian products beyond this
# size are a rules bug (combination explosion) and raise instead of
# silently stalling or exhausting memory.
_MAX_ACTION_COMBINATIONS = 65536
# Trigger cascades are drained in bounded cycles; an event chain longer
# than this is a rules bug and the remainder is dropped.
_TRIGGER_CYCLE_CAP = 16


class GameEngine:
    """A game engine that interprets a v5.0 rules JSON.

    This is the canonical solver-facing engine (the Layer 2 to Layer 3 contract):
    all Layer 3 solvers consume the game through this class.
    """

    def __init__(
        self,
        rules: dict,
        seed: int | None = None,
        variant: str | None = None,
        player_count: int | None = None,
    ):
        # Resolve the declarative ``variants`` section (pure data in the
        # JSON — choosing a variant / player count only selects declared
        # options; nothing game-specific is hardcoded here).
        self.variant = variant
        self.player_count = player_count
        self.rules = self._resolve_variants(rules)
        rules = self.rules
        self.rng = random.Random(seed)
        self.expr = ExprEvaluator()
        # Compiled query filters, keyed by id() of the filter expr object
        # (rule dicts live for the engine's lifetime, so ids stay stable).
        self._compiled_filters: dict[int, Callable[..., Any]] = {}
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

        # Derived view engine (lazy); rule aliases registered so ``call``
        # works in view fields (pure expression definitions).
        self._view_engine = DerivedViewEngine(self._schema, rules.get("functions", {}))

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
        except Exception as exc:
            # Real codegen bugs (not just UnsupportedShapeError) must not be
            # swallowed silently — log and fall back to the interpreter.
            logger.warning("规则编译失败，回退纯解释器: %s: %s", type(exc).__name__, exc)
            self._compiled = None

    def _resolve_variants(self, rules: dict) -> dict:
        """Resolve the declarative ``variants`` section (pure data).

        ``rules["variants"]`` is a self-describing option table; choosing a
        variant / player count only *selects declared data* — the engine
        hardcodes nothing about any concrete game:

        - ``variants.variant`` / ``variants.player_count``: default selection
        - ``variants.options[<variant>].constants``: per-variant constants patch
          (merged into the base ``constants``)
        - any other key whose value is an expression dict is evaluated with
          ``$variant`` / ``$player_count`` / ``$constants`` / ``$players``
          bound and stored into ``constants`` (e.g. ``player_ids``,
          ``deal_target`` — formulas live in the JSON)
        - ``variants.trim_players`` / ``variants.trim_utility``: keep only
          the selected player ids (``players`` list / ``utility`` entries)

        Rules without a ``variants`` section are returned unchanged.
        """
        spec = rules.get("variants")
        if not spec:
            return rules
        name = self.variant or spec.get("variant")
        count = self.player_count if self.player_count is not None else spec.get("player_count")
        options = spec.get("options", {}) or {}
        if name not in options:
            raise ValueError(f"unknown variant {name!r}; declared options: {sorted(options)}")
        constants = dict(rules.get("constants", {}))
        patch = options[name].get("constants", {}) or {}
        constants.update(patch)
        constants["variant"] = name
        constants["player_count"] = count
        ctx = {
            "$variant": name,
            "$player_count": count,
            "$constants": constants,
            "$players": rules.get("players", []),
        }
        evaluator = ExprEvaluator()
        evaluator.set_functions(rules.get("functions", {}))
        reserved = ("variant", "player_count", "options", "trim_players", "trim_utility")
        for key, value in spec.items():
            if key in reserved:
                continue
            if isinstance(value, dict) and any(k in value for k in _EXPR_KEYS):
                constants[key] = evaluator.eval(value, ctx)
        out = dict(rules)
        out["constants"] = constants
        pids = constants.get("player_ids")
        if spec.get("trim_players") and isinstance(pids, list):
            out["players"] = list(pids)
        if spec.get("trim_utility") and isinstance(pids, list):
            keep = set(pids)
            out["utility"] = [u for u in rules.get("utility", []) if u.get("player") in keep]
        return out

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

    # ── Solver contract API ─────────────────────────────────────────────

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

    def get_current_player(self, state: dict) -> str | None:
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

        # Base context built once per projection — rebuilding it per entity
        # (constants + arrays copies) made partial-info projection O(N²).
        base_ctx = None if default_level == "public" else self._build_context(state)

        for vname in view_names:
            entities = self._materialize_view(state, vname)
            if default_level == "public":
                # Perfect information: return all fields
                obs[vname] = entities
            else:
                # Partial information: filter fields per entity
                filtered = []
                for entity in entities:
                    entity_ctx = {**base_ctx, "$node": entity, "$viewer": viewer}
                    entity_obs = dict(entity)
                    dropped = False
                    for rule in rules:
                        if rule.get("view", "") != vname:
                            continue
                        rule_filter = rule.get("filter")
                        if rule_filter is not None:
                            if not self.expr.eval(rule_filter, entity_ctx):
                                # "drop": the filter decides entity survival —
                                # a failing filter removes the whole row
                                # (v5.2, e.g. werewolf's per-viewer role row).
                                if rule.get("drop"):
                                    dropped = True
                                    break
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
                    if dropped:
                        continue
                    filtered.append(entity_obs)
                obs[vname] = filtered

        # Also include env, with optional viewer-level filtering declared in
        # ``visibility.env`` (v5.2): per-field viewer rules hide secret env
        # scalars, e.g. werewolf's ``seerResult`` is only visible to the seer.
        env_obs = dict(state.get("env", {}))
        env_rules = visibility.get("env", {}) or {}
        if env_rules:
            env_ctx = self._build_context(state)
            env_ctx["$viewer"] = viewer
            for field, spec in env_rules.items():
                if field not in env_obs:
                    continue
                rule_filter = spec.get("filter") if isinstance(spec, dict) else None
                if rule_filter is not None and not self.expr.eval(rule_filter, env_ctx):
                    env_obs.pop(field, None)
        obs["env"] = env_obs
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

    def eval_expr(self, expr: dict, extra_ctx: dict | None = None) -> Any:
        """Evaluate a rules expression with optional extra context (generic).

        Generic Layer-2 service (v5.2): lets upper layers (e.g. frontend
        display helpers) evaluate any rules-declared alias/expression —
        e.g. Texas ``best5`` hand evaluation — without reaching into
        engine internals.  Nothing game-specific lives here; the
        expression itself comes from the rules JSON.
        """
        ctx: dict[str, Any] = {
            "$constants": self._constants,
            "$env": {},
            "$players": self._players,
            "_query_fn": self._query_fn,
        }
        if extra_ctx:
            ctx.update(extra_ctx)
        return self.expr.eval(expr, ctx)

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
        """Materialize a derived view from current state.

        Results are cached on the state dict itself (``_view_cache``):
        views are a pure function of (arrays, env, players, constants), and
        every ground-state mutation invalidates the cache via
        ``_execute_op``, so the cache can never go stale.  ``clone_state``
        drops the key, so each clone recomputes fresh.
        """
        cache = state.get("_view_cache")
        if cache is not None and view_name in cache:
            return cache[view_name]
        if self._compiled is not None and self._compiled.materialize is not None and "_arrays" in state:
            result = self._compiled.materialize(state, view_name)
            if result is not None:
                state.setdefault("_view_cache", {})[view_name] = result
                return result
        result = self._view_engine.materialize(state, view_name)
        state.setdefault("_view_cache", {})[view_name] = result
        return result

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
                # 与编译路径（_gen_chance）一致：缺失 outcome/prob 的坏条目
                # 优雅降级跳过，而不是 float(None) 抛 TypeError。
                if outcome_val is None or prob is None:
                    continue
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

    def _execute_effector(self, effect_name: str, ctx: dict, state: dict) -> None:
        """Execute a named effector's op sequence.

        An unknown effector is a rules typo (e.g. a bad ``effectRef``) and
        is logged rather than silently ignored.
        """
        effector = self._effectors.get(effect_name)
        if effector is None:
            logger.warning("unknown effector %r (rules typo?)", effect_name)
            return
        for op in effector.get("ops", []):
            self._execute_op(op, ctx, state)

    def _execute_op(self, op: dict, ctx: dict, state: dict) -> None:
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
                self._invalidate_views(state)

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
                self._invalidate_views(state)
            elif isinstance(arr_name, str) and isinstance(state["env"].get(arr_name), list):
                # Env lists are shared by reference across clones — rebind.
                state["env"][arr_name] = state["env"][arr_name] + [value]
                self._invalidate_views(state)

        elif op_type == "remove":
            """Multiset difference A⊖B — remove up to ``count`` matches by value.

            Rebind (never mutate in place) so env lists — which are shared
            by reference across cloned states — are never corrupted.
            """
            arr_name = self._resolve_array_name(op, ctx)
            value = self.expr.eval(op["value"], ctx)
            count = self.expr.eval(op.get("count", {"const": 1}), ctx)
            count = int(count) if count is not None else 1
            arr = state["_arrays"].get(arr_name)
            if arr is None and isinstance(arr_name, str) and isinstance(state["env"].get(arr_name), list):
                state["env"][arr_name] = _remove_matches(state["env"][arr_name], value, count)
                self._invalidate_views(state)
                return
            if arr is None:
                return
            state["_arrays"][arr_name] = _remove_matches(arr, value, count)
            self._invalidate_views(state)

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
                self._invalidate_views(state)
            elif isinstance(arr_name, str) and arr_name in state["env"]:
                state["env"][arr_name] = fresh
                self._invalidate_views(state)

        elif op_type == "trimByKey":
            arr_name = op["array"]
            max_val = self.expr.eval(op["max"], ctx)
            max_val = int(max_val) if max_val is not None else 0
            key = op["key"]
            value = self.expr.eval(op["value"], ctx)
            on_evict = op.get("onEvict", [])
            arr = state["_arrays"].get(arr_name)
            if arr is None or not isinstance(arr, list):
                return
            # Group by key, trim each group; 非 dict 元素按"不匹配该键"保留
            filtered = [x for x in arr if not (isinstance(x, dict) and x.get(key) == value)]
            group = [x for x in arr if isinstance(x, dict) and x.get(key) == value]
            while len(group) > max_val:
                evicted = group.pop(0)
                if on_evict:
                    # Compute evicted board index from cell_id (e.g. "cell_1_2"):
                    # row-major (row * cols + col), matching do_place.  cols
                    # comes from the grid view over the board array (not a
                    # square-board sqrt guess) so non-square boards work.
                    evicted_cell = evicted.get("cell_id", "")
                    evicted_idx = 0
                    try:
                        parts = evicted_cell.split("_")
                        if len(parts) >= 3:
                            board_arr = state["_arrays"].get("board", [])
                            cols = self._grid_view_cols("board")
                            if cols is None:
                                cols = int(len(board_arr) ** 0.5) if board_arr else 3
                            evicted_idx = int(parts[-2]) * cols + int(parts[-1])
                    except (ValueError, IndexError):
                        pass
                    evict_ctx = {**ctx, "$evicted": evicted, "evicted_index": evicted_idx}
                    for eop in on_evict:
                        self._execute_op(eop, evict_ctx, state)
            state["_arrays"][arr_name] = filtered + group
            self._invalidate_views(state)

        # ── Environment ops ───────────────────────────────────────────

        elif op_type == "setEnv":
            key = op["key"]
            value = self.expr.eval(op["value"], ctx)
            state["env"][key] = value
            self._invalidate_views(state)

        elif op_type == "inc":
            key = op["key"]
            by = self.expr.eval(op["by"], ctx)
            by = int(by) if by is not None else 1
            current = state["env"].get(key, 0)
            if not isinstance(current, (int, float)):
                current = 0
            state["env"][key] = current + by
            self._invalidate_views(state)

        # ── Control flow ──────────────────────────────────────────────

        elif op_type == "branch":
            cond = self.expr.eval(op["if"], ctx)
            branch_ops = op.get("then") if cond else op.get("else", [])
            # 缺 ``then`` 且条件为真 → 空分支（不再遍历 None 崩溃）
            for sub_op in branch_ops or []:
                self._execute_op(sub_op, ctx, state)

        elif op_type == "callEffect":
            effect_ref = op["effectRef"]
            sub_ctx = {**ctx}
            for k, v in op.get("args", {}).items():
                sub_ctx[k] = self.expr.eval(v, ctx)
            self._execute_effector(effect_ref, sub_ctx, state)

        elif op_type == "forEach":
            items = self.expr.eval(op["list"], ctx)
            # list 求值为 None/非 list → 空迭代（不再 TypeError）
            if not isinstance(items, list):
                items = []
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

    @staticmethod
    def _invalidate_views(state: dict):
        """Drop the per-state view cache after any ground-state mutation."""
        state.pop("_view_cache", None)

    def _grid_view_cols(self, arr_name: str) -> int | None:
        """Columns of the grid view over ``arr_name``, per the rules schema.

        The grid view's ``cols`` is the authority for flat (row-major)
        indices into the array; ``None`` when no grid view covers it.
        """
        for vdef in self._schema.get("derivedViews", {}).values():
            src = vdef.get("from", {})
            if src.get("type") == "grid" and src.get("array") == arr_name:
                cols = self.expr.eval(
                    src.get("cols", {"const": 1}),
                    {"$constants": self._constants, "$players": self._players, "$env": {}},
                )
                if isinstance(cols, (int, float)) and cols > 0:
                    return int(cols)
        return None

    # ── Triggers ──────────────────────────────────────────────────────

    def _run_triggers(self, state: dict) -> None:
        """Process queued events and effects, cascading included.

        Trigger effectors may ``emit`` further events / ``enqueueEffect``
        during execution; these are drained in a loop so cascading
        triggers actually fire (previously they were dropped after the
        first pass).  The cycle cap bounds runaway chains; on exhaustion
        the remainder is dropped rather than processed forever.
        """
        for _ in range(_TRIGGER_CYCLE_CAP):
            pending = state.pop("_pending_events", [])
            pending_effects = state.pop("_pending_effects", [])
            if not pending and not pending_effects:
                return

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

        # Cycle cap exhausted — drop whatever was re-queued.
        logger.warning("trigger cascade exceeded %d cycles; dropping remainder", _TRIGGER_CYCLE_CAP)
        state.pop("_pending_events", None)
        state.pop("_pending_effects", None)

    # ── Context building ──────────────────────────────────────────────

    def _build_context(self, state: dict, action: ActionInstance | None = None) -> dict:
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

    def _get_compiled_filter(self, filter_expr: dict) -> Callable[..., Any]:
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
    """Cartesian product of multiple parameter domains.

    Guards against combination explosion: multi-parameter templates with
    large domains are a rules bug and raise instead of stalling.
    """
    if not param_domains:
        return [{}]
    total = 1
    for vals in param_domains.values():
        total *= len(vals)
        if total > _MAX_ACTION_COMBINATIONS:
            raise ValueError(
                f"action expansion exceeds {_MAX_ACTION_COMBINATIONS} combinations "
                f"({total} from domains {list(param_domains.keys())})"
            )
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
