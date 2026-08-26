"""poker family — community-card betting games (Texas Hold'em, …).

Detection signal: the ground state declares hole-card and community
arrays (array names containing ``hole`` / ``community``) and the action
set exposes ``raise`` / ``call`` / ``fold`` choices.  The built spec
mirrors the ``_poker_*`` closures in ``platform/games.py``, generalized
to read seats, array names, env keys and constants from the rules data:

- hole arrays are located per seat (``{pid}_hole`` or the ``p_``-stripped
  key form), env stack/committed/folded scalars likewise
  (``{pid}_stack`` / ``{pid}_committed`` / ``{pid}_folded``);
- the pot is the sum of every seat's committed chips; the reveal flag is
  ``over and env.last_action == "showdown"``, so the AI hole is only
  exposed after a showdown (the hidden-information red line);
- ``raise_amounts`` come from the legal raise actions, falling back to
  the rules ``constants.raise_grid`` (capped by ``constants.stack_size``);
- ``street_name`` reads a ``constants.street_names`` list when declared,
  else falls back to ``"street N"``;
- the hand name evaluates the rules ``best5``-like alias through the
  engine's generic ``eval_expr`` (``None`` without raising when absent).

Solvers are assembled exclusively through ``SolverProvider`` — the
imperfect-information Hybrid solver with ``allow_unknown=True`` (the
game id is a custom, unregistered id).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance

from ....solver_provider import SolverHandle, SolverProvider
from ..games import GameSpec, PlayError
from .helpers import (
    declared_player_counts,
    engine_from_rules_dict,
    normalize_players,
    resolve_all_chance,
)

if TYPE_CHECKING:
    from ..session import GameSession

FAMILY_ID = "poker"

#: 族默认难度预算（与既有德州条目同一量级）。
DIFFICULTY_BUDGETS = {"easy": 150, "normal": 500, "hard": 1200}

#: ``best5`` 首元素类别 → 中文手牌名（frontend 展示域；镜像
#: ``engine_helpers._TEXAS_HAND_NAMES``，类别语义来自规则 alias）。
_HAND_NAMES = {
    0: "高牌",
    1: "一对",
    2: "两对",
    3: "三条",
    4: "顺子",
    5: "同花",
    6: "葫芦",
    7: "四条",
    8: "同花顺",
}


def _pid_key(pid: str) -> str:
    """Env/hole key suffix for a player id (strip a leading ``p_`` prefix)."""
    return pid[2:] if pid.startswith("p_") else pid


def _array_names(ground: dict) -> list[str]:
    """Ground array names declared with ``type == \"array\"``."""
    return [name for name, decl in ground.items() if isinstance(decl, dict) and decl.get("type") == "array"]


def _param_domains(actions: list) -> set[str]:
    """All parameter-domain values declared by the action templates."""
    out: set[str] = set()
    for action in actions:
        params = action.get("params", {}) if isinstance(action, dict) else {}
        for param in params.values():
            if isinstance(param, dict):
                domain = param.get("domain")
                if isinstance(domain, list):
                    out.update(str(value) for value in domain)
    return out


def detect(rules: dict) -> bool:
    """Whether ``rules`` is a community-card betting game.

    Signal: ground arrays contain ``hole`` and ``community`` names, and
    the action params expose ``raise`` / ``call`` / ``fold`` choices.
    """
    ground = rules.get("groundState", {})
    names = _array_names(ground if isinstance(ground, dict) else {})
    if not (any("hole" in name for name in names) and any("community" in name for name in names)):
        return False
    actions = rules.get("actions", [])
    domains = _param_domains(actions if isinstance(actions, list) else [])
    return {"raise", "call", "fold"} <= domains


def _find_best5_alias(rules: dict) -> str | None:
    """Name of the hand-evaluation alias (``best5`` or a key containing it)."""
    functions = rules.get("functions", {})
    if not isinstance(functions, dict):
        return None
    if "best5" in functions:
        return "best5"
    for name in functions:
        if isinstance(name, str) and "best5" in name:
            return name
    return None


def _first_present(names: tuple[str, ...], container: dict) -> str | None:
    """First ``names`` entry present as a key of ``container`` (``None`` otherwise)."""
    for name in names:
        if name in container:
            return name
    return None


def _env_number(env: dict, pid: str, suffix: str, default: int) -> int:
    """Numeric env scalar for ``pid`` (``{pid}_{suffix}`` or ``p_``-stripped key)."""
    for key in (f"{pid}_{suffix}", f"{_pid_key(pid)}_{suffix}"):
        value = env.get(key)
        if value is not None:
            return int(value)
    return default


def _env_flag(env: dict, pid: str, suffix: str) -> bool:
    """Boolean env flag for ``pid`` (``{pid}_{suffix}`` or ``p_``-stripped key)."""
    for key in (f"{pid}_{suffix}", f"{_pid_key(pid)}_{suffix}"):
        value = env.get(key)
        if value is not None:
            return bool(value)
    return False


def build_spec(game_id: str, rules: dict) -> GameSpec:
    """Build the platform ``GameSpec`` for a validated poker rules dict.

    Args:
        game_id: The custom game id (registry-assigned, whitelisted).
        rules: Validated v5 rules JSON (poker family).

    Returns:
        The ``GameSpec`` wiring engine / solver / session closures —
        snapshot keys follow the ``PokerSnapshot`` contract in
        ``platform-frontend/src/types.ts``.
    """
    constants = rules.get("constants", {}) if isinstance(rules.get("constants", {}), dict) else {}
    ground = rules.get("groundState", {}) if isinstance(rules.get("groundState", {}), dict) else {}
    meta = rules.get("meta", {}) if isinstance(rules.get("meta", {}), dict) else {}
    seats = normalize_players(rules) or ("p_sb", "p_bb")
    stack_size = int(constants["stack_size"]) if isinstance(constants.get("stack_size"), int) else 0
    raise_grid = constants.get("raise_grid")
    street_names = constants.get("street_names")
    best5_alias = _find_best5_alias(rules)
    hole_of: dict[str, str | None] = {
        pid: _first_present((f"{pid}_hole", f"{_pid_key(pid)}_hole"), ground) for pid in seats
    }
    community_name = next((name for name in ground if "community" in name), None)

    def _create_engine(seed: int, player_count: int = 2) -> GameEngine:
        return engine_from_rules_dict(rules, seed, player_count=player_count)

    def _create_solver(provider: SolverProvider, engine: GameEngine, seed: int, budget: int) -> SolverHandle:
        return provider.create_solver(
            game_id,
            "hybrid",
            engine,
            seed,
            budget,
            allow_unknown=True,
            imperfect_information=True,
            opponent_model="uniform",
        )

    def _resolve_start(session: GameSession) -> None:
        """Deal blinds and hole cards (chance nodes) before play starts."""
        session.state = resolve_all_chance(session.engine, session.state)

    def _ai_opens(session: GameSession) -> bool:
        """The AI may move first (the betting round decides who acts)."""
        return True

    def _parse_human_action(session: GameSession, payload: dict) -> ActionInstance:
        if session.over:
            raise PlayError("本局已结束")
        if session.current_player != session.player_pid:
            raise PlayError("还没轮到你")
        choice = payload.get("choice")
        amount = payload.get("amount")
        for action in session.engine.get_legal_actions(session.state):
            if action.params.get("choice") != choice:
                continue
            if amount is None or amount == "" or action.params.get("amount") == amount:
                return action
        raise PlayError(f"非法动作: {choice} {amount}")

    def _apply_human(session: GameSession, action: ActionInstance) -> None:
        session.state = session.engine.apply_action(session.state, action)
        session.state = resolve_all_chance(session.engine, session.state)

    def _run_ai(session: GameSession, on_ai_action: Callable[[ActionInstance], None] | None = None) -> None:
        """Let the AI act while it is its turn (multi-action rounds)."""
        while not session.over and session.current_player == session.ai_pid:
            action = session.solver.select_action(session.state)
            if action is None:  # search found nothing — random fallback
                legal = session.engine.get_legal_actions(session.state)
                action = random.choice(legal) if legal else None
            if action is None:
                break
            session.state = session.engine.apply_action(session.state, action)
            session.state = resolve_all_chance(session.engine, session.state)
            session.last_ai_info["action"] = action.canonical_key
            if on_ai_action is not None:
                on_ai_action(action)

    def _best_hand_name(engine: GameEngine, cards: list) -> str | None:
        """Best-hand display name via the rules alias (``None`` when absent)."""
        if best5_alias is None or not cards:
            return None
        try:
            value = engine.eval_expr({"call": [best5_alias, {"const": list(cards)}]}, {"$cards": list(cards)})
        except Exception:  # noqa: BLE001 — a display helper never breaks the snapshot
            return None
        category = value[0] if isinstance(value, list) and value else None
        if not isinstance(category, int):
            return None
        return _HAND_NAMES.get(category, "未知")

    def _build_snapshot(session: GameSession) -> dict:
        env = session.state["env"]
        arrs = session.state["_arrays"]
        over = session.over
        revealed = over and env.get("last_action") == "showdown"
        legal: list[dict] = []
        if not over and session.current_player == session.player_pid:
            for action in session.engine.get_legal_actions(session.state):
                legal.append({"choice": action.params.get("choice"), "amount": action.params.get("amount")})
        raise_amts = sorted({a["amount"] for a in legal if a["choice"] == "raise" and isinstance(a["amount"], int)})
        if not raise_amts and isinstance(raise_grid, list):
            raise_amts = sorted(
                {int(v) for v in raise_grid if isinstance(v, int) and (stack_size <= 0 or v <= stack_size)}
            )
        pot = sum(_env_number(env, pid, "committed", 0) for pid in seats)
        street = int(env.get("street", 0))
        names = street_names if isinstance(street_names, list) and street_names else None
        street_name = names[street] if names and 0 <= street < len(names) else f"street {street}"

        def _cards(pid: str) -> list:
            hole = hole_of.get(pid)
            return list(arrs.get(hole, [])) if hole else []

        def _hand_name(pid: str) -> str | None:
            if not over:
                return None
            cards = list(_cards(pid))
            if community_name:
                cards.extend(list(arrs.get(community_name, [])))
            return _best_hand_name(session.engine, cards)

        return {
            "game_id": session.game_id,
            "player_pid": session.player_pid,
            "ai_pid": session.ai_pid,
            "difficulty": session.difficulty,
            "over": over,
            "winner": session.winner,
            "turn": session.current_player,
            "phase": env.get("phase"),
            "street": street,
            "street_name": street_name,
            "pot": pot,
            "community": list(arrs.get(community_name, [])) if community_name else [],
            "my_hole": _cards(session.player_pid),
            "ai_hole": _cards(session.ai_pid) if revealed else [],
            "revealed": revealed,
            "my_stack": _env_number(env, session.player_pid, "stack", 0),
            "ai_stack": _env_number(env, session.ai_pid, "stack", 0),
            "my_committed": _env_number(env, session.player_pid, "committed", 0),
            "ai_committed": _env_number(env, session.ai_pid, "committed", 0),
            "my_folded": _env_flag(env, session.player_pid, "folded"),
            "ai_folded": _env_flag(env, session.ai_pid, "folded"),
            "last_actor": env.get("last_actor"),
            "last_action": env.get("last_action"),
            "last_ai_action": session.last_ai_info.get("action"),
            "call_to": int(env.get("last_call_to", 0)),
            "my_hand_name": _hand_name(session.player_pid),
            "ai_hand_name": _hand_name(session.ai_pid),
            "payoff": session.engine.get_utility(session.state, session.player_pid) if over else None,
            "legal": legal,
            "raise_amounts": raise_amts,
        }

    def _describe_action(action: ActionInstance) -> str:
        return action.canonical_key

    return GameSpec(
        game_id=game_id,
        display_name=str(meta.get("gameId") or game_id),
        description=str(meta.get("description") or "") or "由规则翻译生成的扑克对弈游戏（poker 族）",
        kind="poker",
        board_size=None,
        seat_options=seats,
        seat_label="座位",
        difficulty_budgets=DIFFICULTY_BUDGETS,
        player_counts=declared_player_counts(rules),
        create_engine=_create_engine,
        create_solver=_create_solver,
        resolve_start=_resolve_start,
        ai_opens=_ai_opens,
        parse_human_action=_parse_human_action,
        apply_human=_apply_human,
        run_ai=_run_ai,
        build_snapshot=_build_snapshot,
        describe_action=_describe_action,
    )


__all__ = ["FAMILY_ID", "detect", "build_spec"]
