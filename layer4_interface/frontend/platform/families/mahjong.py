"""mahjong family — draw-and-discard tile games (guangdong / hongzhong / blood / …).

Detection signal: the ground state declares per-seat ``hand_*`` arrays
(``hand_p0`` …) and the action set exposes ``claim_*`` (peng / gang /
chi / win / pass) plus a ``win_self`` choice.  The built spec mirrors
the ``_mahjong_*`` closures in ``platform/games.py``, generalized to
read seats, variant defaults and player counts from the rules data:

- seats come from ``rules["players"]`` (``normalize_players``); the
  ``player_counts`` option is the declared default plus the full seat
  count — the standard template declares 2 seats as default with 4
  declared seats, yielding ``(2, 4)``;
- variant selection stays purely declarative: no ``variant`` argument is
  passed, so the engine uses the rules' declared default
  (``variants.variant``) — no variant name is hardcoded;
- the AI drives every non-human seat (2-player: the single AI; 4-player:
  all remaining seats) through their draw / claim / discard turns;
- ``ai_hand`` is only serialized after ``over`` — the rules'
  ``visibility`` section marks every ``hand_*`` view hidden to
  non-owners (the hidden-information red line); ``hand_counts`` /
  ``melds`` / ``discards`` still cover every seat.

Solvers are assembled exclusively through ``SolverProvider`` — the
heuristic mahjong solver with ``allow_unknown=True`` (the game id is a
custom, unregistered id).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance

from ....solver_provider import SolverHandle, SolverProvider
from ...engine_helpers import mahjong_tile_name
from ..games import GameSpec, PlayError
from .helpers import (
    declared_player_counts,
    engine_from_rules_dict,
    normalize_players,
    resolve_all_chance,
)

if TYPE_CHECKING:
    from ..session import GameSession

FAMILY_ID = "mahjong"

#: 族默认难度预算（与既有麻将条目同一约定：启发式求解器无预算旋钮，
#: 难度档仅作展示，审查 Minor 12）。
DIFFICULTY_BUDGETS = {"easy": 1, "normal": 1, "hard": 1}


def _array_names(ground: dict) -> list[str]:
    """Ground array names declared with ``type == \"array\"``."""
    return [name for name, decl in ground.items() if isinstance(decl, dict) and decl.get("type") == "array"]


def _action_ids(actions: list) -> list[str]:
    """Action template ids declared by the rule file."""
    return [str(action.get("id", "")) for action in actions if isinstance(action, dict)]


def detect(rules: dict) -> bool:
    """Whether ``rules`` is a draw-and-discard mahjong-style game.

    Signal: ground arrays carry a ``hand_`` prefix (per-seat hands) and
    the action set exposes ``claim_*`` choices plus a ``win`` action
    (``win_self`` / ``claim_win``).
    """
    ground = rules.get("groundState", {})
    if not isinstance(ground, dict):
        return False
    if not any(str(name).startswith("hand_") for name in _array_names(ground)):
        return False
    actions = rules.get("actions", [])
    ids = _action_ids(actions if isinstance(actions, list) else [])
    return any(aid.startswith("claim_") for aid in ids) and any("win" in aid for aid in ids)


def _player_counts(rules: dict, seats: tuple[str, ...]) -> tuple[int, ...]:
    """Player-count options: declared default plus the full seat count.

    ``declared_player_counts`` reads ``variants.player_count`` (the
    default selection).  The standard mahjong template declares the
    default as 2 while four seats exist in ``rules["players"]`` — the
    engine accepts any count up to the seats, so the full seat count is
    offered as a second option, giving ``(2, 4)`` for the template.  A
    two-seat custom rules dict stays ``(2,)``.
    """
    counts = list(declared_player_counts(rules))
    full = len(seats)
    if full > 0 and full not in counts:
        counts.append(full)
    return tuple(sorted(counts))


def build_spec(game_id: str, rules: dict) -> GameSpec:
    """Build the platform ``GameSpec`` for a validated mahjong rules dict.

    Args:
        game_id: The custom game id (registry-assigned, whitelisted).
        rules: Validated v5 rules JSON (mahjong family).

    Returns:
        The ``GameSpec`` wiring engine / solver / session closures —
        snapshot keys follow the ``MahjongSnapshot`` contract in
        ``platform-frontend/src/types.ts``.
    """
    meta = rules.get("meta", {}) if isinstance(rules.get("meta", {}), dict) else {}
    seats = normalize_players(rules) or ("p0", "p1", "p2", "p3")

    def _create_engine(seed: int, player_count: int = 2) -> GameEngine:
        return engine_from_rules_dict(rules, seed, player_count=player_count)

    def _create_solver(provider: SolverProvider, engine: GameEngine, seed: int, budget: int) -> SolverHandle:
        return provider.create_solver(game_id, "mahjong", engine, seed, budget, allow_unknown=True)

    def _resolve_start(session: GameSession) -> None:
        """Deal the wall (chance nodes) before play starts."""
        session.state = resolve_all_chance(session.engine, session.state)

    def _ai_opens(session: GameSession) -> bool:
        """The AI moves first when the human is not the first seat."""
        return session.player_pid != seats[0]

    def _parse_human_action(session: GameSession, payload: dict) -> ActionInstance:
        if session.over:
            raise PlayError("本局已结束")
        if session.current_player != session.player_pid:
            raise PlayError("还没轮到你")
        action_type = payload.get("type")
        for action in session.engine.get_legal_actions(session.state):
            if action.template_id != action_type:
                continue
            matched = True
            for key in ("tile", "tiles"):
                if key in payload and action.params.get(key) != payload[key]:
                    matched = False
            if matched:
                return action
        raise PlayError(f"非法动作: {action_type} {payload}")

    def _apply_human(session: GameSession, action: ActionInstance) -> None:
        session.state = session.engine.apply_action(session.state, action)
        session.state = resolve_all_chance(session.engine, session.state)

    def _run_ai(session: GameSession, on_ai_action: Callable[[ActionInstance], None] | None = None) -> None:
        """Drive every non-human seat through draw / claim / discard turns."""
        while not session.over and session.current_player is not None and session.current_player != session.player_pid:
            action = session.solver.select_action(session.state)
            if action is None:  # heuristic found nothing — random fallback
                legal = session.engine.get_legal_actions(session.state)
                action = random.choice(legal) if legal else None
            if action is None:
                break
            session.state = session.engine.apply_action(session.state, action)
            session.state = resolve_all_chance(session.engine, session.state)
            session.last_ai_info["action"] = action.canonical_key
            if on_ai_action is not None:
                on_ai_action(action)

    def _build_snapshot(session: GameSession) -> dict:
        env = session.state["env"]
        arrs = session.state["_arrays"]
        seats = session.spec.seat_options
        over = session.over

        def _hand(pid: str) -> list:
            return list(arrs.get(f"hand_{pid}", []))

        def _melds(pid: str) -> list:
            return list(arrs.get(f"melds_{pid}", []))

        def _discards(pid: str) -> list:
            return list(arrs.get(f"discard_{pid}", []))

        legal: list[dict] = []
        if not over and session.current_player == session.player_pid:
            for action in session.engine.get_legal_actions(session.state):
                legal.append({"type": action.template_id, **action.params})

        claim_queue = env.get("claim_queue")
        # During a claim the effective actor is the queue head, not env.turn.
        acting = (
            (claim_queue or [None])[int(env.get("claim_index", 0))]
            if env.get("phase") == "claim"
            else session.current_player
        )

        return {
            "game_id": session.game_id,
            "player_pid": session.player_pid,
            "ai_pid": session.ai_pid,
            "difficulty": session.difficulty,
            "over": over,
            "winner": session.winner,
            "turn": acting,
            "phase": env.get("phase"),
            "my_hand": _hand(session.player_pid),
            "ai_hand": _hand(session.ai_pid) if over else [],  # 隐藏红线：终局前不泄露对手手牌
            "hand_counts": {pid: len(_hand(pid)) for pid in seats},
            "melds": {pid: _melds(pid) for pid in seats},
            "discards": {pid: _discards(pid) for pid in seats},
            "wall_remaining": int(env.get("wall_count", 0)),
            "last_discard": env.get("last_discard"),
            "last_action": env.get("last_action"),
            "done": list(env.get("done", [])),
            "winners": list(env.get("winners", [])),
            "payoffs": list(env.get("payoffs", [])),
            "claim": (
                {
                    "queue": list(claim_queue or []),
                    "passed": int(env.get("claim_index", 0)),
                    "actor": env.get("actor"),
                }
                if env.get("phase") == "claim"
                else None
            ),
            "legal": legal,
            "last_ai_action": session.last_ai_info.get("action"),
        }

    def _tile_label(value: object) -> str:
        label = mahjong_tile_name(str(value)) if value is not None else ""
        return label or str(value)

    def _describe_action(action: ActionInstance) -> str:
        if action.template_id == "discard":
            return f"打 {_tile_label(action.params.get('tile'))}"
        if action.template_id == "win_self":
            return "自摸"
        if action.template_id == "claim_win":
            return "荣和"
        if action.template_id == "claim_peng":
            return f"碰 {_tile_label(action.params.get('tile'))}"
        if action.template_id == "claim_gang":
            return f"明杠 {_tile_label(action.params.get('tile'))}"
        if action.template_id == "claim_chi":
            tiles = action.params.get("tiles") or []
            return "吃 " + "".join(_tile_label(tile) for tile in tiles)
        if action.template_id == "gang_concealed":
            return f"暗杠 {_tile_label(action.params.get('tile'))}"
        if action.template_id == "gang_added":
            return f"加杠 {_tile_label(action.params.get('tile'))}"
        if action.template_id == "claim_pass":
            return "过"
        return action.canonical_key

    return GameSpec(
        game_id=game_id,
        display_name=str(meta.get("gameId") or game_id),
        description=str(meta.get("description") or "") or "由规则翻译生成的麻将对弈游戏（mahjong 族）",
        kind="mahjong",
        board_size=None,
        seat_options=seats,
        seat_label="座位",
        difficulty_budgets=DIFFICULTY_BUDGETS,
        player_counts=_player_counts(rules, seats),
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
