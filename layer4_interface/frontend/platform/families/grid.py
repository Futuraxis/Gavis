"""grid family — N×N placement/alignment games (moon chess, gomoku, connect-4 …).

Detection signal: the rules declare a grid cell view
(``derivedViews.cell.from.type == \"grid\"``) and a ground ``board``
array.  The built spec mirrors the ``_moon_*`` / ``_gomoku_*`` closures
in ``platform/games.py``: cell-index action parsing, apply + chance
resolution (vanish check), MCTS AI with budget, and a snapshot carrying
``board`` / ``board_size`` / ``win_length`` / ``turn`` / ``winner`` /
``over`` / ``round`` / ``last_vanish`` / ``last_vanish_color`` for the
generic grid board UI.

Solvers are assembled exclusively through ``SolverProvider`` (MCTS with
``allow_unknown=True`` — the game id is a custom, unregistered id).
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Callable

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance

from ....solver_provider import SolverHandle, SolverProvider
from ...engine_helpers import canonical_family_text
from ..games import GameSpec, PlayError
from .helpers import (
    action_cell_index,
    declared_player_counts,
    engine_from_rules_dict,
    normalize_players,
    rules_board_size,
)

if TYPE_CHECKING:
    from ..session import GameSession

FAMILY_ID = "grid"

#: 族默认难度预算（与平台既有 board 游戏同一量级）。
DIFFICULTY_BUDGETS = {"easy": 200, "normal": 800, "hard": 2000}

_CELL_ID_RE = re.compile(r"^cell_(\d+)_(\d+)$")


def detect(rules: dict) -> bool:
    """Whether ``rules`` is a grid placement game.

    Signal: ``derivedViews.cell.from.type == \"grid\"`` and a ground
    ``board`` array in ``groundState``.
    """
    cell_from = rules.get("derivedViews", {}).get("cell", {}).get("from", {})
    if cell_from.get("type") != "grid":
        return False
    return "board" in rules.get("groundState", {})


def build_spec(game_id: str, rules: dict) -> GameSpec:
    """Build the platform ``GameSpec`` for a validated grid rules dict.

    Args:
        game_id: The custom game id (registry-assigned, whitelisted).
        rules: Validated v5 rules JSON (grid family).

    Returns:
        The ``GameSpec`` wiring engine / solver / session closures.
    """
    constants = rules.get("constants", {})
    board_size = rules_board_size(rules) or 0
    win_length = constants.get("win_length")
    seats = normalize_players(rules) or ("p_black", "p_white")
    meta = rules.get("meta", {})
    descriptive = str(meta.get("description", "")) if isinstance(meta, dict) else ""

    def _create_engine(seed: int, player_count: int = 2) -> GameEngine:
        return engine_from_rules_dict(rules, seed, player_count=player_count)

    def _create_solver(provider: SolverProvider, engine: GameEngine, seed: int, budget: int) -> SolverHandle:
        return provider.create_solver(game_id, "mcts", engine, seed, budget, allow_unknown=True)

    def _resolve_start(session: GameSession) -> None:
        pass

    def _ai_opens(session: GameSession) -> bool:
        return session.player_pid == session.spec.seat_options[1]

    def _parse_human_action(session: GameSession, payload: dict) -> ActionInstance:
        if session.over:
            raise PlayError("本局已结束")
        if session.current_player != session.player_pid:
            raise PlayError("还没轮到你")
        try:
            cell_index = int(payload.get("cell_index", -1))
        except (TypeError, ValueError):
            raise PlayError("需要 cell_index") from None
        for action in session.engine.get_legal_actions(session.state):
            if action_cell_index(action, board_size) == cell_index:
                return action
        raise PlayError(f"非法落子: {cell_index}")

    def _find_vanished(session: GameSession, state: dict) -> tuple[int | None, str | None]:
        """Locate the vanished cell after a ``vanish`` chance outcome."""
        board = state["_arrays"].get("board")
        last_actor = state["env"].get("lastActor")
        last_cell = state["env"].get("lastPlacedCell")
        if not board or last_actor is None or last_cell is None or not board_size:
            return None, None
        match = _CELL_ID_RE.match(str(last_cell))
        if match is None:
            return None, None
        idx = int(match.group(1)) * board_size + int(match.group(2))
        if 0 <= idx < len(board) and board[idx] is None:
            return idx, last_actor
        return None, None

    def _resolve_chance(session: GameSession, state: dict) -> tuple[dict, int | None, str | None]:
        """Resolve all pending chance nodes; report a vanish if it fired."""
        vanished: int | None = None
        color: str | None = None
        while session.engine.get_node_type(state) == "chance":
            outcome, state = session.engine.sample_chance(state)
            if outcome.key == "vanish":
                vanished, color = _find_vanished(session, state)
        return state, vanished, color

    def _apply_human(session: GameSession, action: ActionInstance) -> None:
        session.state = session.engine.apply_action(session.state, action)
        state, vanished, color = _resolve_chance(session, session.state)
        session.state = state
        session.last_ai_info["vanish"] = vanished
        session.last_ai_info["vanish_color"] = color

    def _run_ai(session: GameSession, on_ai_action: Callable[[ActionInstance], None] | None = None) -> None:
        if session.over or session.current_player != session.ai_pid:
            return
        action = session.solver.select_action(session.state)
        if action is None:  # search found nothing — random fallback
            legal = session.engine.get_legal_actions(session.state)
            action = random.choice(legal) if legal else None
        if action is None:
            return
        session.state = session.engine.apply_action(session.state, action)
        session.last_ai_info["move"] = action_cell_index(action, board_size)
        state, vanished, color = _resolve_chance(session, session.state)
        session.state = state
        session.last_ai_info["vanish"] = vanished
        session.last_ai_info["vanish_color"] = color
        if on_ai_action is not None:
            on_ai_action(action)

    def _build_snapshot(session: GameSession) -> dict:
        return {
            "game_id": session.game_id,
            "player_pid": session.player_pid,
            "difficulty": session.difficulty,
            "board": list(session.state["_arrays"].get("board", [])),
            "board_size": board_size or None,
            "win_length": win_length if isinstance(win_length, int) else None,
            "turn": session.current_player,
            "winner": session.winner,
            "over": session.over,
            "last_ai_move": session.last_ai_info.get("move"),
            "last_vanish": session.last_ai_info.get("vanish"),
            "last_vanish_color": session.last_ai_info.get("vanish_color"),
            "round": session.state["env"].get("round", 0),
        }

    def _describe_action(action: ActionInstance) -> str:
        idx = action_cell_index(action, board_size)
        if idx >= 0 and board_size:
            return canonical_family_text("grid", f"cell_{idx // board_size}_{idx % board_size}")
        return canonical_family_text("grid", action.canonical_key)

    return GameSpec(
        game_id=game_id,
        display_name=str(meta.get("gameId") or game_id) if isinstance(meta, dict) else game_id,
        description=descriptive or "由规则翻译生成的对弈游戏（grid 族）",
        kind="board",
        board_size=board_size or None,
        seat_options=seats,
        seat_label="颜色",
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
