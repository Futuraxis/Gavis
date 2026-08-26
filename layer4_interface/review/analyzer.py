"""Post-match review analyzer (Layer 4, C4).

Turns a persisted ``MatchHistory.get`` record into a deterministic
``ReviewReport``: a list of ``KeyNode`` entries (turning point / winning
move / blunder), one mechanical improvement sentence, and a win/loss
summary.

The per-step evaluation is a *generic* fallback over public snapshot
fields only — hidden information is never read, so it can never leak into
the report text.  Terminal snapshots score from the recorded ``winner`` /
``payoff``; non-terminal snapshots score from the ``board`` /
``lastPlacedCell`` neighborhood heuristic.  C2's
``agent.evaluation.evaluate`` is preferred via a runtime import (C2 is
developed in parallel and its module may not exist yet); it is only
invoked when a snapshot embeds a live engine ``state`` and ``engine``,
otherwise the local proxy takes over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

#: Single-step evaluation drop (player perspective) that flags a blunder.
_BLUNDER_DROP = 0.3
#: Corner weight in the generic board neighborhood heuristic.
_CORNER_WEIGHT = 0.5
#: Center weight in the generic board neighborhood heuristic.
_CENTER_WEIGHT = 0.3

#: Display names for the P0 games (Layer-4 presentation concern only).
_GAME_NAMES: dict[str, str] = {
    "moon_chess": "月亮棋",
    "stochastic_gomoku": "随机五子棋",
    "texas_holdem": "德州扑克",
    "mahjong_guangdong": "广东麻将",
    "mahjong_hongzhong": "红中麻将",
    "mahjong_blood": "血战到底",
    "mahjong_sichuan": "四川麻将（血战到底）",
    "mahjong_changsha": "长沙麻将（258将）",
    "mahjong_taiwan": "台湾麻将（16张）",
}


@dataclass
class KeyNode:
    """One noteworthy step in a reviewed match.

    Attributes
    ----------
    step : int
        0-based index into the match move log.
    kind : str
        ``turning_point`` / ``winning_move`` / ``blunder``.
    why : str
        Mechanical reason (never references hidden information).
    """

    step: int
    kind: str
    why: str


@dataclass
class ReviewReport:
    """Deterministic post-match review produced by :func:`analyze`.

    Attributes
    ----------
    key_nodes : list[KeyNode]
        Detected key nodes (at least the turning point when moves exist).
    improvement : str
        One mechanical improvement sentence (reworded by C2 for persona).
    summary : str
        Win/loss + move-count summary.
    """

    key_nodes: list[KeyNode]
    improvement: str
    summary: str


def analyze(match: dict) -> ReviewReport:
    """Analyze a stored match record into a review report.

    Parameters
    ----------
    match : dict
        A ``MatchHistory.get`` record: top-level ``match_id`` / ``winner``
        / ``moves`` plus a ``meta`` sub-dict.

    Returns
    -------
    ReviewReport
        Key nodes, one improvement sentence, and a win/loss summary.
    """
    meta = match.get("meta") if isinstance(match.get("meta"), dict) else {}
    player_pid = str(match.get("player_pid") or meta.get("player_pid") or "")
    ai_pid_raw = match.get("ai_pid") or meta.get("ai_pid")
    ai_pid = str(ai_pid_raw) if ai_pid_raw is not None else None
    winner_raw = match.get("winner") or meta.get("winner")
    winner = str(winner_raw) if winner_raw else None
    game_id = str(match.get("game_id") or meta.get("game_id") or "")
    moves = match.get("moves")
    if not isinstance(moves, list):
        moves = []

    c2_evaluate = _try_import_c2_evaluate()
    scores = [_score_step(move, player_pid, ai_pid, c2_evaluate) for move in moves]

    turning_point = _find_turning_point(scores)
    winning_move = _find_winning_move(moves, winner, player_pid, ai_pid)
    blunder = _find_blunder(moves, scores, winner, player_pid, ai_pid)

    key_nodes = [node for node in (turning_point, winning_move, blunder) if node is not None]
    return ReviewReport(
        key_nodes=key_nodes,
        improvement=_improvement_text(blunder, turning_point),
        summary=_summary_text(game_id, winner, player_pid, len(moves)),
    )


def _try_import_c2_evaluate() -> Callable[..., Any] | None:
    """Return C2's ``evaluate`` when importable, else ``None``.

    C2 (``layer4_interface/agent/evaluation.py``) is developed in
    parallel; the runtime import keeps ``review`` free of a hard
    dependency on a module that may not exist yet.

    Returns
    -------
    Callable[..., Any] | None
        C2's evaluate function, or ``None`` when C2 is not ready.
    """
    try:
        from layer4_interface.agent.evaluation import evaluate
    except ImportError:
        return None
    return evaluate


def _score_step(move: dict, player_pid: str, ai_pid: str | None, c2: Callable[..., Any] | None) -> float:
    """Score one move's snapshot from the player's perspective."""
    snapshot = move.get("snapshot") if isinstance(move, dict) else None
    if not isinstance(snapshot, dict):
        snapshot = {}
    if c2 is not None:
        score = _try_c2_score(snapshot, player_pid, c2)
        if score is not None:
            return score
    return _local_score(snapshot, player_pid, ai_pid)


def _try_c2_score(snapshot: dict, viewer: str, c2: Callable[..., Any]) -> float | None:
    """Delegate to C2 evaluate when the snapshot embeds a live engine state.

    Serialized ``MatchHistory`` snapshots carry no ``state``/``engine``
    keys, so this path is only taken by richer callers; any failure falls
    back to the local proxy.
    """
    state = snapshot.get("state")
    engine = snapshot.get("engine")
    if not isinstance(state, dict) or engine is None:
        return None
    try:
        result = c2(state, viewer, engine)
    except Exception:
        # Optional, untrusted dependency — any failure must degrade to the
        # local proxy rather than aborting the review.
        return None
    if isinstance(result, dict):
        score = result.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            return float(score)
        return None
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        return float(result)
    return None


def _local_score(snapshot: dict, player_pid: str, ai_pid: str | None) -> float:
    """Score a snapshot via the generic terminal / board proxy.

    Terminal snapshots use the recorded ``payoff`` (player perspective)
    when present, else the ``winner`` vs ``player_pid`` sign (±1 / 0).
    Non-terminal snapshots fall back to the board neighborhood heuristic.
    """
    winner = snapshot.get("winner")
    over = snapshot.get("over")
    if over or (winner is not None and winner != ""):
        payoff = snapshot.get("payoff")
        if isinstance(payoff, (int, float)) and not isinstance(payoff, bool):
            return float(payoff)
        if winner == player_pid:
            return 1.0
        if winner is not None and winner != "":
            return -1.0
        return 0.0
    board = snapshot.get("board")
    if isinstance(board, list) and board:
        return _board_score(board, player_pid, ai_pid)
    return 0.0


def _board_score(board: list, player_pid: str, ai_pid: str | None) -> float:
    """Generic board heuristic: material + corners + center.

    Normalized to roughly ``[-1, 1]`` from the player's perspective.
    """
    n = len(board)
    size = int(n**0.5)
    if size * size != n or size < 1:
        return 0.0
    opponent = ai_pid if ai_pid not in (None, player_pid) else _opponent_of(board, player_pid)
    if opponent is None:
        return 0.0

    player_count = sum(1 for cell in board if cell == player_pid)
    opp_count = sum(1 for cell in board if cell == opponent)
    total = player_count + opp_count
    material = (player_count - opp_count) / total if total else 0.0

    corners = [0, size - 1, n - size, n - 1] if size >= 2 else [0]
    player_corners = sum(1 for i in corners if board[i] == player_pid)
    opp_corners = sum(1 for i in corners if board[i] == opponent)
    corner_term = (player_corners - opp_corners) * _CORNER_WEIGHT / len(corners)

    center = board[n // 2]
    if center == player_pid:
        center_term = _CENTER_WEIGHT
    elif center == opponent:
        center_term = -_CENTER_WEIGHT
    else:
        center_term = 0.0

    return _clamp(material + corner_term + center_term, -1.0, 1.0)


def _opponent_of(board: list, player_pid: str) -> str | None:
    """Return the first non-empty cell value that is not ``player_pid``."""
    for cell in board:
        if isinstance(cell, str) and cell not in ("", player_pid):
            return cell
    return None


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into ``[lo, hi]``."""
    return max(lo, min(hi, value))


def _find_turning_point(scores: list[float]) -> KeyNode | None:
    """Return the step with the largest absolute score jump."""
    n = len(scores)
    if n == 0:
        return None
    if n == 1:
        return KeyNode(step=0, kind="turning_point", why="唯一一手")
    best_step = 1
    best_delta = abs(scores[1] - scores[0])
    for i in range(2, n):
        delta = abs(scores[i] - scores[i - 1])
        if delta > best_delta:
            best_delta = delta
            best_step = i
    if best_delta <= 0.0:
        return KeyNode(step=0, kind="turning_point", why="评估值无跳变，取首步")
    return KeyNode(step=best_step, kind="turning_point", why="评估值跳变最大的一手")


def _actor_pid(actor: str, player_pid: str, ai_pid: str | None) -> str | None:
    """Map the stored ``actor`` caption to a concrete player id."""
    if actor == "human":
        return player_pid
    if actor == "ai":
        return ai_pid
    return actor or None


def _find_winning_move(
    moves: list[dict],
    winner: str | None,
    player_pid: str,
    ai_pid: str | None,
) -> KeyNode | None:
    """Return the winner's last move, or ``None`` when no winner is known."""
    if not winner:
        return None
    step: int | None = None
    for i, move in enumerate(moves):
        if not isinstance(move, dict):
            continue
        if _actor_pid(str(move.get("actor", "")), player_pid, ai_pid) == winner:
            step = i
    if step is None:
        return None
    return KeyNode(step=step, kind="winning_move", why="胜方奠定胜局的最后一手")


def _find_blunder(
    moves: list[dict],
    scores: list[float],
    winner: str | None,
    player_pid: str,
    ai_pid: str | None,
) -> KeyNode | None:
    """Return the player's own step with the largest evaluation drop.

    Only reported when the player ultimately lost and the single-step drop
    exceeds ``_BLUNDER_DROP``.
    """
    if winner is None or winner == player_pid:
        return None
    best_step: int | None = None
    best_drop = 0.0
    for i in range(1, len(scores)):
        if not isinstance(moves[i], dict):
            continue
        if _actor_pid(str(moves[i].get("actor", "")), player_pid, ai_pid) != player_pid:
            continue
        drop = scores[i] - scores[i - 1]
        if drop < -_BLUNDER_DROP and drop < best_drop:
            best_drop = drop
            best_step = i
    if best_step is None:
        return None
    return KeyNode(step=best_step, kind="blunder", why="己方评估值显著下降且最终落败的一手")


def _improvement_text(blunder: KeyNode | None, turning_point: KeyNode | None) -> str:
    """Return one mechanical improvement sentence (no hidden information)."""
    if blunder is not None:
        return f"第 {blunder.step + 1} 手后评估明显下滑，避免在那类局面冒险"
    if turning_point is not None:
        return f"第 {turning_point.step + 1} 手后优势发生转折，注意把握后续关键局面"
    return "继续巩固优势，稳扎稳打"


def _summary_text(game_id: str, winner: str | None, player_pid: str, move_count: int) -> str:
    """Return the win/loss + move-count summary."""
    name = _GAME_NAMES.get(game_id, game_id or "对局")
    if winner == player_pid:
        result = "玩家获胜"
    elif winner:
        result = "AI 获胜"
    else:
        result = "平局"
    return f"{name}，{result}，共 {move_count} 手"
