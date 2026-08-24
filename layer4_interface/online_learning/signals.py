"""Signal conversion for online learning (Layer 4).

Builds :class:`OnlineLearningSignal` instances from the recorded store
data — one signal per finished match.  The signal is the stable input
shaped for downstream consumers (empirical opponent table, later PPO
replay ingestion, werewolf LLM feedback), matching the data structure
declared in ``feedback_collector.py``.
"""

from __future__ import annotations

from typing import Any

from .feedback_collector import OnlineLearningSignal

WIN = 1.0
DRAW = 0.0
LOSS = -1.0


def outcome_for(human_pid: str, winner: str | None) -> float:
    """Map a match winner to the human-side outcome (+1 / 0 / -1)."""
    if winner is None:
        return DRAW
    return WIN if winner == human_pid else LOSS


def signal_from_match(
    game_id: str,
    solver_name: str,
    match: dict[str, Any],
) -> OnlineLearningSignal:
    """Convert one store match block (``terminal`` + ``decisions``) into a signal.

    ``solver_name`` is the solver the human played against (e.g.
    ``"hybrid"``); the signal itself carries the raw adapter-level
    decisions, so any consumer that understands the game can rebuild its
    own view (info-set tables, feature trajectories, ...).
    """
    terminal: dict = match.get("terminal", {})
    decisions: list[dict] = match.get("decisions", [])
    human_pid: str = terminal.get("human_pid") or (decisions[0].get("player") if decisions else "")
    solver_suggestions: list[dict | None] = []
    for decision in decisions:
        # The AI's own decisions ARE the recorded suggestions; a human
        # decision has no recorded suggestion yet (future: capture the
        # AI's top pick at the human's turns).
        solver_suggestions.append(decision.get("action") if decision.get("actor") == "ai" else None)
    return OnlineLearningSignal(
        game_id=game_id,
        solver_name=solver_name,
        controlled_player=human_pid,
        state_sequence=[d.get("state", {}) for d in decisions],
        actions_taken=[d.get("action", {}) for d in decisions],
        solver_suggestions=solver_suggestions,
        final_outcome=outcome_for(human_pid, terminal.get("winner")),
        user_rating=None,
        metadata={
            "match_id": terminal.get("match_id"),
            "winner": terminal.get("winner"),
            "utilities": terminal.get("utilities", {}),
            "difficulty": terminal.get("difficulty"),
            "started_at": terminal.get("started_at"),
            "finished_at": terminal.get("finished_at"),
            "decisions": decisions,
            "legal_sets": [d.get("legal", []) for d in decisions],
            "info_keys": [d.get("info_key") for d in decisions],
        },
    )
