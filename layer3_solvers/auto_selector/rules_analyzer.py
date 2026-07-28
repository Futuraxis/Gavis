"""Rule analyzer — extracts game characteristics from rules.json.

Used by the auto-selector to recommend a solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GameProfile:
    """Profile of a game extracted from its rules.json."""
    name: str = "unknown"
    board_size: int = 0
    has_chance_nodes: bool = False
    has_hidden_info: bool = False
    state_space_estimate: int = 0
    action_space_estimate: int = 0
    suggested_solver: str = "mcts"


def analyze_game(rules: dict) -> GameProfile:
    """Analyze a rules.json and return a GameProfile.

    Currently returns a best-guess based on board size.
    Full implementation is future work.
    """
    profile = GameProfile()
    constants = rules.get('constants', {})
    profile.board_size = constants.get('board_size', 0)
    profile.has_chance_nodes = len(rules.get('chance', [])) > 0
    profile.has_hidden_info = False  # not yet detectable

    # Estimate state space
    if profile.board_size > 0:
        cells = profile.board_size ** 2
        profile.state_space_estimate = 3 ** cells
        profile.action_space_estimate = cells

    # Suggest solver
    profile.suggested_solver = suggest_solver(profile)

    return profile


def suggest_solver(profile: GameProfile) -> str:
    """Suggest a solver based on game profile.

    Rules of thumb:
      - State space ≤ 10⁶ → PSRO (tabular is feasible)
      - Has chance nodes → MCTS (natural chance handling)
      - State space > 10⁶ → PPO (neural net generalization)
      - Small board + need equilibrium → CFR
    """
    if profile.state_space_estimate <= 20000:
        return "psro"
    if profile.has_chance_nodes:
        return "mcts"
    if profile.state_space_estimate <= 10_000_000:
        return "cfr"
    return "ppo"
