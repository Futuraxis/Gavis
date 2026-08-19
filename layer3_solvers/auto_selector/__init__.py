"""Auto-Selector — automatically chooses the best solver for a game.

This module is a placeholder.  Future work will implement rule analysis
to select between MCTS, CFR, PPO, and PSRO based on game characteristics.
"""

from .rules_analyzer import analyze_game, suggest_solver

__all__ = ["analyze_game", "suggest_solver"]
