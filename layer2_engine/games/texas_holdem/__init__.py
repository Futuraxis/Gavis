"""Texas Hold'em — heads-up no-limit, blinds 1/2, stacks 100 (v5.0).

Game logic lives entirely in ``rules/texas_holdem.json`` (effectors,
chance dealing, visibility).  This adapter adds:
  1. Structured observations (``get_observation``) for UI/RL consumers
  2. ``resolve_chance`` — advance through all pending chance nodes
"""

from __future__ import annotations

from .texas_env_adapter import TexasHoldemAdapter

__all__ = ["TexasHoldemAdapter"]
