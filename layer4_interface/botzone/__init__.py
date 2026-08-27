"""Botzone adapter entrypoints.

This package is Layer 4: it translates Botzone's stdin/stdout lifecycle
into one Gavis solver decision.  Game execution remains in Layer 2 and
solver assembly is injected from the application registry.
"""

from .runner import BotzoneError, decide, main
from .texas_holdem import decide_texas_holdem, is_texas_holdem_payload

__all__ = ["BotzoneError", "decide", "main", "decide_texas_holdem", "is_texas_holdem_payload"]
