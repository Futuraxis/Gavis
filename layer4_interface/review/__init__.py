"""Layer 4: Review — post-match analysis backend (C4).

Exposes the frozen C4 contract: :class:`KeyNode`, :class:`ReviewReport`
and :func:`analyze`.
"""

from __future__ import annotations

from .analyzer import KeyNode, ReviewReport, analyze

__all__ = ["KeyNode", "ReviewReport", "analyze"]
