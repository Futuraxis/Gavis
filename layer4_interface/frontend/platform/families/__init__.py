"""Custom-game rule-family registry (auto-discovery).

Each submodule in this package (``helpers.py`` is shared utilities and
has no family) that exposes a ``FAMILY_ID`` string, a ``detect(rules)``
predicate and a ``build_spec(game_id, rules)`` factory is a family.
Discovery is automatic via ``pkgutil`` — adding a family is just adding
one module, with no central registry to edit:

- ``grid`` — N×N grid placement/alignment games (moon chess, gomoku,
  connect-4, …)
- later waves: ``poker`` / ``mahjong`` / ``social``

The detected family is what turns a validated *rules* dict into a
platform ``GameSpec`` (``layer4_interface/frontend/platform/games.py``),
so custom games get playable sessions with zero per-game adapter code.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any, Protocol

from ..games import GameSpec

logger = logging.getLogger(__name__)


class FamilyModule(Protocol):
    """The module protocol every discovered family submodule implements."""

    FAMILY_ID: str

    def detect(self, rules: dict) -> bool:
        """Whether ``rules`` belongs to this family."""
        ...

    def build_spec(self, game_id: str, rules: dict) -> GameSpec:
        """Build the platform ``GameSpec`` for a validated rules dict."""
        ...


def _discover() -> dict[str, Any]:
    """Import family submodules; returns ``{FAMILY_ID: module}`` (sorted)."""
    found: dict[str, Any] = {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda item: item.name):
        if info.name == "helpers":
            continue
        try:
            module = importlib.import_module(f"{__name__}.{info.name}")
        except Exception as exc:  # noqa: BLE001 — a broken family must not break the platform
            logger.warning("自定义游戏族加载失败: %s: %s", info.name, exc)
            continue
        family_id = getattr(module, "FAMILY_ID", None)
        if isinstance(family_id, str) and family_id:
            found[family_id] = module
    return found


_FAMILIES: dict[str, Any] = _discover()

#: Registered family ids (auto-discovered, sorted).
FAMILY_IDS: tuple[str, ...] = tuple(_FAMILIES)


def detect_family(rules: dict) -> FamilyModule | None:
    """Return the first family module whose ``detect(rules)`` is True.

    Args:
        rules: A rules JSON dict (already schema/engine validated).

    Returns:
        The matching ``FamilyModule`` or ``None`` when no family claims
        the rules (the game is not supported by the platform).
    """
    for module in _FAMILIES.values():
        if module.detect(rules):
            return module
    return None


__all__ = ["FamilyModule", "FAMILY_IDS", "detect_family"]
