"""SolverProvider — Layer 4's contract for solver assembly (dependency injection).

Architecture §12.1: Layer 4 never imports Layer 3.  The play apps and
the platform consume solvers through the minimal :class:`SolverHandle`
protocol, and receive a :class:`SolverProvider` at assembly time from
the application layer (``train-cli/games.py``, via the ``train_cli``
import bridge).  Only that concrete provider is allowed to import
``layer3_solvers`` on behalf of
the frontend — grep for ``layer3_solvers`` under ``layer4_interface/``
finds nothing by construction.

The provider is injected by the server entry points (``main()`` in each
``server.py``), so ``python -m layer4_interface.frontend.play_*.server``
keeps working without any Layer-3 reference in Layer 4.
"""

from __future__ import annotations

from typing import Any, Protocol


class SolverHandle(Protocol):
    """Minimal solver surface consumed by Layer 4 (SolverBase subset).

    The play sessions and the benchmark runner only call
    ``select_action`` / ``solve`` / ``train`` and read ``name``; the
    full SolverBase contract stays inside Layer 3.
    """

    @property
    def name(self) -> str: ...

    def select_action(self, state: dict[str, Any]) -> Any | None: ...

    def solve(self, state: dict[str, Any], **kwargs: Any) -> Any | None: ...

    def train(self, episodes: int, **kwargs: Any) -> Any: ...


class SolverProvider(Protocol):
    """Instantiates solver handles; implemented in the app layer.

    ``game_id`` lets the implementation pick game-specific behavior
    (e.g. imperfect-information search for Texas Hold'em, CFR exclusion
    for the same game); ``budget`` is the per-difficulty search budget
    already resolved by the caller.
    """

    def create_solver(
        self,
        game_id: str,
        name: str,
        engine: Any,
        seed: int,
        budget: int,
        **kwargs: Any,
    ) -> SolverHandle: ...
