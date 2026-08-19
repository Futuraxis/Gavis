"""SolverAdapter — the single contract between Layer 2 and Layer 3.

Every solver (MCTS, CFR, PPO, PSRO) consumes the game exclusively through
this Protocol.  ``GameEngine`` is the canonical implementation, but any
object satisfying the Protocol can be used for testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

# ── Core data types ──────────────────────────────────────────────

NodeType = Literal["player", "chance", "terminal"]
"""Type of a game node."""


State = dict[str, Any]
"""Game state — a generic dict with ground arrays + env scalars.

Ground arrays are stored under ``_arrays``, environment scalars under ``env``.
Derived views are computed on-the-fly by the engine.
"""


@dataclass
class ActionInstance:
    """A concrete action generated from an action template at runtime."""

    template_id: str
    type: str
    actor_id: str
    params: dict
    canonical_key: str


@dataclass
class ChanceOutcome:
    """A single outcome of a chance node."""

    key: str
    probability: float
    effect_ref: str
    canonical_key: str


Obs = dict[str, Any]
"""An observation returned by ``get_observation()``.

For perfect-information games this includes materialized derived views;
for imperfect-information games it is the player's partial view after
visibility projection.
"""

# ── Protocol ─────────────────────────────────────────────────────


@runtime_checkable
class SolverAdapter(Protocol):
    """The single interface between Engine (Layer 2) and Solvers (Layer 3).

    Every game is presented to solvers through this lens.  ``GameEngine``
    is the reference implementation.
    """

    def create_initial_state(self) -> State:
        """Return a fresh game state.  Called once per episode."""
        ...

    def get_node_type(self, state: State) -> NodeType:
        """Return 'player', 'chance', or 'terminal' for the given state."""
        ...

    def get_current_player(self, state: State) -> str | None:
        """Return the player whose turn it is, or None if not a player node."""
        ...

    def get_legal_actions(self, state: State) -> list[ActionInstance]:
        """Return all legal actions from this state."""
        ...

    def apply_action(self, state: State, action: ActionInstance) -> State:
        """Return a *new* state after applying the given action."""
        ...

    def get_chance_outcomes(self, state: State) -> list[ChanceOutcome]:
        """Return all possible chance outcomes with their probabilities."""
        ...

    def apply_chance(self, state: State, outcome: ChanceOutcome) -> State:
        """Return a new state after applying the chance outcome."""
        ...

    def is_terminal(self, state: State) -> bool:
        """Return True if the state is terminal."""
        ...

    def get_utility(self, state: State, player: str) -> float:
        """Return the utility (payoff) for ``player`` at this state."""
        ...

    # ── RL / online-learning extensions ────────────────────────

    def get_observation(self, state: State, player: str) -> Obs:
        """Return the observation for ``player`` (used by PPO, etc.)."""
        ...

    def get_info_set_key(self, state: State, player: str) -> str:
        """Return a canonical info-set key (used by CFR)."""
        ...

    # ── Visibility projection (v5.0) ──────────────────────────

    def project_observation(self, state: State, viewer: str) -> Obs:
        """Return the state as seen by ``viewer`` after visibility rules.

        For perfect-information games this is the full state; for
        imperfect-information games it includes only the visible
        fields per the visibility rules declared in the JSON.
        """
        ...

    # ── Interface Layer extension ─────────────────────────────

    def load_state(self, state: State) -> State:
        """Import an externally-constructed state (e.g. from VLM).

        The default implementation simply returns the state; subclasses
        may validate or transform it.
        """
        return state
