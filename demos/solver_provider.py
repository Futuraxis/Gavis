"""Default SolverProvider assembly for the frontend apps (app layer).

This is the ONLY module allowed to import ``layer3_solvers`` on behalf
of the Layer 4 frontend: it implements every solver the play apps and
the platform benchmark need (plus the random baseline) behind the
``layer4_interface.solver_provider`` protocol, and the server entry
points inject it at ``main()`` time.

``create_solver(game_id, name, engine, seed, budget, **kwargs)`` keeps
the flat call shape the frontend used historically; ``default_provider``
is the shared instance passed to ``PlayManager`` /
``BenchmarkRunner`` by each ``server.py``.
"""

from __future__ import annotations

import random
from typing import Any

from layer2_engine.interfaces.solver_adapter import ActionInstance, SolverAdapter
from layer3_solvers import (
    CFR,
    MCTS,
    CFRConfig,
    HybridConfig,
    HybridSolver,
    MCTSConfig,
    OllamaConfig,
    OllamaSolver,
    SolverConfig,
)
from layer3_solvers.base import SolverBase, SolverMetrics
from layer3_solvers.mahjong.heuristic import MahjongHeuristicAI


class RandomSolver(SolverBase):
    """Uniform random policy — the benchmark baseline."""

    def __init__(self, adapter: SolverAdapter, seed: int | None = None) -> None:
        super().__init__(adapter, SolverConfig(seed=seed))
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random"

    def select_action(self, state) -> ActionInstance | None:
        legal = self.adapter.get_legal_actions(state)
        return self._rng.choice(legal) if legal else None

    def train(self, episodes: int, **kwargs: Any) -> SolverMetrics:
        return SolverMetrics(episodes=episodes)


class DefaultSolverProvider:
    """Implements :class:`layer4_interface.solver_provider.SolverProvider`."""

    def create_solver(
        self,
        game_id: str,
        name: str,
        engine: SolverAdapter,
        seed: int,
        budget: int,
        **kwargs: Any,
    ) -> Any:
        """Instantiate a solver by name; raises ValueError on mismatch."""
        if name == "mcts":
            return MCTS(engine, MCTSConfig(seed=seed, budget=budget))
        if name == "cfr":
            if game_id == "texas_holdem":
                raise ValueError("CFR 不适用于德州扑克（不完全信息）")
            return CFR(engine, CFRConfig(seed=seed, iterations=1000, depth_limit=8))
        if name == "hybrid":
            return HybridSolver(
                engine,
                HybridConfig(
                    seed=seed,
                    mode="search",
                    imperfect_information=(game_id == "texas_holdem"),
                    mcts_budget=budget,
                    opponent_model="uniform",
                ),
            )
        if name == "random":
            return RandomSolver(engine, seed)
        if name == "mahjong":
            return MahjongHeuristicAI(engine, SolverConfig(seed=seed))
        if name == "ollama":
            return OllamaSolver(engine, OllamaConfig(model=kwargs["model"]), player_id=kwargs["player_id"])
        raise ValueError(f"未知求解器: {name}")


def create_solver(game_id: str, name: str, engine: SolverAdapter, seed: int, budget: int, **kwargs: Any) -> Any:
    """Module-level convenience (default provider) — mirrors the old API."""
    return default_provider.create_solver(game_id, name, engine, seed, budget, **kwargs)


default_provider = DefaultSolverProvider()
