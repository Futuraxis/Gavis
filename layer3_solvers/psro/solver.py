"""PSRO Solver — Policy-Space Response Oracles.

Implements the PSRO algorithm: maintain a pool of policies,
compute the meta-game payoff matrix, solve for Nash equilibrium,
and iteratively add best responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from layer2_engine.interfaces.solver_adapter import (
    ActionInstance,
    SolverAdapter,
    State,
)

from ..base import SolverBase, SolverConfig, SolverMetrics
from .agent import Agent
from .gym_adapter import GymAdapter
from .meta_game import exploitability, gamescape
from .nash_solver import solve_nash
from .tabular_q import tabular_q_best_response


@dataclass
class PSROConfig(SolverConfig):
    num_iters: int = 20
    num_steps_per_iter: int = 5000
    epsilon: float = 0.1
    alpha: float = 0.1
    evaluation_episodes: int = 10
    num_workers: int = 0  # 元博弈评估并行线程数（0=自动，1=串行）


class PSROSolver(SolverBase):
    """PSRO (Policy-Space Response Oracles) solver.

    Maintains a pool of policies and iteratively expands it towards
    a Nash equilibrium of the meta-game.
    """

    def __init__(self, adapter: SolverAdapter, config: SolverConfig | None = None):
        super().__init__(adapter, config or PSROConfig())
        self._gym = GymAdapter(adapter)

        # Get state/action dimensions from the gym adapter
        obs_dim = self._gym.observation_space.n
        n_actions = self._gym.action_space.n

        # Initialize with one random policy
        tmp = np.random.rand(obs_dim, n_actions)
        pi = np.eye(n_actions)[tmp.argmax(-1)]
        self._policy_pool: list[np.ndarray] = [pi]
        self._nash_mixture: np.ndarray | None = None
        self._nash_weights: np.ndarray | None = None  # per-member mixture weights
        self._payoff_matrix: np.ndarray | None = None
        self._expl_history: list[float] = []
        self._div_history: list[float] = []

    @property
    def name(self) -> str:
        return f"PSRO(it={getattr(self.config, 'num_iters', 20)})"

    def select_action(self, state: State) -> Optional[ActionInstance]:
        """Select action using the Nash mixture policy."""
        if self._nash_mixture is None:
            legal = self.adapter.get_legal_actions(state)
            return legal[0] if legal else None

        # Encode state
        obs = self._gym._encode_state(state)
        agent = Agent(self._nash_mixture)
        mask = self._gym.available_actions()
        action_idx = agent.step(obs, amask=mask)

        # Convert back to ActionInstance
        legal = self.adapter.get_legal_actions(state)
        return self._gym._int_to_action(action_idx, legal)

    def train(self, episodes: int = 20, **kwargs) -> SolverMetrics:
        """Run PSRO training.

        Parameters
        ----------
        episodes : int
            Number of PSRO iterations (not episodes — PSRO is not episode-based).
        """
        num_iters = episodes if episodes > 0 else getattr(self.config, "num_iters", 20)
        num_steps = getattr(self.config, "num_steps_per_iter", 5000)
        eps = getattr(self.config, "epsilon", 0.1)
        alpha = getattr(self.config, "alpha", 0.1)
        n_eval = getattr(self.config, "evaluation_episodes", 10)
        n_workers = getattr(self.config, "num_workers", 0)

        verbose = kwargs.get("verbose", False)

        for niter in range(1, num_iters + 1):
            # Expand the cached gamescape with only the newly added policies.
            payoff_matrix = gamescape(
                self._gym,
                self._policy_pool,
                Ne=n_eval,
                previous=self._payoff_matrix,
                num_workers=n_workers,
            )
            self._payoff_matrix = payoff_matrix

            # Solve for Nash
            nash_p = solve_nash(payoff_matrix)
            self._nash_mixture = self._build_nash_mixture(nash_p)
            self._nash_weights = nash_p

            # Compute exploitability
            expl = exploitability(self._gym, self._nash_mixture, self._policy_pool, Ne=n_eval, num_workers=n_workers)

            # Train new best response
            beta = tabular_q_best_response(
                self._gym,
                num_steps=num_steps,
                epsilon=eps,
                alpha=alpha,
                opponent_policy=self._nash_mixture,
            )

            # Check for duplicate policy (tolerance-aware — exact float
            # equality is brittle across platforms/BLAS variants).
            is_duplicate = any(np.allclose(p, beta) for p in self._policy_pool)
            if is_duplicate:
                if verbose:
                    print(f"  PSRO iter {niter}: strategy exhausted, stopping")
                break

            self._policy_pool.append(beta)
            self._expl_history.append(expl)

            if verbose:
                print(
                    f"  PSRO iter {niter:3d}/{num_iters}  "
                    f"expl={expl:.4f}  "
                    f"pool={len(self._policy_pool)}  "
                    f"nash_w={nash_p[0]:.3f}"
                )

        return SolverMetrics(
            episodes=num_iters,
            win_rate=0.0,
            avg_return=0.0,
            extra={
                "pool_size": len(self._policy_pool),
                "final_exploitability": self._expl_history[-1] if self._expl_history else 0.0,
                "num_policies": len(self._policy_pool),
            },
        )

    def save(self, path: str) -> None:
        """Save PSRO state (policy pool + Nash mixture + member weights).

        Policies are stacked into a plain 3-D array — no ``dtype=object``
        pickling, so the file can be loaded with ``allow_pickle=False``.
        """
        weights = self._nash_weights
        pool = np.stack(self._policy_pool) if self._policy_pool else np.zeros((0, 0))
        np.savez_compressed(
            path,
            policy_pool=pool,
            nash_mixture=self._nash_mixture if self._nash_mixture is not None else np.zeros(0),
            nash_weights=weights if weights is not None else np.zeros(0),
            payoff_matrix=(self._payoff_matrix if self._payoff_matrix is not None else np.zeros((0, 0))),
            expl_history=np.array(self._expl_history),
        )

    def load(self, path: str) -> None:
        """Load PSRO state (no pickle — ``allow_pickle`` stays off)."""
        data = np.load(path, allow_pickle=False)
        self._policy_pool = [np.asarray(p) for p in data["policy_pool"]]
        mixture = data["nash_mixture"]
        self._nash_mixture = mixture if mixture.size else None
        weights = data["nash_weights"]
        self._nash_weights = weights if weights.size else None
        if "payoff_matrix" in data.files:
            payoff_matrix = data["payoff_matrix"]
            self._payoff_matrix = payoff_matrix if payoff_matrix.size else None
        else:
            # Compatibility with model files saved before payoff caching.
            self._payoff_matrix = None

    def _build_nash_mixture(self, weights: np.ndarray) -> np.ndarray:
        """Build the Nash mixture policy from weighted pool."""
        mixture = np.zeros_like(self._policy_pool[0])
        for w, pi in zip(weights, self._policy_pool):
            mixture += w * pi
        return mixture
