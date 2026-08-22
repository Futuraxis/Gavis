"""Nash equilibrium solver via linear programming.

Computes a Nash equilibrium of a zero-sum two-player game
from a payoff matrix using ``scipy.optimize.linprog``.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def solve_nash(reward_matrix: np.ndarray) -> np.ndarray:
    """Solve for the Nash equilibrium of a zero-sum game.

    Parameters
    ----------
    reward_matrix : np.ndarray, shape (strategy_count, strategy_count)
        Payoff matrix for player 1 (row player).

    Returns
    -------
    np.ndarray, shape (strategy_count,)
        Nash equilibrium mixed strategy for player 1 (row player).

    Notes
    -----
    Solves the linear program::

        max  v
        s.t. R^T · x ≥ v · 1
             sum(x) = 1
             x ≥ 0

    where x is the row player's mixed strategy.
    """
    from scipy.optimize import linprog

    strategy_count = reward_matrix.shape[0]

    # We solve the column player's LP and convert:
    # min  -v  s.t.  -R^T x + v ≤ 0,  sum(x) = 1,  x ≥ 0
    # Equivalent to: min c^T [x; v]  s.t.  a_ub [x; v] ≤ b_ub, a_eq [x; v] = b_eq
    #
    # Variables: [x_0, x_1, ..., x_{strategy_count-1}, v]

    c = np.zeros(strategy_count + 1)
    c[-1] = -1.0  # minimize -v

    # a_ub: -R^T x + v ≤ 0  →  -R^T x + v ≤ 0
    # (for each column j: -sum_i R_ij * x_i + v ≤ 0)
    a_ub = np.zeros((strategy_count, strategy_count + 1))
    a_ub[:, :strategy_count] = -reward_matrix.T
    a_ub[:, -1] = 1.0
    b_ub = np.zeros(strategy_count)

    # a_eq: sum(x) = 1
    a_eq = np.zeros((1, strategy_count + 1))
    a_eq[0, :strategy_count] = 1.0
    b_eq = np.array([1.0])

    # bounds: x_i ∈ [0, 1], v free
    bounds = [(0, 1)] * strategy_count + [(None, None)]

    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not result.success:
        # Fallback: uniform distribution — 显式告警而不是无痕降级
        logger.warning("solve_nash: LP failed (%s); falling back to uniform", result.message)
        return np.ones(strategy_count) / strategy_count

    nash = result.x[:strategy_count]
    nash = np.maximum(nash, 0)  # clamp numerical negatives
    nash /= nash.sum()  # re-normalize
    return nash
