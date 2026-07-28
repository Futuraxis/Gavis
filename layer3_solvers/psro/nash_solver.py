"""Nash equilibrium solver via linear programming.

Computes a Nash equilibrium of a zero-sum two-player game
from a payoff matrix using ``scipy.optimize.linprog``.
"""

from __future__ import annotations

import numpy as np


def solve_nash(R_matrix: np.ndarray) -> np.ndarray:
    """Solve for the Nash equilibrium of a zero-sum game.

    Parameters
    ----------
    R_matrix : np.ndarray, shape (N, N)
        Payoff matrix for player 1 (row player).

    Returns
    -------
    np.ndarray, shape (N,)
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

    N = R_matrix.shape[0]

    # We solve the column player's LP and convert:
    # min  -v  s.t.  -R^T x + v ≤ 0,  sum(x) = 1,  x ≥ 0
    # Equivalent to: min c^T [x; v]  s.t.  A_ub [x; v] ≤ b_ub, A_eq [x; v] = b_eq
    #
    # Variables: [x_0, x_1, ..., x_{N-1}, v]

    c = np.zeros(N + 1)
    c[-1] = -1.0  # minimize -v

    # A_ub: -R^T x + v ≤ 0  →  -R^T x + v ≤ 0
    # (for each column j: -sum_i R_ij * x_i + v ≤ 0)
    A_ub = np.zeros((N, N + 1))
    A_ub[:, :N] = -R_matrix.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(N)

    # A_eq: sum(x) = 1
    A_eq = np.zeros((1, N + 1))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])

    # bounds: x_i ∈ [0, 1], v free
    bounds = [(0, 1)] * N + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not result.success:
        # Fallback: uniform distribution
        return np.ones(N) / N

    nash = result.x[:N]
    nash = np.maximum(nash, 0)  # clamp numerical negatives
    nash /= nash.sum()  # re-normalize
    return nash
