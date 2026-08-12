"""Werewolf solvers — Bayesian belief tracking + decision for Werewolf.

``BeliefTracker`` maintains each player's posterior role distribution from
the common prior (known role pool) and the public history (speeches, votes,
deaths); ``BayesSolver`` turns that posterior into decisions (vote the most
suspicious, night skills by expected utility / information gain).
"""

from .bayes_solver import BayesConfig, BayesSolver
from .belief import BeliefTracker

__all__ = ["BeliefTracker", "BayesSolver", "BayesConfig"]
