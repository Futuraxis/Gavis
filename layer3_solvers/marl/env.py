"""MARLEnv — shared multi-agent episode runner for the MARL solvers.

``run_episode`` drives one full episode through the ``SolverAdapter``:
chance nodes are resolved automatically (weighted by probability), and
every decision made by the acting player is recorded as a
``Transition``.  Solvers only supply a per-decision policy callback;
QMix / HAPPO / MAAC all share this runner.

--- Reward design ---

Rewards are terminal-only: intermediate transitions get ``0.0``; at the
terminal state ``get_utility(state, p)`` is assigned to **each player's
own last transition** (marked ``done=True``).  In turn-based games a
player's final decision is often not the game-ending decision, and
without this the terminal utility would never enter that player's
returns.  Earlier transitions bootstrap normally.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from layer2_engine.interfaces.solver_adapter import (
    SolverAdapter,
    State,
)

from .action_space import ActionSpace
from .encoders import GameEncoder

# ``select_idx`` returns the chosen action index plus an info dict
# (HAPPO fills ``log_prob`` / ``value`` / ``next_value``; others ignore it).
SelectFn = Callable[[int, State, np.ndarray], tuple[int, dict]]
# ``NextValueFn`` evaluates the successor state for the acting player
# (HAPPO's critic; None for QMix / MAAC).
NextValueFn = Callable[[State, int], float]


@dataclass(slots=True)
class Transition:
    """One decision of one player during an episode."""

    player_idx: int
    obs: np.ndarray  # acting player's observation vector
    mask: np.ndarray  # float32 legal-action mask over the action space
    action: int  # chosen action index
    log_prob: float = 0.0  # HAPPO: policy log-prob of ``action``
    value: float = 0.0  # HAPPO: critic value at decision time
    next_value: float = 0.0  # HAPPO: critic value after the action
    reward: float = 0.0  # 0.0 unless terminal (payoff of the acting player)
    done: bool = False  # True on the acting player's final transition
    global_state: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    next_obs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    next_global_state: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    next_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


@dataclass(slots=True)
class EpisodeTrajectory:
    """The full record of one episode."""

    transitions: list[Transition] = field(default_factory=list)
    payoffs: dict[str, float] = field(default_factory=dict)


def resolve_players(adapter: SolverAdapter) -> list[str]:
    """Agent ids in fixed order (``rules['players']``, or env scalars)."""
    rules = getattr(adapter, "rules", {})
    players = rules.get("players") if isinstance(rules, dict) else None
    if players:
        return [str(p) for p in players]
    state = adapter.create_initial_state()
    env = state.get("env", {})
    players = env.get("player_ids")
    if players:
        return [str(p) for p in players]
    return ["p_black", "p_white"]


def resolve_device(device: str | None) -> torch.device:
    """Resolve 'cpu' / 'cuda' (falls back to cpu when unavailable)."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def weighted_choice(outcomes: list, rng: random.Random):
    """Sample a chance outcome by its probability (uniform fallback)."""
    probs = [float(getattr(o, "probability", 0.0) or 0.0) for o in outcomes]
    if sum(probs) <= 0:
        return rng.choice(outcomes)
    return rng.choices(outcomes, weights=probs, k=1)[0]


def run_episode(
    adapter: SolverAdapter,
    players: list[str],
    rng: random.Random,
    encoder: GameEncoder,
    action_space: ActionSpace,
    select_idx: SelectFn,
    next_value_fn: NextValueFn | None = None,
    max_steps: int = 0,
) -> EpisodeTrajectory:
    """Play one episode, returning the trajectory and terminal payoffs.

    Parameters
    ----------
    adapter : SolverAdapter
    players : list[str]
        Agent ids in fixed order.
    rng : random.Random
        Seeded RNG for chance outcomes.
    encoder : GameEncoder
        Observation / joint-state encoder.
    action_space : ActionSpace
        Fixed action space.
    select_idx : SelectFn
        ``select_idx(player_idx, state, mask) -> (idx, info)``.
    next_value_fn : NextValueFn, optional
        Evaluates the successor state for the acting player; the result
        lands in ``info['next_value']`` (HAPPO critic; unused elsewhere).
    max_steps : int
        Step guard (0 = unlimited) against pathological loops.
    """
    state = adapter.create_initial_state()
    player_idx = {p: i for i, p in enumerate(players)}
    traj = EpisodeTrajectory()
    last_by_player: dict[int, Transition] = {}
    steps = 0

    while True:
        if max_steps and steps >= max_steps:
            break
        steps += 1
        node = adapter.get_node_type(state)
        if node == "chance":
            outcomes = adapter.get_chance_outcomes(state)
            if not outcomes:
                break
            state = adapter.apply_chance(state, weighted_choice(outcomes, rng))
            continue
        if node != "player":
            break
        current = adapter.get_current_player(state)
        if current is None or current not in player_idx:
            break
        pid = player_idx[current]
        legal = adapter.get_legal_actions(state)
        if not legal:
            break
        mask = action_space.legal_mask(state, legal)
        action_idx, info = select_idx(pid, state, mask)
        action = action_space.action_from_index(action_idx, legal)
        if action is None:
            break
        try:
            next_state = adapter.apply_action(state, action)
        except Exception:
            # Engine/rules edge case (e.g. mahjong degenerate chi chains
            # crashing payoff evaluation): end the episode as a draw.
            traj.payoffs = {p: 0.0 for p in players}
            break

        done = adapter.is_terminal(next_state)
        if not done and next_value_fn is not None:
            info["next_value"] = float(next_value_fn(next_state, pid))
        reward = 0.0
        if done:
            reward = float(adapter.get_utility(next_state, current))
            for p in players:
                traj.payoffs[p] = float(adapter.get_utility(next_state, p))

        transition = Transition(
            player_idx=pid,
            obs=encoder.encode_obs(state, current).astype(np.float32),
            mask=mask,
            action=action_idx,
            log_prob=float(info.get("log_prob", 0.0)),
            value=float(info.get("value", 0.0)),
            next_value=float(info.get("next_value", 0.0)),
            reward=reward,
            done=done,
            global_state=encoder.encode_global(state).astype(np.float32),
            next_obs=encoder.encode_obs(next_state, current).astype(np.float32),
            next_global_state=encoder.encode_global(next_state).astype(np.float32),
            next_mask=action_space.legal_mask(next_state),
        )
        traj.transitions.append(transition)
        last_by_player[pid] = transition
        state = next_state
        if done:
            break

    # Terminal utility only reaches each player's own final decision.
    if adapter.is_terminal(state):
        for p in players:
            payoff = float(adapter.get_utility(state, p))
            traj.payoffs[p] = payoff
            last = last_by_player.get(player_idx[p])
            if last is not None:
                last.reward = payoff
                last.done = True
    return traj
