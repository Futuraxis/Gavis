"""MARLEnv — shared multi-agent episode runner for the MARL solvers.

``run_episode`` drives one full episode through the ``GameEngine``:
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
returns.  Earlier transitions bootstrap normally.  (Design trade-off:
the payoff lands undiscounted on the player's last decision even though
the opponent may act afterwards — a mild overestimate at γ=0.99; revisit
if γ is lowered.)

Episodes that end abnormally (exception, ``max_steps``, dead chance/player
node) mark every recorded transition ``done=True`` with zero payoffs, so
HAPPO/MAAC never bootstrap a truncated episode as if it continued.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import State

from .action_space import ActionSpace
from .encoders import GameEncoder

logger = logging.getLogger(__name__)

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


def resolve_players(engine: GameEngine) -> list[str]:
    """Agent ids in fixed order (``rules['players']``, or env scalars)."""
    rules = getattr(engine, "rules", {})
    players = rules.get("players") if isinstance(rules, dict) else None
    if players:
        return [str(p) for p in players]
    state = engine.create_initial_state()
    env = state.get("env", {})
    players = env.get("player_ids")
    if players:
        return [str(p) for p in players]
    logger.warning(
        "resolve_players: no rules['players'] / env.player_ids — falling back to "
        "['p_black', 'p_white']; a game with different player ids will misalign agents"
    )
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
    engine: GameEngine,
    players: list[str],
    rng: random.Random,
    encoder: GameEncoder,
    action_space: ActionSpace,
    select_idx: SelectFn | dict[int, SelectFn],
    next_value_fn: NextValueFn | None | dict[int, NextValueFn] | None = None,
    max_steps: int = 0,
) -> EpisodeTrajectory:
    """Play one episode, returning the trajectory and terminal payoffs.

    Parameters
    ----------
    engine : GameEngine
    players : list[str]
        Agent ids in fixed order.
    rng : random.Random
        Seeded RNG for chance outcomes.
    encoder : GameEncoder
        Observation / joint-state encoder.
    action_space : ActionSpace
        Fixed action space.
    select_idx : SelectFn | dict[int, SelectFn]
        ``select_idx(player_idx, state, mask) -> (idx, info)``; a dict maps
        each acting seat to its own policy callback (对手编排：不同座位用
        不同策略 —— 学习器用当前策略、对手座位用冻结快照）.
    next_value_fn : NextValueFn | dict[int, NextValueFn] | None, optional
        Evaluates the successor state for the acting player; the result
        lands in ``info['next_value']`` (HAPPO critic; unused elsewhere).
        A dict maps each acting seat to its own critic; seats missing from
        the dict get no bootstrap value (对手座位不评估 critic）.
    max_steps : int
        Step guard (0 = unlimited) against pathological loops.
    """
    selectors = select_idx if isinstance(select_idx, dict) else {i: select_idx for i in range(len(players))}
    next_values = (
        next_value_fn
        if isinstance(next_value_fn, dict)
        else ({i: next_value_fn for i in range(len(players))} if next_value_fn is not None else {})
    )
    state = engine.create_initial_state()
    player_idx = {p: i for i, p in enumerate(players)}
    traj = EpisodeTrajectory()
    last_by_player: dict[int, Transition] = {}
    steps = 0
    abnormal = False

    def _abort():
        """Abnormal episode end: record nothing further, flag truncation."""
        nonlocal abnormal
        abnormal = True

    while True:
        if max_steps and steps >= max_steps:
            _abort()
            break
        steps += 1
        node = engine.get_node_type(state)
        if node == "chance":
            outcomes = engine.get_chance_outcomes(state)
            if not outcomes:
                _abort()
                break
            state = engine.apply_chance(state, weighted_choice(outcomes, rng))
            continue
        if node != "player":
            break  # terminal — normal end, payoffs settled below
        current = engine.get_current_player(state)
        if current is None or current not in player_idx:
            _abort()
            break
        pid = player_idx[current]
        legal = engine.get_legal_actions(state)
        if not legal:
            _abort()
            break
        # Forced decision (exactly one legal action — e.g. no-choice
        # ``claim_pass`` states): apply it directly without recording a
        # transition.  Mahjong 4-player games are ~75% forced steps, so
        # recording them only pollutes the replay buffer with zero-signal
        # transitions and slows training ~4x; skipping is semantically
        # identical (the outcome is deterministic).  The next loop
        # iteration handles any chance nodes naturally.
        if len(legal) == 1:
            state = engine.apply_action(state, legal[0])
            steps += 1
            if max_steps and steps >= max_steps:
                _abort()
                break
            continue
        mask = action_space.legal_mask(state, legal)
        action_idx, info = selectors[pid](pid, state, mask)
        action = action_space.action_from_index(action_idx, legal)
        if action is None:
            _abort()
            break
        try:
            next_state = engine.apply_action(state, action)
        except Exception:
            # Engine/rules edge case (e.g. mahjong degenerate chi chains
            # crashing payoff evaluation): end the episode as a draw.
            _abort()
            break

        # Roll forward chance nodes so the recorded successor is a real
        # decision/terminal state (审查 P1-13): the old code kept the raw
        # post-action state, whose legal mask is empty on chance nodes
        # (mahjong claim_pass → draw, texas bet → deal).  An all-zero
        # ``next_mask`` collapsed the QMix/MAAC bootstrap (targets ≈ −1e9).
        forced_end = False
        while engine.get_node_type(next_state) == "chance":
            outcomes = engine.get_chance_outcomes(next_state)
            if not outcomes:
                # Dead chance node: the successor is not a decision state —
                # abort rather than record a bootstrap-free transition.
                forced_end = True
                break
            next_state = engine.apply_chance(next_state, weighted_choice(outcomes, rng))
            steps += 1
            if max_steps and steps >= max_steps:
                # Pathological chance→chance loop: hard stop without recording.
                forced_end = True
                break
        if forced_end:
            _abort()
            break

        done = engine.is_terminal(next_state)
        nv = next_values.get(pid)
        if not done and nv is not None:
            info["next_value"] = float(nv(next_state, pid))
        reward = 0.0
        if done:
            reward = float(engine.get_utility(next_state, current))
            for p in players:
                traj.payoffs[p] = float(engine.get_utility(next_state, p))

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

    if abnormal:
        # Truncated/aborted episode: zero payoffs and close every recorded
        # transition so HAPPO/MAAC do not bootstrap a never-resolved chain.
        traj.payoffs = {p: 0.0 for p in players}
        for t in traj.transitions:
            t.done = True
        return traj

    # Terminal utility only reaches each player's own final decision.
    if engine.is_terminal(state):
        for p in players:
            payoff = float(engine.get_utility(state, p))
            traj.payoffs[p] = payoff
            last = last_by_player.get(player_idx[p])
            if last is not None:
                last.reward = payoff
                last.done = True
    return traj
