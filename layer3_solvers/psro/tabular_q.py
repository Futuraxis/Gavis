"""Tabular Q-learning best-response solver for PSRO.

Given an environment and a fixed opponent policy, train a new
policy via tabular Q-learning.

The best response is trained for **both seats**: on alternate episodes
the Q agent acts as player 1 and as player 2, so the single shared
tabular policy (keyed by the perspective-relative ``GymAdapter``
encoding) is explicitly learned for every state the policy can be
asked to act in — including the color-swapped states the second seat
faces.  The old one-seat-only loop only ever trained player 1's
decision states, leaving the shared table untrained (≈ random) on the
other half of the game tree.
"""

from __future__ import annotations

import numpy as np

from .agent import Agent, TabularQAgent


def tabular_q_best_response(
    env,
    num_steps: int = 10000,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    gamma: float = 0.99,
    opponent_policy: np.ndarray | None = None,
    seed: int | None = None,
    alternate_seats: bool = True,
) -> np.ndarray:
    """Train a best-response policy via tabular Q-learning (both seats).

    Parameters
    ----------
    env : GymAdapter
        Must expose ``available_actions()``, ``get_current_player()`` and
        ``players`` (the two seat ids in order).
    num_steps : int
        Total training steps.
    epsilon, alpha, gamma : float
        Q-learning hyper-parameters.
    opponent_policy : np.ndarray, optional
        Fixed opponent policy. If None, uses random.
    seed : int, optional
        Seed for the Q init / exploration RNGs (审查 P2-23).
    alternate_seats : bool
        Alternate the trained seat per episode (default True). When
        False the agent only ever trains as player 1 (legacy mode).

    Returns
    -------
    np.ndarray, shape (state_dim, action_dim)
        Greedy policy (one-hot) derived from trained Q-table.
    """
    obs_dim = env.observation_space.n if hasattr(env.observation_space, "n") else env.observation_space.shape[0]
    n_actions = env.action_space.n

    if seed is not None:
        np.random.seed(seed)  # Q-table init uses the global RNG
    agent = TabularQAgent(obs_dim, n_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)
    agent.reset_rng(seed)
    opponent = (
        Agent(opponent_policy, action_dim=n_actions) if opponent_policy is not None else Agent(action_dim=n_actions)
    )
    opponent.reset_rng(seed)

    players = getattr(env, "players", None) or ()
    target_idx = 0  # seat the Q agent trains as this episode
    ep_count = 0

    obs, _ = env.reset()
    for step in range(num_steps):
        mask = env.available_actions()
        current = env.get_current_player()

        # Route the turn by the environment's own current-player query
        # (C-10); degrade to strict parity alternation only for synthetic
        # envs that do not report matching player ids.
        if current is not None and players and current in players:
            agent_acts = current == players[target_idx]
        else:
            agent_acts = step % 2 == target_idx

        if agent_acts:
            # Agent's decision.
            action = agent.select_action(obs, mask)
            next_obs, reward, done, _, _ = env.step(action)

            # Opponent's reply, then update on the agent's OWN next
            # decision state.  Updating before the opponent replies (the
            # old order) bootstrapped Q(s,a) from an opponent-turn state
            # the agent never acts in, and dropped the terminal ±1 payoff
            # of opponent-ending games entirely (审查 P1-1).
            if not done:
                opp_mask = env.available_actions()
                opp_action = opponent.step(next_obs, amask=opp_mask)
                next_obs, r2, done, _, _ = env.step(opp_action)
                reward = reward + gamma * r2
            # The adapter reports every reward from player 1's perspective;
            # in a zero-sum game player 2's payoff is the negation, so flip
            # the sign when training the SECOND seat.  Without this the
            # white seat's Q table learned to maximise p1's (black) payoff
            # and the trained white policy played at losing level (审查
            # 2026-08: quick test showed black 0.825 / white 0.275 vs random
            # before the flip).
            if target_idx == 1:
                reward = -reward
            agent.update(obs, action, reward, next_obs, done, next_mask=env.available_actions())
            obs = next_obs
        else:
            # Opponent's decision (covers the color-swapped half of the
            # game tree when the agent trains as player 2).
            opp_action = opponent.step(obs, amask=mask)
            next_obs, _, done, _, _ = env.step(opp_action)
            obs = next_obs

        if done:
            obs, _ = env.reset()
            ep_count += 1
            if alternate_seats:
                target_idx = ep_count % 2

    return agent.get_greedy_policy()
