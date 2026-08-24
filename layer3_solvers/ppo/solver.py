"""PPO Solver — Proximal Policy Optimization with GameEngine.

Refactored from the original ``PPOAgent`` to consume the ``GameEngine``
Protocol instead of a hard-coded environment.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from layer2_engine.core.state_graph import ActionInstance, State
from layer2_engine.core.engine import GameEngine

from ..base import SolverBase, SolverConfig, SolverMetrics
from .networks import ActorCriticNetwork
from .rollout_buffer import RolloutBuffer


@dataclass
class PPOConfig(SolverConfig):
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 32
    state_dim: int = 38  # default for MoonChess
    action_dim: int = 9  # default for 3×3 grid
    hidden_dim: int = 128  # ActorCriticNetwork 隐藏层宽度
    update_frequency: int = 16  # 每 N 局更新一次；0 表示每局更新（保持旧行为）


class PPOSolver(SolverBase):
    """PPO solver that works with any ``GameEngine``.

    Designed specifically for games where the engine provides
    ``get_feature_vector()`` and ``get_action_mask()`` .
    """

    def __init__(self, engine: GameEngine, config: SolverConfig | None = None):
        super().__init__(engine, config or PPOConfig())
        cfg = self.config
        self._state_dim = getattr(cfg, "state_dim", 38)
        self._action_dim = getattr(cfg, "action_dim", 9)
        # Generic board size: read the rules' declared constant (pure data);
        # falls back to the 3×3 default when absent.
        self._board_size = (getattr(self.engine, "_constants", {}) or {}).get("board_size", 3)

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)
            np.random.seed(cfg.seed)

        self.device = self._resolve_device(cfg.device)
        self.network = ActorCriticNetwork(
            self._state_dim,
            self._action_dim,
            hidden_dim=getattr(cfg, "hidden_dim", 128),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg.learning_rate)
        self.rng = random.Random(cfg.seed)
        self.buffer = RolloutBuffer(rng=self.rng)
        self._mcts_opponent = None  # lazy (created on first 'mcts' opponent move)
        self._last_agent_step = None  # (feature, action_idx, mask, log_prob, value)
        self._last_log_prob = 0.0
        self._last_value = 0.0

    @property
    def name(self) -> str:
        return f"PPO(dim={self._state_dim})"

    # ── SolverBase ────────────────────────────────────────────────

    def select_action(self, state: State) -> Optional[ActionInstance]:
        """Select best action (greedy, no noise) for a given state."""
        feature = self._get_features(state)
        mask = self._get_mask(state)

        mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        state_tensor = torch.as_tensor(feature, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not mask_tensor.any():
            return None

        with torch.no_grad():
            logits, _ = self.network(state_tensor)
            masked_logits = logits.masked_fill(mask_tensor.unsqueeze(0) == 0, -1e9)
            action = int(torch.argmax(masked_logits, dim=1).item())

        return self._action_from_index(state, action)

    def train(self, episodes: int = 100, **kwargs) -> SolverMetrics:
        """Train PPO via self-play or vs random opponent.

        Parameters
        ----------
        episodes : int
            Number of episodes to train.
        opponent : str, optional
            'random' (default), 'self' (self-play), or 'mcts'
        """
        opponent_type = kwargs.get("opponent", "random")
        verbose = kwargs.get("verbose", False)
        controlled_player = kwargs.get("controlled_player", "p_black")

        wins = 0
        total_reward = 0.0
        total_steps = 0

        for ep in range(episodes):
            state = self.engine.create_initial_state()
            self._last_agent_step = None
            ep_reward = 0.0
            step = 0

            while not self.engine.is_terminal(state):
                nt = self.engine.get_node_type(state)
                if nt == "player":
                    cp = self.engine.get_current_player(state)
                    if cp == controlled_player:
                        action = self._select_action_train(state)
                        log_prob = self._last_log_prob
                        value = self._last_value
                        mask = self._get_mask(state)

                        next_state = self.engine.apply_action(state, action)
                        done = self.engine.is_terminal(next_state)
                        reward = self._get_reward(next_state, controlled_player, done)
                        next_value = 0.0 if done else self._evaluate_value(next_state)

                        self.buffer.add(
                            state=self._get_features(state),
                            action=self._action_to_index(action),
                            action_mask=mask,
                            log_prob=log_prob,
                            reward=reward,
                            done=done,
                            value=value,
                            next_value=next_value,
                        )
                        ep_reward += reward
                        self._last_agent_step = (
                            self._get_features(state),
                            self._action_to_index(action),
                            mask,
                            log_prob,
                            value,
                        )
                        state = next_state
                        step += 1
                    else:
                        # Opponent move
                        opp_action = self._opponent_action(state, opponent_type, controlled_player)
                        if opp_action is None:
                            break
                        state = self.engine.apply_action(state, opp_action)
                        if self.engine.is_terminal(state) and self._last_agent_step is not None:
                            feat, act_idx, m, lp, v = self._last_agent_step
                            self.buffer.add(
                                state=feat,
                                action=act_idx,
                                action_mask=m,
                                log_prob=lp,
                                reward=self._get_reward(state, controlled_player, True),
                                done=True,
                                value=v,
                                next_value=0.0,
                            )
                elif nt == "chance":
                    outcomes = self.engine.get_chance_outcomes(state)
                    if outcomes:
                        o = self._sample_outcome(outcomes, self.rng)
                        if o is not None:
                            state = self.engine.apply_chance(state, o)
                else:
                    break
                total_steps += 1  # 环境总步数（含对手/运气步）

            # Episode end
            if self.engine.is_terminal(state):
                winner = state["env"].get("winner")
                if winner == controlled_player:
                    wins += 1
            total_reward += ep_reward

            # PPO update（攒批：每 update_frequency 局更新一次）
            if len(self.buffer) > 0 and (
                self.config.update_frequency <= 0 or (ep + 1) % self.config.update_frequency == 0
            ):
                metrics = self._update()
                if verbose and (ep + 1) % max(1, episodes // 10) == 0:
                    win_pct = wins / (ep + 1) * 100
                    print(
                        f"  PPO ep {ep + 1:4d}/{episodes}  "
                        f"win={win_pct:5.1f}%  "
                        f"pl={metrics['policy_loss']:.4f}  "
                        f"vl={metrics['value_loss']:.4f}  "
                        f"ent={metrics['entropy']:.4f}"
                    )

        if len(self.buffer) > 0:
            self._update()

        return SolverMetrics(
            episodes=episodes,
            win_rate=wins / max(1, episodes),
            avg_return=total_reward / max(1, episodes),
            extra={"steps": total_steps},
        )

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "model_meta": {
                    "input_dim": self._state_dim,
                    "action_dim": self._action_dim,
                    "hidden_dim": getattr(self.config, "hidden_dim", 128),
                },
            },
            target,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        meta = checkpoint.get("model_meta")
        if meta and meta.get("input_dim") != self._state_dim:
            raise ValueError(
                f"checkpoint 的 input_dim={meta.get('input_dim')}，与当前配置 {self._state_dim} 不一致，拒绝加载"
            )
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # ── Internal: training step ───────────────────────────────────

    def _update(self) -> dict[str, float]:
        cfg = self.config
        self.buffer.compute_returns_and_advantages(cfg.gamma, cfg.gae_lambda)
        advantages = torch.as_tensor(self.buffer.advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        self.buffer.advantages = advantages.cpu().numpy()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        batch_count = 0

        for _ in range(cfg.update_epochs):
            for batch in self.buffer.iterate_minibatches(cfg.minibatch_size, self.device):
                logits, values = self.network(batch.states)
                masked_logits = logits.masked_fill(batch.action_masks == 0, -1e9)
                dist = torch.distributions.Categorical(logits=masked_logits)
                new_log_probs = dist.log_prob(batch.actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch.old_log_probs)
                unclipped = ratio * batch.advantages
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * batch.advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, batch.returns)
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                batch_count += 1

        self.buffer.clear()
        if batch_count == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        return {
            "policy_loss": total_policy_loss / batch_count,
            "value_loss": total_value_loss / batch_count,
            "entropy": total_entropy / batch_count,
        }

    # ── Internal: helpers ─────────────────────────────────────────

    def _select_action_train(self, state: State) -> ActionInstance:
        """Select action with exploration noise (for training)."""
        feature = self._get_features(state)
        mask = self._get_mask(state)
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        state_tensor = torch.as_tensor(feature, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            masked_logits = logits.masked_fill(mask_tensor.unsqueeze(0) == 0, -1e9)
            dist = torch.distributions.Categorical(logits=masked_logits)
            action_idx = int(dist.sample().item())
            self._last_log_prob = float(dist.log_prob(torch.tensor(action_idx, device=self.device)).item())
            self._last_value = float(value.squeeze(0).item())
        return self._action_from_index(state, action_idx)

    def _evaluate_value(self, state: State) -> float:
        feature = self._get_features(state)
        state_tensor = torch.as_tensor(feature, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, value = self.network(state_tensor)
        return float(value.squeeze(0).item())

    def _get_features(self, state: State) -> np.ndarray:
        """Feature vector from engine ground state / projected observation.

        Generic solver-side encoding (v5.2) — no game-specific engine
        method is probed.  Board cells are encoded as empty / self /
        opponent from the current player's perspective, sourced from (in
        order):

          1. ``state['_arrays']['board']`` — engine ground array (flat)
          2. ``obs['cell']`` — engine ``cell`` grid view (entities with
             ``x``/``y``/``occupant``)
          3. ``obs['board']`` — 2-D list shape (legacy adapter view)

        Games without a board fall back to flattening the numeric fields
        of the observation (padded to ``state_dim``).
        """
        cp = self.engine.get_current_player(state) or "p_black"
        obs = self.engine.get_observation(state, cp)
        flat = (state.get("_arrays", {}) or {}).get("board")
        cells = obs.get("cell")
        rows = None
        if isinstance(flat, list):
            size = int(round(len(flat) ** 0.5))
            rows = [flat[i * size : (i + 1) * size] for i in range(size)] if size else None
        elif cells:
            size = int(round(len(cells) ** 0.5)) or self._board_size
            rows = [["" for _ in range(size)] for _ in range(size)]
            for ent in cells:
                if isinstance(ent, dict) and "x" in ent and "y" in ent:
                    x, y = int(ent["x"]), int(ent["y"])
                    if 0 <= x < size and 0 <= y < size:
                        rows[y][x] = ent.get("occupant") or ""
        elif isinstance(obs.get("board"), list):
            rows = obs["board"]
        if rows:
            features = []
            for row in rows:
                for cell in row:
                    if cell is None or cell == "":
                        features.extend([1, 0, 0])
                    elif cell == cp:
                        features.extend([0, 1, 0])
                    else:
                        features.extend([0, 0, 1])
            arr = np.asarray(features, dtype=np.float32)
        else:
            scalars = [v for k, v in obs.items() if k != "env" and isinstance(v, (int, float, bool))]
            arr = np.asarray([float(v) for v in scalars], dtype=np.float32)
        if len(arr) < self._state_dim:
            arr = np.pad(arr, (0, self._state_dim - len(arr)))
        return arr[: self._state_dim]

    def _get_mask(self, state: State) -> np.ndarray:
        """Action mask from legal actions (canonical ``place:r,c`` keys)."""
        mask = np.zeros(self._action_dim, dtype=np.float32)
        legal = self.engine.get_legal_actions(state)
        for a in legal:
            idx = self._index_from_key(a)
            if idx is not None and 0 <= idx < self._action_dim:
                mask[idx] = 1.0
        return mask

    def _index_from_key(self, action: ActionInstance) -> int | None:
        """Map a legal action to its grid index via canonical key/params."""
        import re as _re

        m = _re.match(r"^place:(\d+),(\d+)$", action.canonical_key)
        if m:
            y, x = int(m.group(1)), int(m.group(2))
            return y * self._board_size + x
        cell = action.params.get("cell", {})
        cell_id = cell.get("id", "") if isinstance(cell, dict) else str(cell)
        try:
            _, r, c = cell_id.split("_")
            return int(r) * self._board_size + int(c)
        except (ValueError, IndexError):
            return None

    def _action_from_index(self, state: State, idx: int) -> ActionInstance:
        """Convert action index to the matching legal ActionInstance."""
        legal = self.engine.get_legal_actions(state)
        for a in legal:
            if self._index_from_key(a) == idx:
                return a
        raise ValueError(f"找不到编号 {idx} 对应的合法动作（legal={len(legal)} 个）")

    def _action_to_index(self, action: ActionInstance) -> int:
        """Convert ActionInstance back to index."""
        idx = self._index_from_key(action)
        if idx is not None:
            return idx
        raise ValueError(f"无法解析动作的格子编号: {action.canonical_key!r}")

    def _get_reward(self, state: State, player: str, done: bool) -> float:
        if not done:
            return 0.0
        return self.engine.get_utility(state, player)

    @staticmethod
    def _sample_outcome(outcomes, rng):
        """按 probability 加权采样，概率总和不必为 1。"""
        if not outcomes:
            return None
        total = sum(o.probability for o in outcomes)
        if total <= 0:
            return rng.choice(outcomes)
        r = rng.random() * total
        cumsum = 0.0
        for o in outcomes:
            cumsum += o.probability
            if r < cumsum:
                return o
        return outcomes[-1]

    def _opponent_action(self, state: State, opponent_type: str, controlled_player: str) -> Optional[ActionInstance]:
        """Return an opponent action.

        - 'random': uniform random.
        - 'self': the agent's own training policy (sampled, not greedy —
          greedy self-play never explores and collapses to a single line).
        - 'mcts': a small MCTS search (previously silently fell back to
          random).
        Unknown types raise ValueError.
        """
        legal = self.engine.get_legal_actions(state)
        if not legal:
            return None
        if opponent_type == "random":
            return self.rng.choice(legal)
        if opponent_type == "self":
            # Training-mode sampling; overwrites _last_log_prob/_last_value,
            # which is safe — the controlled player's buffer entry is
            # recorded immediately after its own move, before any opponent
            # move can run.
            return self._select_action_train(state)
        if opponent_type == "mcts":
            return self._mcts_opponent_action(state)
        raise ValueError(f"未知的对手类型: {opponent_type!r}（可选: random/self/mcts）")

    def _mcts_opponent_action(self, state: State) -> Optional[ActionInstance]:
        """Lazily-created small MCTS opponent (budget 500)."""
        if self._mcts_opponent is None:
            from ..mcts.solver import MCTS, MCTSConfig

            self._mcts_opponent = MCTS(
                self.engine,
                MCTSConfig(seed=getattr(self.config, "seed", None), budget=500),
            )
        return self._mcts_opponent.select_action(state)

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return resolved
