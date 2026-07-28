"""PPO Agent 实现。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from binding.exceptions import InvalidActionMaskError
from encoding.moon_state_encoder import action_index_to_cell_id, cell_id_to_action_index

from .networks import ActorCriticNetwork
from .rollout_buffer import RolloutBuffer


@dataclass(slots=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 32


class PPOAgent:
    """只负责 PPO 自己的动作选择与更新。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 9,
        *,
        device: str | torch.device | None = None,
        config: PPOConfig | None = None,
    ) -> None:
        self.device = self._resolve_device(device)
        self.config = config or PPOConfig()
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.buffer = RolloutBuffer()
        self.action_dim = action_dim

    def select_action(
        self,
        state_vector: np.ndarray,
        action_mask: np.ndarray,
        legal_actions: list[str] | None = None,
    ) -> tuple[int, float, float]:
        mask_tensor = self._validate_mask(action_mask, legal_actions=legal_actions)
        state_tensor = torch.as_tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.network(state_tensor)
        masked_logits = logits.masked_fill(mask_tensor.unsqueeze(0) == 0, -1e9)
        distribution = torch.distributions.Categorical(logits=masked_logits)
        action = int(distribution.sample().item())
        if mask_tensor[action].item() == 0:
            raise InvalidActionMaskError("采样到了非法动作，说明 mask 校验失败。")
        return action, float(distribution.log_prob(torch.tensor(action, device=self.device)).item()), float(
            value.squeeze(0).item()
        )

    def evaluate_value(self, state_vector: np.ndarray) -> float:
        state_tensor = torch.as_tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, value = self.network(state_tensor)
        return float(value.squeeze(0).item())

    def record_transition(
        self,
        *,
        state: np.ndarray,
        action: int,
        action_mask: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        next_value: float,
    ) -> None:
        self.buffer.add(
            state=state,
            action=action,
            action_mask=action_mask,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value,
            next_value=next_value,
        )

    def update(self) -> dict[str, float]:
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        self.buffer.compute_returns_and_advantages(self.config.gamma, self.config.gae_lambda)
        advantages = torch.as_tensor(self.buffer.advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        self.buffer.advantages = advantages.cpu().numpy()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        batch_count = 0

        for _ in range(self.config.update_epochs):
            for batch in self.buffer.iterate_minibatches(self.config.minibatch_size, self.device):
                logits, values = self.network(batch.states)
                masked_logits = logits.masked_fill(batch.action_masks == 0, -1e9)
                distribution = torch.distributions.Categorical(logits=masked_logits)
                new_log_probs = distribution.log_prob(batch.actions)
                entropy = distribution.entropy().mean()

                ratio = torch.exp(new_log_probs - batch.old_log_probs)
                unclipped = ratio * batch.advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * batch.advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, batch.returns)
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
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

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
            },
            target,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    @staticmethod
    def build_action(actor_id: str, action_index: int) -> dict:
        return {
            "actorId": actor_id,
            "actionType": "place_piece",
            "parameters": {"targetCellId": action_index_to_cell_id(action_index)},
        }

    @staticmethod
    def action_from_cell_id(cell_id: str) -> int:
        return cell_id_to_action_index(cell_id)

    def _validate_mask(
        self,
        action_mask: np.ndarray,
        *,
        legal_actions: list[str] | None = None,
    ) -> torch.Tensor:
        mask = np.asarray(action_mask, dtype=np.float32)
        if mask.shape != (self.action_dim,):
            raise InvalidActionMaskError(f"动作掩码维度必须为 ({self.action_dim},)，收到 {mask.shape}。")
        if not np.any(mask > 0):
            raise InvalidActionMaskError("动作掩码全为 0，当前状态不可采样。")
        if legal_actions is not None:
            legal_indices = {cell_id_to_action_index(cell_id) for cell_id in legal_actions}
            mask_indices = {idx for idx, value in enumerate(mask.tolist()) if value > 0}
            if legal_indices != mask_indices:
                raise InvalidActionMaskError("legalActions 与 action_mask 不一致。")
        return torch.as_tensor(mask, dtype=torch.float32, device=self.device)

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return resolved
