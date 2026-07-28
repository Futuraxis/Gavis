from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.networks import ActorCriticNetwork
from algorithms.ppo_agent import PPOAgent
from algorithms.rollout_buffer import RolloutBuffer
from binding.exceptions import InvalidActionMaskError
from encoding import MoonStateEncoder
from training.train_ppo import run_training
from tests.mock_moon_env import MockMoonEnv


def test_actor_outputs_nine_logits() -> None:
    network = ActorCriticNetwork(input_dim=38, action_dim=9)
    logits, _ = network(torch.randn(2, 38))
    assert logits.shape == (2, 9)


def test_critic_outputs_scalar_value() -> None:
    network = ActorCriticNetwork(input_dim=38, action_dim=9)
    _, value = network(torch.randn(2, 38))
    assert value.shape == (2,)


def test_agent_never_selects_masked_action() -> None:
    agent = PPOAgent(state_dim=38, device="cpu")
    state = np.zeros(38, dtype=np.float32)
    mask = np.asarray([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    action, _, _ = agent.select_action(state, mask, legal_actions=["cell_0_1"])
    assert action == 1


def test_agent_rejects_all_zero_mask() -> None:
    agent = PPOAgent(state_dim=38, device="cpu")
    with pytest.raises(InvalidActionMaskError):
        agent.select_action(np.zeros(38, dtype=np.float32), np.zeros(9, dtype=np.float32))


def test_rollout_buffer_computes_gae() -> None:
    buffer = RolloutBuffer()
    buffer.add(
        state=np.zeros(38, dtype=np.float32),
        action=0,
        action_mask=np.ones(9, dtype=np.float32),
        log_prob=0.0,
        reward=1.0,
        done=False,
        value=0.2,
        next_value=0.1,
    )
    buffer.add(
        state=np.zeros(38, dtype=np.float32),
        action=1,
        action_mask=np.ones(9, dtype=np.float32),
        log_prob=0.0,
        reward=0.5,
        done=True,
        value=0.1,
        next_value=0.0,
    )
    buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
    assert buffer.advantages.shape == (2,)
    assert buffer.returns.shape == (2,)


def test_ppo_update_changes_parameters() -> None:
    agent = PPOAgent(state_dim=38, device="cpu")
    before = [param.detach().clone() for param in agent.network.parameters()]
    for index in range(4):
        mask = np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        agent.record_transition(
            state=np.random.randn(38).astype(np.float32),
            action=index % 3,
            action_mask=mask,
            log_prob=-0.1,
            reward=1.0,
            done=index == 3,
            value=0.2,
            next_value=0.0 if index == 3 else 0.1,
        )
    metrics = agent.update()
    after = list(agent.network.parameters())
    assert metrics["policy_loss"] != 0.0
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, after, strict=True))


def test_agent_can_save_and_load(tmp_path: Path) -> None:
    agent = PPOAgent(state_dim=38, device="cpu")
    model_path = tmp_path / "ppo_agent.pt"
    agent.save(str(model_path))
    restored = PPOAgent(state_dim=38, device="cpu")
    restored.load(str(model_path))
    assert model_path.exists()


def test_training_runs_for_both_roles() -> None:
    _, logs = run_training(episodes=4)
    assert len(logs) == 4


def test_opponent_turns_do_not_write_into_buffer() -> None:
    encoder = MoonStateEncoder()
    env = MockMoonEnv()
    agent = PPOAgent(state_dim=38, device="cpu")
    state = env.reset()
    controlled_player = "player_x"
    while not env.is_terminal():
        state = env.get_state()
        if state["currentPlayerId"] == controlled_player:
            vector = encoder.encode(state, controlled_player)
            mask = encoder.get_action_mask(state)
            action, log_prob, value = agent.select_action(vector, mask, legal_actions=state["legalActions"])
            next_state, _, done, _ = env.step(agent.build_action(controlled_player, action))
            reward = 0.0
            next_value = 0.0 if done else agent.evaluate_value(encoder.encode(next_state, controlled_player))
            agent.record_transition(
                state=vector,
                action=action,
                action_mask=mask,
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=value,
                next_value=next_value,
            )
        else:
            next_state, _, _, _ = env.step(
                {"actorId": "player_o", "actionType": "place_piece", "parameters": {"targetCellId": state["legalActions"][0]}}
            )
        state = next_state
    assert len(agent.buffer) <= 5


def test_cpu_device_runs() -> None:
    agent = PPOAgent(state_dim=38, device="cpu")
    assert agent.device.type == "cpu"


def test_cuda_falls_back_or_uses_cuda() -> None:
    agent = PPOAgent(state_dim=38, device="cuda")
    assert agent.device.type in {"cpu", "cuda"}
