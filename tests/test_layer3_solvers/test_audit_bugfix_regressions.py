"""Regression tests for the architecture-audit bug fixes (2026-08-13).

Each test pins one fixed bug from the audit report (C-xx / M-xx / minor
items) so it cannot silently regress:

  - C-01: CFR player indices must not assume 'p_black'/'p_white'.
  - C-02: MoonStateEncoder.get_action_mask must not hash ActionInstance.
  - C-03: BeliefTracker.signal_fn must be a real dataclass field.
  - C-04: binding schemas must work without pydantic (dataclass fallback).
  - C-05: StateTracker must re-record FIFO-replaced pieces.
  - C-06: Hybrid rollouts must not call the non-Protocol ``sample_chance``.
  - C-08: MAAC eval-time action selection must be deterministic (greedy).
  - C-10: PSRO turn routing must use env.get_current_player, not parity.
  - M-03: CFR.train(episodes) must actually run ``episodes`` iterations.
  - M-07: joint sampling must respect the role-count constraint.
  - M-09: suggest_solver must check chance before small-state PSRO.
  - M-13: Q-learning target max must mask illegal actions.
  - M-14: the random Agent must not hardcode 9 actions.
  - minor: _find_target token matching; template policy bilingual roles;
    PPO fallback encoding distinguishes sides; PPO 'mcts' opponent works;
    PSRO save/load roundtrip without pickle.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from layer2_engine.core.state_graph import clone_state
from layer2_engine.interfaces.solver_adapter import ActionInstance, ChanceOutcome

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ── Minimal protocol-only adapters ─────────────────────────────────


class TwoStepGame:
    """Minimal 2-player / 2-ply game with player ids p0/p1 (no chance).

    Deliberately does NOT provide ``sample_chance`` — solvers must work
    through the SolverAdapter Protocol alone.
    """

    def __init__(self) -> None:
        self.rules = {"utility": [{"player": "p0"}, {"player": "p1"}]}

    def create_initial_state(self) -> dict:
        return {"_arrays": {}, "env": {"turn": "p0", "ply": 0}, "_schema": None}

    def get_node_type(self, state: dict) -> str:
        return "terminal" if state["env"]["ply"] >= 2 else "player"

    def get_current_player(self, state: dict) -> str | None:
        return None if state["env"]["ply"] >= 2 else ("p0" if state["env"]["ply"] == 0 else "p1")

    def get_legal_actions(self, state: dict) -> list[ActionInstance]:
        cp = self.get_current_player(state)
        return [ActionInstance("move", "player", cp, {}, f"move_{cp}_{i}") for i in range(2)]

    def apply_action(self, state: dict, action: ActionInstance) -> dict:
        s = clone_state(state)
        s["env"]["ply"] += 1
        return s

    def get_chance_outcomes(self, state: dict) -> list:
        return []

    def apply_chance(self, state: dict, outcome) -> dict:
        return state

    def is_terminal(self, state: dict) -> bool:
        return state["env"]["ply"] >= 2

    def get_utility(self, state: dict, player: str) -> float:
        return 1.0 if player == "p0" else -1.0

    def get_info_set_key(self, state: dict, player: str) -> str:
        return f"{state['env']['ply']}|{player}"

    def get_observation(self, state: dict, player: str) -> dict:
        return {"ply": state["env"]["ply"]}

    def project_observation(self, state: dict, viewer: str) -> dict:
        return {}


class ChanceGame(TwoStepGame):
    """p0 acts, then a chance node, then terminal."""

    def get_node_type(self, state: dict) -> str:
        if state["env"]["ply"] == 1:
            return "chance"
        return "terminal" if state["env"]["ply"] >= 2 else "player"

    def get_current_player(self, state: dict) -> str | None:
        return None if state["env"]["ply"] != 0 else "p0"

    def get_chance_outcomes(self, state: dict) -> list[ChanceOutcome]:
        return [
            ChanceOutcome("h", 0.5, "eff", "h"),
            ChanceOutcome("t", 0.5, "eff", "t"),
        ]

    def apply_chance(self, state: dict, outcome) -> dict:
        s = clone_state(state)
        s["env"]["ply"] = 2
        return s


# ── C-01 / M-03: CFR on non-p_black player ids ─────────────────────


class TestCFRPlayerIndexing:
    def test_solve_with_arbitrary_player_ids(self):
        from layer3_solvers.cfr.solver import CFR, CFRConfig

        adapter = TwoStepGame()
        cfr = CFR(adapter, CFRConfig(seed=7, iterations=10, depth_limit=8))
        assert cfr._players == ["p0", "p1"]  # noqa: SLF001 — regression pin
        strategy = cfr.solve(adapter.create_initial_state())
        assert strategy
        assert abs(sum(strategy.values()) - 1.0) < 1e-6
        assert len(cfr.info_sets) > 0

    def test_train_uses_episodes_as_iterations(self):
        from layer3_solvers.cfr.solver import CFR, CFRConfig

        adapter = TwoStepGame()
        cfr = CFR(adapter, CFRConfig(seed=7, iterations=100))
        cfr.train(episodes=5, verbose=False)
        assert cfr._iter == 5  # noqa: SLF001 — M-03 regression pin
        assert len(cfr.info_sets) > 0

    def test_reset_clears_info_sets_and_iter_count(self):
        from layer3_solvers.cfr.solver import CFR, CFRConfig

        adapter = TwoStepGame()
        cfr = CFR(adapter, CFRConfig(seed=7, iterations=10))
        cfr.solve(adapter.create_initial_state())
        assert cfr.info_sets and cfr._iter > 0  # noqa: SLF001
        cfr.reset()
        assert not cfr.info_sets
        assert cfr._iter == 0  # noqa: SLF001


# ── C-06: Hybrid rollout must not require sample_chance ────────────


class TestHybridChanceRollout:
    def test_rollout_prior_uses_protocol_only(self):
        from layer3_solvers.hybrid.solver import HybridConfig, HybridSolver

        adapter = ChanceGame()  # no sample_chance method
        solver = HybridSolver(adapter, HybridConfig(seed=1, mcts_budget=10))
        value = solver._rollout_prior(adapter.create_initial_state(), "p0")  # noqa: SLF001
        assert value == 1.0  # p0 utility at the reached terminal state


# ── C-02: encoder action mask with ActionInstance ──────────────────


class TestEncoderMaskUnhashable:
    def test_get_action_mask_with_action_instances(self):
        from layer4_interface.encoding.moon_state_encoder import MoonStateEncoder

        encoder = MoonStateEncoder()
        state = {
            "board": [[None] * 3 for _ in range(3)],
            "currentPlayerId": "player_x",
            "pieceOrder": {},
            "stepCount": 0,
            "status": "running",
            "legalActions": [
                ActionInstance("place", "player", "p0", {"cell": {"id": "cell_0_0"}}, "place:cell_0_0"),
                ActionInstance("place", "player", "p0", {"cell": {"id": "cell_1_1"}}, "place:cell_1_1"),
            ],
            "playerSymbols": {"X": "player_x"},
        }
        mask = encoder.get_action_mask(state)  # used to raise TypeError
        assert mask.shape == (9,)
        assert mask[0] == 1.0  # cell_0_0
        assert mask[4] == 1.0  # cell_1_1
        assert mask[1] == 0.0


# ── C-03: BeliefTracker.signal_fn field ────────────────────────────


class TestBeliefSignalFn:
    def test_signal_fn_constructible(self):
        from dataclasses import fields

        from layer3_solvers.werewolf.belief import BeliefTracker

        names = {f.name for f in fields(BeliefTracker)}
        assert "signal_fn" in names  # C-03: must be a real field
        tracker = BeliefTracker(
            ["p0", "p1", "p2"],
            ["wolf", "villager", "villager"],
            "villager",
            signal_fn=lambda *_: 0.5,
            rng=random.Random(1),
        )
        assert tracker.signal_fn("x") == 0.5


# ── M-07 / minor: joint sampling + exact target matching ───────────


class TestBeliefSampling:
    def test_sample_assignment_respects_role_counts(self):
        from layer3_solvers.werewolf.belief import BeliefTracker

        players = [f"p{i}" for i in range(6)]
        tracker = BeliefTracker(
            players,
            ["wolf", "wolf", "villager", "villager", "villager", "seer"],
            "seer",
            rng=random.Random(7),
        )
        for _ in range(50):
            assign = tracker.sample_assignment()
            roles = list(assign.values())
            assert roles.count("wolf") == 2
            assert roles.count("villager") == 3

    def test_find_target_exact_token_match(self):
        from layer3_solvers.werewolf.belief import _find_target

        players = ["p1", "p10"]
        assert _find_target("我怀疑p10是狼", players) == "p10"  # not p1!
        assert _find_target("投p1", players) == "p1"
        assert _find_target("P10有问题", players) == "p10"  # case-insensitive
        assert _find_target("没人", players) is None


# ── C-04: binding schemas without pydantic ─────────────────────────


class TestSchemasFallback:
    def test_observation_works_without_pydantic(self, monkeypatch):
        import importlib
        import sys

        mod = importlib.import_module("layer4_interface.binding.schemas")
        saved = sys.modules.get("pydantic")
        monkeypatch.setitem(sys.modules, "pydantic", None)  # force ImportError
        try:
            reloaded = importlib.reload(mod)
            obs = reloaded.Observation(gameId="g", boardObservation=[["X"]])
            assert obs.gameId == "g"
            assert obs.source == "screen_capture"  # default applied
            assert obs.model_dump()["gameId"] == "g"
            default = reloaded.Observation()
            assert default.boardObservation == [[None] * 3 for _ in range(3)]
            assert default.confidence == [[0.0] * 3 for _ in range(3)]
        finally:
            if saved is None:
                monkeypatch.delitem(sys.modules, "pydantic", raising=False)
            else:
                monkeypatch.setitem(sys.modules, "pydantic", saved)
            importlib.reload(mod)  # restore the real pydantic-backed module


# ── C-05: StateTracker FIFO re-placement ───────────────────────────


class TestStateTrackerFIFO:
    def test_replaced_piece_gets_new_seq(self):
        from layer4_interface.binding.schemas import Observation
        from layer4_interface.binding.state_tracker import StateTracker

        tracker = StateTracker()
        empty = [[None] * 3 for _ in range(3)]
        with_x = [["X", None, None], [None] * 3, [None] * 3]
        tracker.update(Observation(boardObservation=with_x, frameSeq=1))
        tracker.update(Observation(boardObservation=empty, frameSeq=2))  # eviction
        tracker.update(Observation(boardObservation=with_x, frameSeq=3))  # NEW piece, same cell
        order = tracker.infer_piece_order()
        cell_entries = [e for e in order["player_x"] if e["cellId"] == "cell_0_0"]
        assert [e["placedSeq"] for e in cell_entries] == [1, 2]  # C-05 regression pin


# ── C-08: MAAC eval determinism ────────────────────────────────────


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestMAACGreedyEval:
    def test_select_action_is_deterministic(self):
        from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter
        from layer3_solvers.marl.maac import MAACConfig, MAACSolver

        adapter = MoonChessAdapter(seed=42)
        solver = MAACSolver(adapter, MAACConfig(seed=42))
        state = adapter.create_initial_state()
        first = solver.select_action(state)
        second = solver.select_action(state)
        assert first is not None and second is not None
        assert first.canonical_key == second.canonical_key  # greedy, not sampled


# ── C-10: PSRO turn routing via get_current_player ─────────────────


class TestPSROTurnRouting:
    def test_estimate_reward_uses_env_current_player(self):
        from layer3_solvers.psro.meta_game import estimate_reward

        log: list[str] = []

        class LoggingAgent:
            def __init__(self, name: str) -> None:
                self._name = name

            def step(self, obs: int, amask=None) -> int:  # noqa: N803
                log.append(self._name)
                return 0

        class InvertedTurnEnv:
            """Column acts FIRST — turn parity would route p1 first."""

            def __init__(self) -> None:
                self.turns = 0

            def reset(self) -> tuple[int, dict]:
                self.turns = 0
                return 0, {}

            def available_actions(self) -> np.ndarray:
                return np.array([True])

            def get_current_player(self) -> str:
                return "column" if self.turns == 0 else "row"

            @property
            def players(self) -> tuple[str, str]:
                return "row", "column"

            def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
                self.turns += 1
                return 0, 1.0 if self.turns >= 2 else 0.0, self.turns >= 2, False, {}

        estimate_reward(
            InvertedTurnEnv(),
            num_episodes=1,
            p1=LoggingAgent("one"),
            p2=LoggingAgent("two"),
            max_steps=2,
        )
        assert log == ["two", "one"]  # C-10: column (p2) acted first


# ── M-09: auto_selector precedence ─────────────────────────────────


class TestSuggestSolver:
    def test_chance_wins_over_small_state_space(self):
        from layer3_solvers.auto_selector.rules_analyzer import GameProfile, suggest_solver

        profile = GameProfile(
            board_size=3,
            state_space_estimate=100,  # small → old code returned 'psro'
            has_chance_nodes=True,
        )
        assert suggest_solver(profile) == "mcts"

    def test_small_state_without_chance_still_psro(self):
        from layer3_solvers.auto_selector.rules_analyzer import GameProfile, suggest_solver

        profile = GameProfile(board_size=3, state_space_estimate=100, has_chance_nodes=False)
        assert suggest_solver(profile) == "psro"


# ── M-13 / M-14: PSRO agents ───────────────────────────────────────


class TestPSROAgents:
    def test_q_update_masks_illegal_next_actions(self):
        from layer3_solvers.psro.agent import TabularQAgent

        agent = TabularQAgent(state_dim=2, action_dim=3, gamma=0.9, alpha=0.1, epsilon=0.0)
        agent.Q[1] = np.array([0.0, 0.0, 100.0])  # action 2 is illegal at state 1
        mask = np.array([True, True, False])
        agent.update(0, 0, reward=1.0, next_obs=1, done=False, next_mask=mask)
        # With masking the target is 1.0 (max over legal ≈ 0), so Q stays ≈0.1;
        # without masking the target would be 1.0 + 0.9*100 = 91 → Q ≈ 9.1.
        assert abs(agent.Q[0, 0]) < 1.0

    def test_random_agent_uses_action_dim(self):
        from layer3_solvers.psro.agent import Agent

        agent = Agent(action_dim=3)
        for _ in range(20):
            assert 0 <= agent.step(0) < 3  # no hardcoded randint(9)

    def test_random_agent_without_dim_raises(self):
        from layer3_solvers.psro.agent import Agent

        with pytest.raises(ValueError):
            Agent().step(0)  # unmasked random policy needs action_dim


# ── minor: template policy bilingual roles ─────────────────────────


class TestTemplatePolicyRoles:
    def test_english_wolf_role_is_detected(self):
        from layer3_solvers.social.base import LanguageObservation
        from layer3_solvers.social.template_policy import TemplatePolicy

        policy = TemplatePolicy(seed=1)
        obs = LanguageObservation(
            role="wolf",
            phase="speech",
            history=[{"speaker": "p3", "text": "我是村民", "round": 1}],
        )
        speech = policy.decide_speech(obs)
        assert "普通村民" in speech  # a wolf must bluff as a villager
        assert "可疑" in speech

    def test_villager_speaks_neutrally(self):
        from layer3_solvers.social.base import LanguageObservation
        from layer3_solvers.social.template_policy import TemplatePolicy

        policy = TemplatePolicy(seed=1)
        obs = LanguageObservation(
            role="villager",
            phase="speech",
            history=[{"speaker": "p3", "text": "我是村民", "round": 1}],
        )
        assert "普通村民" not in policy.decide_speech(obs)


# ── minor: PPO fallback encoding + mcts opponent ───────────────────


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestPPOFixes:
    def test_fallback_encoding_distinguishes_sides(self):
        from layer3_solvers.ppo.solver import PPOConfig, PPOSolver

        class BoardOnlyAdapter(TwoStepGame):
            def get_observation(self, state: dict, player: str) -> dict:
                return {"board": [[player, "p1" if player == "p0" else "p0", None]]}

        solver = PPOSolver(BoardOnlyAdapter(), PPOConfig(seed=1, state_dim=9, action_dim=9))
        feats = solver._get_features(BoardOnlyAdapter().create_initial_state())  # noqa: SLF001
        # cell0 = self → [0,1,0]; cell1 = opponent → [0,0,1]; cell2 = empty
        assert feats[1] == 1.0 and feats[2] == 0.0
        assert feats[4] == 0.0 and feats[5] == 1.0
        assert feats[6] == 1.0

    def test_mcts_opponent_returns_legal_action(self):
        from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter
        from layer3_solvers.ppo.solver import PPOConfig, PPOSolver

        adapter = MoonChessAdapter(seed=42)
        solver = PPOSolver(adapter, PPOConfig(seed=42))
        state = adapter.create_initial_state()
        action = solver._opponent_action(state, "mcts", "p_black")  # noqa: SLF001
        assert action is not None
        legal_keys = {a.canonical_key for a in adapter.get_legal_actions(state)}
        assert action.canonical_key in legal_keys


# ── minor: PSRO save/load roundtrip without pickle ─────────────────


class TestPSROSaveLoad:
    def test_pool_roundtrip(self, tmp_path):
        from layer3_solvers.psro.solver import PSROConfig, PSROSolver

        adapter = TwoStepGame()
        config = PSROConfig(seed=42, num_iters=1, num_steps_per_iter=1)
        solver = PSROSolver(adapter, config)
        path = tmp_path / "psro_roundtrip.npz"
        solver.save(str(path))

        restored = PSROSolver(adapter, config)
        restored.load(str(path))
        assert len(restored._policy_pool) == len(solver._policy_pool)  # noqa: SLF001
        np.testing.assert_array_equal(restored._policy_pool[0], solver._policy_pool[0])  # noqa: SLF001
