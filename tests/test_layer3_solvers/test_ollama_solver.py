"""Tests for OllamaSolver (local-LLM player) — prompt building, JSON reply
parsing and fallback behaviour.  The model call itself is mocked; the
werewolf engine is exercised for real (no network needed)."""

from __future__ import annotations

import pytest

from layer2_engine.games.werewolf.werewolf_adapter import WerewolfAdapter
from layer3_solvers import OllamaConfig, OllamaSolver


def _adapter(seed: int = 7) -> WerewolfAdapter:
    return WerewolfAdapter(seed=seed)


def _advance(adapter, phase_target: str, max_steps: int = 200):
    """Play random legal moves until ``phase_target`` (return state)."""
    import random
    from dataclasses import replace

    rng = random.Random(1)
    state = adapter.create_initial_state()
    for _ in range(max_steps):
        if state["env"].get("phase") == phase_target:
            return state
        nt = adapter.get_node_type(state)
        if nt == "chance":
            outs = adapter.get_chance_outcomes(state)
            if not outs:
                return state
            state = adapter.apply_chance(state, rng.choices(outs, weights=[o.probability for o in outs], k=1)[0])
        elif nt == "player":
            legal = adapter.get_legal_actions(state)
            if not legal:
                return state
            a = rng.choice(legal)
            if a.template_id == "speak":
                a = replace(a, params={**a.params, "text": "x"})
            state = adapter.apply_action(state, a)
            if adapter.is_terminal(state):
                return state
        else:
            return state
    return state


class _FakeModel:
    """Replaces the network call with a canned reply."""

    def __init__(self, solver: OllamaSolver, reply: str):
        self.solver = solver
        self.reply = reply

    def __enter__(self):
        self._orig = self.solver._ask_model
        self.solver._ask_model = lambda prompt: self.reply
        self.last_prompt = None
        self.solver._ask_model = self._fake
        return self

    def _fake(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.reply

    def __exit__(self, *args):
        self.solver._ask_model = self._orig


def test_prompt_contains_role_alive_speech():
    adapter = _adapter()
    solver = OllamaSolver(adapter, OllamaConfig(fallback_seed=1), player_id="p0")
    # 白天发言阶段
    st = None
    for s in range(30):
        cand = _advance(adapter, "day_speech", max_steps=500)
        if cand is not None and adapter.get_node_type(cand) == "player":
            st = cand
            break
        adapter = _adapter(seed=30 + s)
        solver = OllamaSolver(adapter, OllamaConfig(fallback_seed=1), player_id="p0")
    assert st is not None
    with _FakeModel(solver, '{"intent": "claim", "speech": "我是预言家"}') as fake:
        solver.select_action(st)
    assert fake.last_prompt is not None
    assert "身份是" in fake.last_prompt
    assert "存活玩家" in fake.last_prompt
    assert "请只输出一个 JSON" in fake.last_prompt


def test_speak_reply_maps_to_speech_with_text():
    adapter = _adapter()
    solver = OllamaSolver(adapter, OllamaConfig(fallback_seed=1), player_id="p0")
    st = None
    for s in range(30):
        cand = _advance(adapter, "day_speech", max_steps=500)
        if cand is not None and adapter.get_node_type(cand) == "player":
            st = cand
            break
        adapter = _adapter(seed=60 + s)
        solver = OllamaSolver(adapter, OllamaConfig(fallback_seed=1), player_id="p0")
    assert st is not None
    with _FakeModel(solver, '{"intent": "accuse", "speech": "我怀疑 p2"}'):
        action = solver.select_action(st)
    assert action is not None
    assert action.template_id == "speak"
    assert action.params["intent"]["id"] == "accuse"
    assert action.params["text"] == "我怀疑 p2"
    # 动作必须合法
    legal_keys = {a.canonical_key for a in adapter.get_legal_actions(st)}
    assert action.canonical_key in legal_keys


def test_target_reply_maps_to_action():
    adapter = _adapter()
    solver = OllamaSolver(adapter, OllamaConfig(fallback_seed=1), player_id="p0")
    st = _advance(adapter, "day_vote", max_steps=500)
    if adapter.get_node_type(st) != "player":
        pytest.skip("no game reached day_vote")
    legal = adapter.get_legal_actions(st)
    target = next(a.params["target"]["id"] for a in legal if a.template_id == "vote")
    with _FakeModel(solver, f'{{"target": "{target}"}}'):
        action = solver.select_action(st)
    assert action is not None
    assert action.template_id == "vote"
    assert action.params["target"]["id"] == target


def test_malformed_reply_falls_back():
    adapter = _adapter()
    solver = OllamaSolver(adapter, OllamaConfig(fallback_seed=5), player_id="p0")
    st = _advance(adapter, "day_vote", max_steps=500)
    if adapter.get_node_type(st) != "player":
        pytest.skip("no game reached day_vote")
    with _FakeModel(solver, "抱歉我什么都不知道"):
        action = solver.select_action(st)
    assert action is not None
    legal_keys = {a.canonical_key for a in adapter.get_legal_actions(st)}
    assert action.canonical_key in legal_keys


def test_bad_json_falls_back():
    adapter = _adapter()
    solver = OllamaSolver(adapter, OllamaConfig(fallback_seed=5), player_id="p0")
    st = _advance(adapter, "day_vote", max_steps=500)
    if adapter.get_node_type(st) != "player":
        pytest.skip("no game reached day_vote")
    with _FakeModel(solver, '{"target": "p99"}'):  # 目标不存在
        action = solver.select_action(st)
    assert action is not None
    legal_keys = {a.canonical_key for a in adapter.get_legal_actions(st)}
    assert action.canonical_key in legal_keys


def test_model_exception_falls_back():
    adapter = _adapter()
    solver = OllamaSolver(adapter, OllamaConfig(fallback_seed=5), player_id="p0")
    st = _advance(adapter, "day_vote", max_steps=500)
    if adapter.get_node_type(st) != "player":
        pytest.skip("no game reached day_vote")

    def boom(prompt):
        raise TimeoutError("ollama down")

    solver._ask_model = boom
    action = solver.select_action(st)
    assert action is not None
    legal_keys = {a.canonical_key for a in adapter.get_legal_actions(st)}
    assert action.canonical_key in legal_keys
