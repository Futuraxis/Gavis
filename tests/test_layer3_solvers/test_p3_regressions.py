"""Regression tests for the Layer3 P3 review batch (2026-08-22).

Covers the behavior-changing P3 fixes from .docs/review: CFR train-reset /
metrics (cfr.md), MARL abnormal-end done marking (happo_maac.md), and
LLM prompt/solver hygiene (llm.md).
"""

from __future__ import annotations

import random

import numpy as np

from layer2_engine.core.state_graph import clone_state
from layer2_engine.interfaces.solver_adapter import ActionInstance
from layer3_solvers.cfr import CFR, CFRConfig

# ── 最小双步双人游戏（CFR 测试用，与 test_audit_bugfix_regressions 同构） ──


class TwoStepGame:
    """Minimal 2-player / 2-ply game with player ids p0/p1 (no chance)."""

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
        if s["env"]["ply"] >= 2:
            s["env"]["winner"] = "p0"  # 终局：p0 胜（utility 表 p0=+1）
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


# ── CFR P3-3 / P3-4 ─────────────────────────────────────────────────


def test_cfr_train_resets_previous_run():
    """P3-4: 每次 train() 独立训练；warm-start 只经显式 solve()。"""
    cfr = CFR(TwoStepGame(), CFRConfig(seed=7, iterations=100))
    cfr.train(episodes=5, verbose=False)
    assert cfr._iter == 5  # noqa: SLF001
    cfr.train(episodes=3, verbose=False)
    # 第二次 train 从零开始，而非 5+3=8（此前 CFR+ 权重跨 train 累积）
    assert cfr._iter == 3  # noqa: SLF001


def test_cfr_train_avg_return_is_win_minus_loss():
    """P3-3: avg_return = (wins − losses)/total，不再与 win_rate 同值。"""
    cfr = CFR(TwoStepGame(), CFRConfig(seed=7, iterations=10))
    metrics = cfr.train(episodes=5, verbose=False)
    # avg = (w − (N−w))/N = 2·win_rate − 1（无平局时恒等式，钉住公式）
    assert abs(metrics.avg_return - (2 * metrics.win_rate - 1)) < 1e-9
    assert isinstance(metrics.extra["info_sets"], int)


# ── MARL P3: 异常/截断 episode 全 transition 打 done 标记 ────────────


def test_run_episode_truncation_marks_all_done():
    from layer2_engine.games.moon_chess.moon_env_adapter import MoonChessAdapter
    from layer3_solvers.marl.action_space import ActionSpace
    from layer3_solvers.marl.encoders import GameEncoder
    from layer3_solvers.marl.env import run_episode

    adapter = MoonChessAdapter(seed=1)
    players = ["p_black", "p_white"]
    encoder = GameEncoder.build_from_adapter(adapter, players)
    action_space = ActionSpace.build_from_adapter(adapter)

    def _pick(pid: int, state: dict, mask: np.ndarray) -> tuple[int, dict]:
        return int(np.flatnonzero(mask)[0]), {}

    traj = run_episode(
        adapter, players, random.Random(1), encoder, action_space, _pick, max_steps=2
    )
    assert traj.transitions
    # 截断：全部 transition 关闭（HAPPO/MAAC 不再对未终结子序列 bootstrap）
    assert all(t.done for t in traj.transitions)
    assert all(t.reward == 0.0 for t in traj.transitions)
    assert traj.payoffs == {p: 0.0 for p in players}


# ── LLM P3-2 / P3-6 ─────────────────────────────────────────────────


class _RulesOnlyAdapter:
    """OllamaSolver 只读 rules 的场景（_default_player 便宜推导用）。"""

    rules = {"players": [{"id": "p3"}, {"id": "p7"}]}


def test_ollama_default_player_uses_rules():
    """P3-2: _default_player 不再 create_initial_state（8 席位曾重复 8 次发牌）。"""
    from layer3_solvers.llm.ollama_solver import OllamaConfig, OllamaSolver

    solver = OllamaSolver(_RulesOnlyAdapter(), OllamaConfig())
    assert solver.player_id == "p3"


def test_ollama_prompt_intents_derived_from_legal():
    """P3-6: 意图枚举从合法 speak 动作动态提取，不再硬编码。"""
    from layer3_solvers.llm.ollama_solver import OllamaConfig, OllamaSolver

    solver = OllamaSolver(_RulesOnlyAdapter(), OllamaConfig(), player_id="p3")
    obs = {
        "phase": "day_speech",
        "my_role": "villager",
        "alive": [1, 1, 1],
        "round": 1,
        "deaths_arr": [],
    }
    legal = [
        ActionInstance("speak", "action", "p0", {"intent": {"id": "claim"}, "text": ""}, "speak:claim"),
        ActionInstance("speak", "action", "p0", {"intent": {"id": "accuse"}, "text": ""}, "speak:accuse"),
    ]
    prompt = solver._build_prompt(obs, legal)  # noqa: SLF001
    assert '"intent": "accuse|claim"' in prompt
    # 无 speak 动作时回退兜底枚举
    prompt2 = solver._build_prompt(obs, [])  # noqa: SLF001
    assert '"intent": "claim|accuse|defend|question|persuade"' in prompt2


def test_ollama_prompt_deaths_and_votes_formatted():
    """P3-4: 死亡/投票不再以 Python repr 进 prompt。"""
    from layer3_solvers.llm.ollama_solver import OllamaConfig, OllamaSolver

    solver = OllamaSolver(_RulesOnlyAdapter(), OllamaConfig(), player_id="p3")
    obs = {
        "phase": "day_vote",
        "my_role": "villager",
        "alive": [1, 1, 0],
        "round": 2,
        "deaths_arr": ["p2", "p0"],
        "vote_log": [{"voter": "p1", "target": "p2", "round": 2}],
        "witch_save_used": 0,
        "witch_poison_used": 1,
    }
    legal = [ActionInstance("vote", "action", "p1", {"target": {"id": "p2"}}, "vote:p2")]
    prompt = solver._build_prompt(obs, legal)  # noqa: SLF001
    assert "昨夜/近日死亡：p2、p0" in prompt
    assert "p1 → p2" in prompt
    assert "已用解药：未用，已用毒药：已用" in prompt  # 中文渲染，非 True/False
    assert "{" not in prompt.split("投票记录")[1].split("\n")[1]  # 非 repr
