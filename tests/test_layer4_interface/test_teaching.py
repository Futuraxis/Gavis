"""Tests for teaching matches (教学对局).

Covers the three integration red lines documented in ``agent/coach.py``:

1. 教练看的 = 玩家看的 —— ``Coach`` 的观测来自玩家自己的投影（含玩家
   手牌视图），永不触碰 ground ``_arrays``，``assert_no_hidden`` 照常成立。
2. 双脑分离 —— AI 自己的落子路径零改动；教练在**玩家座位**上用同一
   求解器契约算参考动作（``raw_solver``）。
3. 在线学习防污染 —— 教练的参考动作不被 ``RecordingHandle`` 采集。
"""

from __future__ import annotations

from typing import Any

import pytest

from layer4_interface.agent import (
    PERSONAS,
    SCENARIOS,
    Coach,
    DialogueEngine,
    Skills,
    TeachContext,
    assert_no_hidden,
    scan,
)
from layer4_interface.frontend.engine_helpers import engine_from_rules, resolve_all_chance
from layer4_interface.frontend.platform.history import MatchHistory
from layer4_interface.frontend.platform.session import PlayManager
from layer4_interface.online_learning.recorder import RecordingHandle, TrajectoryRecorder
from train_cli import default_provider

_TEACH_SCENARIOS = ("teach_greet", "teach_turn", "teach_move")


@pytest.fixture
def manager(tmp_path: pytest.TempPathFactory) -> PlayManager:
    return PlayManager(provider=default_provider, history=MatchHistory(tmp_path), seed=42)


@pytest.fixture
def coach_manager(tmp_path: pytest.TempPathFactory) -> PlayManager:
    """PlayManager with a deterministic (fallback-line) agent factory."""

    def _factory(persona_key: str) -> DialogueEngine:
        return DialogueEngine(PERSONAS[persona_key])

    return PlayManager(
        provider=default_provider,
        history=MatchHistory(tmp_path),
        seed=42,
        agent_factory=_factory,
    )


def _drained_scenarios(snapshot: dict) -> list[str]:
    return [entry["scenario"] for entry in snapshot.get("chat", [])]


# ── Coach (agent layer) ───────────────────────────────────────────


class TestCoach:
    def test_build_extracts_player_hand_from_own_projection(self):
        """教练的手牌来自玩家自己的投影视图（hand_view_p0），非 ground 数组。"""
        engine = engine_from_rules("mahjong", seed=42, variant="guangdong", player_count=4)
        state = engine.create_initial_state()
        while engine.get_node_type(state) == "chance":
            _, state = engine.sample_chance(state)
        solver = default_provider.create_solver("mahjong_guangdong", "mahjong", engine, 42, 1)

        ctx = Coach.build(state, "p0", engine, solver)

        assert len(ctx.hand) == 14  # 庄家起手 14 张
        assert ctx.teaching is True
        assert ctx.legal_count == 14  # 14 个可打的选择
        # 红线 1：观测是玩家投影，hidden_guard 照常通过
        assert_no_hidden(ctx)

    def test_reference_is_a_legal_action_at_player_seat(self):
        """参考动作 = 求解器在玩家座位的推理结果（合法动作集合的一员）。"""
        engine = engine_from_rules("mahjong", seed=42, variant="guangdong", player_count=4)
        state = engine.create_initial_state()
        while engine.get_node_type(state) == "chance":
            _, state = engine.sample_chance(state)
        solver = default_provider.create_solver("mahjong_guangdong", "mahjong", engine, 42, 1)

        reference = Coach.reference_action(state, "p0", engine, solver)

        assert reference is not None
        legal_keys = {a.canonical_key for a in engine.get_legal_actions(state)}
        assert reference.canonical_key in legal_keys

    def test_reference_none_when_not_player_turn(self):
        """非玩家回合不算参考（防把别人的动作讲成玩家的）。"""
        engine = engine_from_rules("mahjong", seed=42, variant="guangdong", player_count=4)
        state = engine.create_initial_state()
        while engine.get_node_type(state) == "chance":
            _, state = engine.sample_chance(state)
        solver = default_provider.create_solver("mahjong_guangdong", "mahjong", engine, 42, 1)

        # p1 不是当前行动者（庄家 p0 先动）
        assert Coach.reference_action(state, "p1", engine, solver) is None

    def test_reference_silent_on_solver_failure(self):
        """求解器抛异常 → 静默降级为无参考（教学不阻断对局）。"""
        engine = engine_from_rules("mahjong", seed=42, variant="guangdong", player_count=4)
        state = engine.create_initial_state()
        while engine.get_node_type(state) == "chance":
            _, state = engine.sample_chance(state)

        class _Boom:
            def select_action(self, state: dict) -> Any:
                raise RuntimeError("boom")

        assert Coach.reference_action(state, "p0", engine, _Boom()) is None

    def test_build_without_solver_has_no_reference(self):
        """``teach_turn`` 导读不带参考（不剧透答案）。"""
        engine = engine_from_rules("mahjong", seed=42, variant="guangdong", player_count=4)
        state = engine.create_initial_state()
        while engine.get_node_type(state) == "chance":
            _, state = engine.sample_chance(state)

        ctx = Coach.build(state, "p0", engine, None)

        assert ctx.reference is None

    def test_extract_hand_texas_hole_view(self):
        """德州：私有视图命名是 ``<seat>_hole_view``（p_sb → sb_hole_view）。"""
        engine = engine_from_rules("texas_holdem", seed=42)
        state = resolve_all_chance(engine, engine.create_initial_state())

        ctx = Coach.build(state, "p_sb", engine, None)

        assert len(ctx.hand) == 2

    def test_extract_hand_empty_for_grid_games(self):
        """棋类游戏没有手牌视图 → 教练退化为纯局面讲解。"""
        engine = engine_from_rules("moon_chess", seed=42)
        state = engine.create_initial_state()

        ctx = Coach.build(state, "p_black", engine, None)

        assert ctx.hand == []

    def test_review_marks_match_and_mismatch(self):
        engine = engine_from_rules("mahjong", seed=42, variant="guangdong", player_count=4)
        state = engine.create_initial_state()
        while engine.get_node_type(state) == "chance":
            _, state = engine.sample_chance(state)
        solver = default_provider.create_solver("mahjong_guangdong", "mahjong", engine, 42, 1)
        pre = Coach.build(state, "p0", engine, solver)
        reference = Coach.reference_action(state, "p0", engine, solver)
        legal = [a for a in engine.get_legal_actions(state) if a.canonical_key != reference.canonical_key]

        matched = Coach.review(pre, reference, lambda a: str(a.canonical_key))
        mismatch = Coach.review(pre, legal[0], lambda a: str(a.canonical_key))

        assert matched.matched is True
        assert matched.player_action == reference.canonical_key
        assert mismatch.matched is False
        assert mismatch.player_action == legal[0].canonical_key
        # 行动前上下文不被讲评破坏（hand/reference 属于决策时刻）
        assert mismatch.hand == pre.hand
        assert mismatch.reference == pre.reference


# ── Teaching scan (hidden guard variant) ──────────────────────────


class TestTeachingScan:
    def test_teaching_scan_allows_player_own_hand(self):
        """教学模式：玩家自己的手牌可以讨论（教练看的正是玩家投影）。"""
        output = scan("你的手牌已经听牌了，这手可以考虑进攻。", "mahjong", teaching=True)
        assert output == "你的手牌已经听牌了，这手可以考虑进攻。"

    def test_teaching_scan_still_rewrites_ai_hand(self):
        """教学模式：AI/对手的隐藏信息仍然拦截（教练看不到它们）。"""
        output = scan("我的手牌是清一色。你的手牌不错。", "mahjong", teaching=True)
        assert "清一色" not in output
        assert "你的手牌不错" in output

    def test_teaching_scan_rewrites_opponent_hole(self):
        output = scan("对手的底牌是♠A ♥K，稳赢。你的底牌是小对子。", "texas_holdem", teaching=True)
        assert "♠A" not in output.split("你的底牌")[0]
        assert "你的底牌是小对子" in output

    def test_default_scan_unchanged(self):
        """非教学模式：任何手牌讨论照旧全拦（既有红线不变）。"""
        assert "不细说" in scan("你的手牌不错。", "mahjong")


# ── Dialogue engine (teaching prompt + payload) ───────────────────


class _RecordingLLM:
    """LLM stub that captures the system / user prompt and echoes a reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.system: str | None = None
        self.user: str | None = None

    def complete_chat(self, system: str, user: str, max_tokens: int) -> str:
        self.system = system
        self.user = user
        return self.reply


class TestDialogueTeach:
    def _teach_ctx(self) -> TeachContext:
        engine = engine_from_rules("mahjong", seed=42, variant="guangdong", player_count=4)
        state = engine.create_initial_state()
        while engine.get_node_type(state) == "chance":
            _, state = engine.sample_chance(state)
        return Coach.build(state, "p0", engine, None)

    def test_teaching_system_prompt_and_payload(self):
        llm = _RecordingLLM("这手先把孤张打掉。")
        engine_dialog = DialogueEngine(PERSONAS["teacher"], llm=llm)
        ctx = self._teach_ctx()

        message = engine_dialog.reply(ctx, "teach_turn")

        assert message.text == "这手先把孤张打掉。"
        assert message.mood == "thinking"
        assert "教练" in llm.system
        assert "AI/对手" in llm.system
        assert "player_hand" in llm.user  # 机械事实含玩家手牌

    def test_normal_system_prompt_without_teaching_ctx(self):
        llm = _RecordingLLM("这步不错。")
        engine = engine_from_rules("moon_chess", seed=42)
        ctx = Skills.build(engine.create_initial_state(), "p_black", engine)

        DialogueEngine(PERSONAS["gentle"], llm=llm).reply(ctx, "good_move")

        assert "教练" not in llm.system

    def test_teach_scenarios_have_fallbacks_for_every_persona(self):
        for persona in PERSONAS.values():
            for scenario in _TEACH_SCENARIOS:
                assert scenario in SCENARIOS
                assert persona.fallback_lines.get(scenario), f"{persona.key}/{scenario} 缺少兜底台词"

    def test_teach_fallback_lines_used_without_llm(self):
        ctx = self._teach_ctx()
        for persona in PERSONAS.values():
            message = DialogueEngine(persona).reply(ctx, "teach_move")
            assert message.text in persona.fallback_lines["teach_move"]


# ── PlayManager integration ───────────────────────────────────────


class TestPlayManagerTeaching:
    def test_start_teaching_flags_snapshot_and_defaults_teacher_persona(self, coach_manager):
        session = coach_manager.start("moon_chess", "p_black", "easy", teaching=True)

        assert session.teaching is True
        snapshot = session.snapshot()
        assert snapshot["teaching"] is True
        assert session.persona == "teacher"
        # 开局即有教练消息：teach_greet（+ 轮到玩家时的 teach_turn）
        scenarios = _drained_scenarios(snapshot)
        assert "teach_greet" in scenarios
        assert "teach_turn" in scenarios

    def test_explicit_persona_respected_in_teaching(self, coach_manager):
        session = coach_manager.start("moon_chess", "p_black", "easy", persona="gentle", teaching=True)
        assert session.persona == "gentle"

    def test_non_teaching_start_has_no_teach_messages(self, coach_manager):
        session = coach_manager.start("moon_chess", "p_black", "easy")
        assert session.teaching is False
        assert session.snapshot()["teaching"] is False
        assert "teach_greet" not in _drained_scenarios(session.snapshot())
        assert session.pending_teach is None

    def test_step_sets_pending_teach_review(self, manager):
        """教学局每步（行动前算参考 → 行动后并入讲评）。"""
        session = manager.start("moon_chess", "p_black", "easy", teaching=True)
        assert Coach.reference_action(session.state, session.player_pid, session.engine, session.raw_solver)

        session.step({"cell_index": 0})

        assert session.pending_teach is not None
        assert isinstance(session.pending_teach.matched, bool)
        assert session.pending_teach.player_action is not None

    def test_step_matched_with_deterministic_solver(self, manager):
        """确定性求解器（麻将启发式）下：走参考动作 → matched=True。

        MCTS 是随机求解器（两次独立调用可能给出不同参考），matched 的
        会话级断言用确定性的麻将启发式做。
        """
        session = manager.start("mahjong_guangdong", "p0", "easy", teaching=True)
        reference = Coach.reference_action(session.state, session.player_pid, session.engine, session.raw_solver)
        assert reference is not None

        session.step({"type": "discard", "tile": reference.params["tile"]})

        assert session.pending_teach is not None
        assert session.pending_teach.matched is True

    def test_step_mismatch_marks_not_matched(self, manager):
        session = manager.start("mahjong_guangdong", "p0", "easy", teaching=True)
        reference = Coach.reference_action(session.state, session.player_pid, session.engine, session.raw_solver)
        other = next(
            action.params["tile"]
            for action in session.engine.get_legal_actions(session.state)
            if action.template_id == "discard" and action.params["tile"] != reference.params["tile"]
        )

        session.step({"type": "discard", "tile": other})

        assert session.pending_teach is not None
        assert session.pending_teach.matched is False

    def test_non_teaching_step_leaves_no_pending_teach(self, manager):
        session = manager.start("moon_chess", "p_black", "easy")
        session.step({"cell_index": 0})
        assert session.pending_teach is None

    def test_move_queues_teach_messages(self, coach_manager):
        session = coach_manager.start("mahjong_guangdong", "p0", "easy", teaching=True)
        legal = session.snapshot()["legal"]
        tile = next(action["tile"] for action in legal if action["type"] == "discard")

        snapshot = coach_manager.move(session.game_id, {"type": "discard", "tile": tile})

        scenarios = _drained_scenarios(snapshot)
        assert "teach_move" in scenarios

    def test_say_uses_coach_context_in_teaching(self, coach_manager):
        session = coach_manager.start("moon_chess", "p_black", "easy", teaching=True)

        message = coach_manager.say(session.game_id, "chat")

        assert message is not None  # 教练在场（agent_factory 装配）
        assert message["text"]

    def test_hint_returns_coach_reference_in_teaching(self, coach_manager):
        session = coach_manager.start("mahjong_guangdong", "p0", "easy", teaching=True)

        hint = coach_manager.hint(session.game_id, "specific")

        assert hint["action"]  # 参考动作 canonical key
        assert hint["hint"].startswith("教练建议")
        legal_keys = {a.canonical_key for a in session.engine.get_legal_actions(session.state)}
        assert hint["action"] in legal_keys

    def test_record_carries_teaching_flag(self, manager):
        session = manager.start("texas_holdem", "p_sb", "easy", teaching=True)
        snapshot = manager.move(session.game_id, {"choice": "fold"})
        assert snapshot["over"] is True
        matches = manager._history.list_matches()  # type: ignore[union-attr]
        assert matches[0]["teaching"] is True


# ── Online-learning anti-pollution (red line 3) ────────────────────


class _StubLearning:
    """LearningHooks stub that mirrors LearningManager's solver wrapping."""

    def __init__(self) -> None:
        self.store: Any = None

    def enabled(self, game_id: str) -> bool:
        return True

    def wrap_handle(self, session: Any, solver: Any) -> Any:
        recorder = TrajectoryRecorder(
            store=self.store,
            game_id=session.spec.game_id,
            match_id=session.game_id,
            started_at=session.started_at,
        )
        session.recorder = recorder
        return RecordingHandle(solver, recorder, session)

    def on_finished(self, session: Any) -> None:  # pragma: no cover - not exercised
        pass


class TestTeachingLearningIsolation:
    def test_reference_action_not_recorded_as_ai_decision(self, tmp_path):
        """教练替玩家算的参考动作不得混入 AI 决策轨迹（raw_solver 红线）。"""
        learning = _StubLearning()
        manager = PlayManager(
            provider=default_provider,
            history=MatchHistory(tmp_path),
            seed=42,
            learning=learning,
        )
        session = manager.start("mahjong_guangdong", "p0", "easy", teaching=True)
        assert session.solver is not session.raw_solver  # solver 已被录制句柄包装
        assert session.recorder is not None

        legal = session.snapshot()["legal"]
        tile = next(action["tile"] for action in legal if action["type"] == "discard")
        manager.move(session.game_id, {"type": "discard", "tile": tile})

        records = session.recorder.pending
        # 玩家决策恰好一条（human actor，玩家座位）
        human = [r for r in records if r["actor"] == "human"]
        assert len(human) == 1 and human[0]["player"] == "p0"
        # AI 决策都在 AI 座位（p1/p2/p3）——玩家座位上**没有** "ai" 记录，
        # 即教练的参考动作没有经录制句柄泄漏进训练数据。
        ai_records = [r for r in records if r["actor"] == "ai"]
        assert ai_records  # AI 座位照常采集（在线学习不受教学模式影响）
        assert all(r["player"] != "p0" for r in ai_records)
