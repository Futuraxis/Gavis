"""Tests for the opponent-mode companion (二人非教练对手模式, P1).

Covers the four red lines / design points from
``docs/design/companion-redesign.md`` §3 + §8:

1. **adversarial scan 红线**（§3.3）：拦「你的/玩家的 + 牌面」、放行
   「我的/AI 的 + **牌力措辞**」（模糊，如「一对K」「这手同花」）；但
   **具体花色点数**（黑桃4 / ♠A / s10）一律拦——报牌等于明牌，破坏二人
   博弈；``revealed`` 全放行；与 ``teaching`` 镜像、与 default 互斥。
2. **数据入口**（§3.1）：``OpponentContext`` 来自 AI 自己的投影（含 AI
   手牌、不含玩家底牌），``assert_no_hidden`` 照常成立；``player_actions``
   为玩家公开动作序列。
3. **说话时机双触发**（§3.2）：AI 行动后 ``opp_react``、玩家行动后
   ``opp_read``；二人非教练 ``is_opponent_mode`` 成立、教练/多人/无 agent
   不成立。
4. **coach 不回归**：教学路径零改动（既有 ``test_teaching.py`` 已覆盖，
   此处补一条对手模式不污染教学的断言）。
"""

from __future__ import annotations

from typing import Any

import pytest

from layer2_engine.core.llm import ChatReply
from layer4_interface.agent import (
    PERSONAS,
    SCENARIOS,
    DialogueEngine,
    Opponent,
    OpponentContext,
    assert_no_hidden,
    scan,
)
from layer4_interface.frontend.engine_helpers import engine_from_rules, resolve_all_chance
from layer4_interface.frontend.platform.history import MatchHistory
from layer4_interface.frontend.platform.session import PlayManager
from train_cli import default_provider

_OPP_SCENARIOS = ("opp_react", "opp_read", "opp_taunt")


@pytest.fixture
def opp_manager(tmp_path: pytest.TempPathFactory) -> PlayManager:
    """PlayManager with a deterministic (fallback-line) agent factory."""

    def _factory(persona_key: str) -> DialogueEngine:
        return DialogueEngine(PERSONAS[persona_key])

    return PlayManager(
        provider=default_provider,
        history=MatchHistory(tmp_path),
        seed=42,
        agent_factory=_factory,
    )


@pytest.fixture
def plain_manager(tmp_path: pytest.TempPathFactory) -> PlayManager:
    """PlayManager without an agent factory (agent=None → cheerleader off)."""
    return PlayManager(provider=default_provider, history=MatchHistory(tmp_path), seed=42)


def _drained_scenarios(snapshot: dict) -> list[str]:
    return [entry["scenario"] for entry in snapshot.get("chat", [])]


def _drained_speakers(snapshot: dict) -> list[str]:
    return [entry.get("speaker", "") for entry in snapshot.get("chat", [])]


# ── adversarial scan (hidden guard variant) ──────────────────────────


class TestAdversarialScan:
    def test_adversarial_blocks_player_hole(self):
        """对手模式：玩家的底牌仍然拦截（AI 本就没有，防 LLM 幻觉编造）。"""
        output = scan("你的底牌是♠A，稳赢吧？", "texas_holdem", adversarial=True)
        assert "♠A" not in output
        assert "不细说" in output

    def test_adversarial_allows_ai_own_hand(self):
        """对手模式：AI 自己的**牌力措辞**（无花色点数）允许讨论。"""
        output = scan("我手里一对K，这手可以压一压你。", "texas_holdem", adversarial=True)
        assert output == "我手里一对K，这手可以压一压你。"

    def test_adversarial_blocks_ai_own_specific_cards(self):
        """对手模式：AI 自己底牌的**具体花色点数**仍拦（banter「不报牌」）。

        回归：用户报 AI 自报「黑桃4和黑桃10」明牌破坏对局——adversarial
        扫描须拦下任何带花色+点数的具体牌面（中文花色名 / 符号 / 英文字母），
        只改写命中句、保留无牌面的句子。
        """
        # 中文花色名（用户实际遇到的写法）——句号切句，无牌面句保留、命中句改写
        out = scan(
            "这局不急，局面还难分高下呢。我手上正好是黑桃4和黑桃10，"
            "虽然同花但不算大，我就先稳着跟一跟。你慢慢来，我陪着你走。",
            "texas_holdem",
            adversarial=True,
        )
        assert "黑桃4" not in out and "黑桃10" not in out
        assert "这局不急" in out  # 无牌面的句子保留
        assert "你慢慢来" in out  # 无牌面的句子保留
        assert "不细说" in out  # 命中句改写
        # 符号记法 + 英语记法
        assert "♥K" not in scan("我的底牌是♥K。", "texas_holdem", adversarial=True)
        assert "s10" not in scan("我摸到 s10。", "texas_holdem", adversarial=True)
        # 模糊牌力措辞照旧放行
        assert (
            scan("我的底牌还行，这手同花不算大。", "texas_holdem", adversarial=True) == "我的底牌还行，这手同花不算大。"
        )

    def test_adversarial_mahjong_blocks_player_hand(self):
        output = scan("你的手牌已经听牌了。我的手牌是清一色。", "mahjong", adversarial=True)
        assert "听牌" not in output.split("我的手牌")[0]
        assert "清一色" in output  # AI 自己的手牌放行

    def test_adversarial_werewolf_blocks_player_role(self):
        """对手模式：玩家身份拦截（「你是<角色>」「你的身份是」）；AI 自报身份允许。"""
        output = scan("你是狼人。我是预言家。", "werewolf", adversarial=True)
        # 玩家身份被拦（改写）；AI 自报身份放行
        assert "你是狼人" not in output
        assert "我是预言家" in output

    def test_revealed_allows_everything(self):
        """揭底后双方牌公开，全放行（可做完整复盘式对手点评）。"""
        text = "你的底牌是♠A。我的底牌是♥K♦K。你底牌小对子，我一对K。"
        assert scan(text, "texas_holdem", revealed=True) == text

    def test_revealed_overrides_adversarial(self):
        """revealed 优先级最高：即便 adversarial=True 也全放行。"""
        text = "你的底牌是♠A。"
        assert scan(text, "texas_holdem", adversarial=True, revealed=True) == text

    def test_default_scan_still_blocks_all(self):
        """默认模式（啦啦队）：底牌记法照旧全拦（既有红线不变）。

        注意：默认扫描拦「底牌」关键词 + 2+ 带花色牌面（♠A ♥K）；不带
        花色的牌型描述（「一对K」）不在默认模式拦截集里——这是既有行为
        （啦啦队模式本就不该出现手牌讨论，scan 是双保险防 LLM 幻觉），
        P1 不改动默认模式语义。
        """
        assert "不细说" in scan("你的底牌是♠A。", "texas_holdem")
        assert "不细说" in scan("我的底牌是♥K♦K。", "texas_holdem")

    def test_teaching_mirror_of_adversarial(self):
        """teaching 与 adversarial 镜像（对**模糊**措辞）：teaching 放行玩家
        牌、拦 AI 牌；adversarial 放行 AI 牌力措辞、拦玩家牌。**具体花色点数**
        在 adversarial 也拦（见 ``test_adversarial_blocks_ai_own_specific_cards``）。
        """
        # teaching：玩家牌放行（教练看的正是玩家投影）、AI 牌拦
        assert scan("你的底牌是♠A。", "texas_holdem", teaching=True) == "你的底牌是♠A。"
        assert "不细说" in scan("我的底牌是♥K。", "texas_holdem", teaching=True)
        # adversarial：AI 的**模糊**底牌措辞放行、玩家牌拦（镜像）
        assert scan("我的底牌还行。", "texas_holdem", adversarial=True) == "我的底牌还行。"
        assert "不细说" in scan("你的底牌是♠A。", "texas_holdem", adversarial=True)


# ── Opponent context (data entry) ────────────────────────────────────


class TestOpponentContext:
    def test_build_uses_ai_projection_and_passes_no_hidden(self):
        """OpponentContext.observation 是 AI 自己的投影；assert_no_hidden 成立。"""
        engine = engine_from_rules("texas_holdem", seed=42)
        state = resolve_all_chance(engine, engine.create_initial_state())

        ctx = Opponent.build(state, "p_bb", "p_sb", engine, [])

        assert ctx.adversarial is True
        assert ctx.ai_pid == "p_bb"
        assert ctx.human_pid == "p_bb"  # 基类字段 = 观测视角（AI 自己）
        assert ctx.revealed is False
        assert_no_hidden(ctx)  # AI 投影不含玩家隐藏字段

    def test_build_extracts_ai_own_hand(self):
        """ai_hand 取 AI 自己的底牌（从 AI 投影的 bb_hole_view）。"""
        engine = engine_from_rules("texas_holdem", seed=42)
        state = resolve_all_chance(engine, engine.create_initial_state())

        ctx = Opponent.build(state, "p_bb", "p_sb", engine, [])

        assert len(ctx.ai_hand) == 2  # 德州两张底牌

    def test_build_player_actions_from_log(self):
        """player_actions = 玩家公开动作序列（从 log 过滤 actor=="human"）。"""
        engine = engine_from_rules("texas_holdem", seed=42)
        state = resolve_all_chance(engine, engine.create_initial_state())
        log = [
            {"actor": "human", "action": "跟注 2"},
            {"actor": "ai", "action": "过牌"},
            {"actor": "human", "action": "下注 10"},
        ]

        ctx = Opponent.build(state, "p_bb", "p_sb", engine, log)

        assert ctx.player_actions == ["跟注 2", "下注 10"]

    def test_build_empty_log_yields_empty_player_actions(self):
        engine = engine_from_rules("texas_holdem", seed=42)
        state = resolve_all_chance(engine, engine.create_initial_state())

        ctx = Opponent.build(state, "p_bb", "p_sb", engine, [])

        assert ctx.player_actions == []

    def test_build_grid_game_has_empty_hand(self):
        """棋类游戏没有手牌视图 → ai_hand 为空（对手退化为纯局面说话）。"""
        engine = engine_from_rules("moon_chess", seed=42)
        state = engine.create_initial_state()

        ctx = Opponent.build(state, "p_white", "p_black", engine, [])

        assert ctx.ai_hand == []
        assert_no_hidden(ctx)

    def test_build_revealed_when_showdown_terminal(self):
        """终局 showdown 后 revealed=True（双方牌公开，全放行）。"""
        engine = engine_from_rules("texas_holdem", seed=42)
        state = resolve_all_chance(engine, engine.create_initial_state())
        # 构造终局 showdown 标记：is_terminal=True 且 last_action=="showdown"。
        state["env"]["last_action"] = "showdown"

        class _TermEngine:
            """Stub：仅覆盖 is_terminal（其余委托真实引擎行为）。"""

            def __init__(self, real: Any) -> None:
                self._real = real

            def is_terminal(self, s: dict) -> bool:
                return True

            def __getattr__(self, name: str) -> Any:
                return getattr(self._real, name)

        ctx = Opponent.build(state, "p_bb", "p_sb", _TermEngine(engine), [])
        assert ctx.revealed is True


# ── Dialogue engine (opponent prompt + payload) ──────────────────────


class _RecordingLLM:
    """LLM stub that captures the system / user prompt and echoes a reply."""

    def __init__(self, reply: str, *, reasoning: str = "") -> None:
        self.reply = reply
        self.reasoning = reasoning
        self.system: str | None = None
        self.user: str | None = None
        self.max_tokens: int | None = None
        #: 模拟 LLMClient.last_error（None = 无传输故障；设值模拟故障）。
        self.last_error: Exception | None = None

    def complete_chat(self, system: str, user: str, max_tokens: int) -> str:
        self.system = system
        self.user = user
        self.max_tokens = max_tokens
        return self.reply

    def complete_chat_reply(self, system: str, user: str, max_tokens: int) -> ChatReply:
        self.system = system
        self.user = user
        self.max_tokens = max_tokens
        return ChatReply(text=self.reply, reasoning=self.reasoning)


class TestDialogueOpponent:
    def _opp_ctx(self) -> OpponentContext:
        engine = engine_from_rules("texas_holdem", seed=42)
        state = resolve_all_chance(engine, engine.create_initial_state())
        log = [{"actor": "human", "action": "跟注 2"}]
        return Opponent.build(state, "p_bb", "p_sb", engine, log)

    def test_adversarial_system_prompt_and_payload(self):
        """OpponentContext 驱动对手系统 prompt + 对手机械事实（ai_hand/player_actions）。"""
        llm = _RecordingLLM("我这对K先压你一手。")
        engine_dialog = DialogueEngine(PERSONAS["banter"], llm=llm)
        ctx = self._opp_ctx()

        message = engine_dialog.reply(ctx, "opp_react")

        assert message.text == "我这对K先压你一手。"
        assert message.mood == "neutral"
        assert "对手" in llm.system  # 对手身份
        assert "未公开信息" in llm.system  # 红线（不报玩家牌）
        assert "ai_hand" in llm.user  # 机械事实含 AI 自己的牌
        assert "player_actions" in llm.user  # 含玩家公开动作序列

    def test_adversarial_scan_applied_to_llm_output(self):
        """LLM 若幻觉编造玩家底牌 → adversarial scan 改写该句；AI 自己的牌放行.

        scan 按句改写（句末标点切分）：玩家底牌句被改写、AI 自己的牌句
        原样保留——故测试用句号分隔两句，验证定向放行（同句逗号连接会
        整句改写，那是 scan 的既有句级粒度）。
        """
        llm = _RecordingLLM("你的底牌是♠A。我一对K稳赢。")
        engine_dialog = DialogueEngine(PERSONAS["banter"], llm=llm)
        ctx = self._opp_ctx()

        message = engine_dialog.reply(ctx, "opp_taunt")

        assert "♠A" not in message.text  # 玩家底牌句被改写
        assert "一对K" in message.text  # AI 自己的牌句保留

    def test_opp_scenarios_have_fallbacks_for_every_persona(self):
        for persona in PERSONAS.values():
            for scenario in _OPP_SCENARIOS:
                assert scenario in SCENARIOS
                assert persona.fallback_lines.get(scenario), f"{persona.key}/{scenario} 缺少兜底台词"

    def test_opp_fallback_lines_used_without_llm(self):
        ctx = self._opp_ctx()
        for persona in PERSONAS.values():
            message = DialogueEngine(persona).reply(ctx, "opp_read")
            assert message.text in persona.fallback_lines["opp_read"]


# ── LLM token budget decoupling (reasoning-model empty-content fix) ──


class TestDialogueMaxTokensDecoupled:
    """``max_tokens``（LLM token 预算）与 ``max_len``（可见正文清洗上限）解耦.

    根因：旧代码把 ``max_len=100`` 同时当 token 预算传给 LLM，推理模型
    （qwen3 / DeepSeek-R1）的思考阶段耗尽 100 token → ``content`` 空、
    只产出 reasoning → 误判「空回复」回退。对手模式每步 2 次 LLM 调用
    把这个潜在问题显化。修复后 token 预算独立（默认 512），可见正文仍
    截到 ``max_len`` 字符内。
    """

    def test_default_max_tokens_decoupled_from_max_len(self):
        engine = DialogueEngine(PERSONAS["gentle"])
        assert engine.max_len == 100  # 可见正文清洗上限不变
        assert engine.max_tokens == 512  # LLM token 预算独立、远大于 100

    def test_explicit_max_tokens_override(self):
        engine = DialogueEngine(PERSONAS["gentle"], max_tokens=2048)
        assert engine.max_tokens == 2048

    def test_llm_called_with_max_tokens_not_max_len(self):
        """LLM 收到的 ``max_tokens`` 是预算（512），不是清洗上限（100）。"""
        llm = _RecordingLLM("好棋。")
        engine = engine_from_rules("moon_chess", seed=42)
        ctx = Opponent.build(engine.create_initial_state(), "p_white", "p_black", engine, [])
        DialogueEngine(PERSONAS["gentle"], llm=llm).reply(ctx, "opp_read")
        assert llm.max_tokens == 512  # 不是 100

    def test_only_reasoning_diagnostic_logs_token_hint(self, caplog: pytest.LogCaptureFixture):
        """推理模型思考阶段耗尽 token → content 空、reasoning 非空 → 可操作诊断。"""
        llm = _RecordingLLM("", reasoning="思考了很多但没产出正文……")
        engine = engine_from_rules("moon_chess", seed=42)
        ctx = Opponent.build(engine.create_initial_state(), "p_white", "p_black", engine, [])
        with caplog.at_level("WARNING"):
            DialogueEngine(PERSONAS["gentle"], llm=llm).reply(ctx, "opp_read")
        msgs = " ".join(r.message for r in caplog.records)
        assert "仅产出思维链" in msgs  # 区分于泛化「空回复」
        assert "max_tokens" in msgs  # 给出可操作调参提示

    def test_transport_error_diagnostic_includes_last_error(self, caplog: pytest.LogCaptureFixture):
        llm = _RecordingLLM("")
        llm.last_error = RuntimeError("LLM 端点不可达")
        engine = engine_from_rules("moon_chess", seed=42)
        ctx = Opponent.build(engine.create_initial_state(), "p_white", "p_black", engine, [])
        with caplog.at_level("WARNING"):
            DialogueEngine(PERSONAS["gentle"], llm=llm).reply(ctx, "opp_read")
        msgs = " ".join(r.message for r in caplog.records)
        assert "未产出正文" in msgs
        assert "端点不可达" in msgs


# ── PlayManager integration (dual trigger + speaker) ─────────────────


class TestPlayManagerOpponent:
    def test_is_opponent_mode_for_two_player_non_teaching(self, opp_manager):
        session = opp_manager.start("moon_chess", "p_black", "easy")
        assert session.is_opponent_mode is True

    def test_not_opponent_mode_when_teaching(self, opp_manager):
        """教练模式不走对手通道（coach 零改动）。"""
        session = opp_manager.start("moon_chess", "p_black", "easy", teaching=True)
        assert session.is_opponent_mode is False

    def test_not_opponent_mode_for_multi_player(self, opp_manager):
        """多人非教练（麻将 4 人）在 P1 仍走啦啦队 fallback（P2 才换群聊）。"""
        session = opp_manager.start("mahjong_guangdong", "p0", "easy", player_count=4)
        assert session.is_opponent_mode is False

    def test_not_opponent_mode_when_agent_off(self, plain_manager):
        """无 agent_factory → agent=None → 不进入对手模式（纯对局）。"""
        session = plain_manager.start("moon_chess", "p_black", "easy")
        assert session.agent is None
        assert session.is_opponent_mode is False

    def test_speaker_is_persona_display_name_in_opponent_mode(self, opp_manager):
        session = opp_manager.start("moon_chess", "p_black", "easy", persona="banter")
        assert session.speaker == "轻松吐槽"

    def test_speaker_coach_prefix_in_teaching(self, opp_manager):
        session = opp_manager.start("moon_chess", "p_black", "easy", persona="gentle", teaching=True)
        assert session.speaker == "教练 · 温柔陪伴"

    def test_move_queues_opp_react_and_opp_read(self, opp_manager):
        """双触发：AI 行动后 opp_react、玩家行动后 opp_read 同时出现在快照 chat。"""
        session = opp_manager.start("texas_holdem", "p_sb", "easy", persona="banter")
        # 开局 greet 已被 start 快照 drain；这里发起第一手。
        snapshot = opp_manager.move(session.game_id, {"choice": "call"})
        scenarios = _drained_scenarios(snapshot)
        # opp_read（玩家行动后读人）必出现；opp_react 在 AI 行动后出现。
        # （若 AI 在 call 后未行动——例如直接终局——则只 opp_read；正常 call→check 必有 AI 行动。）
        if not snapshot["over"]:
            assert "opp_read" in scenarios
            assert "opp_react" in scenarios

    def test_pending_chat_entries_carry_speaker(self, opp_manager):
        """pending_chat 条目带 speaker 字段（前端按此渲染头像/名字）。"""
        session = opp_manager.start("texas_holdem", "p_sb", "easy", persona="banter")
        snapshot = opp_manager.move(session.game_id, {"choice": "call"})
        for entry in snapshot.get("chat", []):
            assert entry.get("speaker") == "轻松吐槽"

    def test_greet_at_start_has_speaker(self, opp_manager):
        """开局 greet 消息也带 speaker（对手身份贯穿全程）。"""
        session = opp_manager.start("moon_chess", "p_black", "easy", persona="banter")
        snapshot = session.snapshot()
        greet_entries = [e for e in snapshot["chat"] if e["scenario"] == "greet"]
        assert greet_entries
        assert greet_entries[0].get("speaker") == "轻松吐槽"

    def test_terminal_opponent_uses_ai_win_lose_game_over(self, opp_manager):
        """终局走 ai_win/ai_lose/game_over（对手视角）；弃牌局 revealed=False 不报玩家牌。"""
        session = opp_manager.start("texas_holdem", "p_sb", "easy", persona="banter")
        snapshot = opp_manager.move(session.game_id, {"choice": "fold"})
        assert snapshot["over"] is True
        scenarios = _drained_scenarios(snapshot)
        assert "ai_win" in scenarios  # 玩家弃牌 → AI 胜
        assert "game_over" in scenarios

    def test_say_uses_opponent_context_in_opponent_mode(self, opp_manager):
        """自由对话（say）在对手模式用 OpponentContext（adversarial scan）。"""
        session = opp_manager.start("moon_chess", "p_black", "easy", persona="banter")
        message = opp_manager.say(session.game_id, "help")
        assert message is not None
        assert message["text"]  # 兜底台词非空
        assert message["speaker"] == "轻松吐槽"

    def test_coach_path_unchanged_in_teaching(self, opp_manager):
        """对手模式不污染教学路径：teaching 仍走 teach_* 场景、speaker 带教练前缀。"""
        session = opp_manager.start("moon_chess", "p_black", "easy", persona="teacher", teaching=True)
        snapshot = session.snapshot()
        scenarios = _drained_scenarios(snapshot)
        assert "teach_greet" in scenarios
        # 教学路径不产生对手场景
        assert "opp_react" not in scenarios
        assert "opp_read" not in scenarios
        speakers = _drained_speakers(snapshot)
        assert all(s == "教练 · 认真教学" for s in speakers if s)


# ── Multi-player cheerleader fallback (P1 preserves old behavior) ────


class TestMultiPlayerCheerleaderFallback:
    def test_multi_player_does_not_queue_opp_scenarios(self, opp_manager):
        """多人非教练不走对手模式：move 后 pending_chat 不含 opp_*（啦啦队 fallback）。"""
        session = opp_manager.start("mahjong_guangdong", "p0", "easy", player_count=4)
        legal = session.snapshot()["legal"]
        tile = next(action["tile"] for action in legal if action["type"] == "discard")
        snapshot = opp_manager.move(session.game_id, {"type": "discard", "tile": tile})
        scenarios = _drained_scenarios(snapshot)
        # 麻将评估恒中性（C4），blunder/good_move 都不命中 → 除非终局，move 后无 chat
        for s in scenarios:
            assert not s.startswith("opp_"), f"多人局不应出现对手场景: {s}"
