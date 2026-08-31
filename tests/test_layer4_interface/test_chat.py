"""Tests for the chat-first orchestrator (layer4_interface/frontend/platform/chat.py).

Covers the intent contract shared with the frontend:

- deterministic fallback routing (no LLM): play-by-name, play-clarify with
  chips, resume, grid text moves, history/review/create/settings/platform/
  benchmark/learning/help, default chat, plus specific-feature questions
  (“怎么改难度/教学对局是什么/视觉识别怎么用”) → 主题文档 help；
- LLM + function-calling path: ``play_game`` / ``make_move`` tool calls
  mapped and *validated against the engine contract* (invalid actions →
  clarify, never an exception), and ``get_platform_help`` info-tool loop
  (具体功能提问先取权威主题文档再作答);
- ``build_tools`` shape (make_move present only with a live session);
- empty input → help text.

No real network: the fake LLM returns ``ChatReply`` directly.
"""

from __future__ import annotations

import pytest

from layer2_engine.core.llm import ChatReply, StreamChunk, ToolCall
from layer4_interface.frontend.platform.chat import (
    build_tools,
    chat_turn,
    chat_turn_stream,
    fallback_intent,
)
from layer4_interface.frontend.platform.game_knowledge import game_knowledge_text
from layer4_interface.frontend.platform.history import MatchHistory
from layer4_interface.frontend.platform.platform_knowledge import (
    PLATFORM_TOPIC_KEYS,
    PLATFORM_TOPICS,
    match_platform_topic,
    platform_help_index,
    platform_help_text,
)
from layer4_interface.frontend.platform.session import PlayManager
from train_cli import default_provider


@pytest.fixture
def manager(tmp_path) -> PlayManager:
    return PlayManager(provider=default_provider, history=MatchHistory(tmp_path), seed=42)


class _FakeLLM:
    """Stands in for ``LLMClient.complete_tools`` (no transport)."""

    def __init__(self, *, text: str = "", tool_calls: tuple[tuple[str, dict], ...] = ()) -> None:
        self._text = text
        self._tool_calls = [ToolCall(name, args) for name, args in tool_calls]

    def complete_tools(self, messages: list[dict], tools: list[dict], **_: object) -> ChatReply:
        return ChatReply(text=self._text, tool_calls=self._tool_calls)


class _RecordingLLM(_FakeLLM):
    """Fake LLM that records the exact ``messages`` it was handed."""

    def __init__(self, *, text: str = "") -> None:
        super().__init__(text=text)
        self.seen: list[dict] = []

    def complete_tools(self, messages: list[dict], tools: list[dict], **_: object) -> ChatReply:
        self.seen = list(messages)
        return super().complete_tools(messages, tools, **_)


class _ScriptedLLM:
    """Fake LLM playing a fixed reply script; records every round's messages.

    Models the agentic tool loop: each ``complete_tools`` call pops the
    next scripted ``ChatReply`` so a test can simulate "model asks for a
    tool → gets the ``role: 'tool'`` result → answers from it".
    """

    def __init__(self, replies: list[ChatReply]) -> None:
        self._replies = list(replies)
        self.seen: list[list[dict]] = []

    def complete_tools(self, messages: list[dict], tools: list[dict], **_: object) -> ChatReply:
        self.seen.append(list(messages))
        if not self._replies:
            return ChatReply(text="")
        return self._replies.pop(0)


# ── Deterministic fallback (no LLM) ───────────────────────────────


class TestFallbackIntent:
    def test_play_by_name(self, manager: PlayManager) -> None:
        result = fallback_intent("我想玩月亮棋", _games(manager), None)
        assert result.intent == "play"
        assert result.params["game_id"] == "moon_chess"

    def test_play_without_name_clarifies(self, manager: PlayManager) -> None:
        result = fallback_intent("来一局", _games(manager), None)
        assert result.intent == "clarify"
        assert result.params.get("chips")

    def test_history_and_review(self, manager: PlayManager) -> None:
        assert fallback_intent("看看我的战绩", _games(manager), None).intent == "history"
        assert fallback_intent("复盘一下上一局", _games(manager), None).intent == "review"

    def test_create_settings_platform(self, manager: PlayManager) -> None:
        assert fallback_intent("创建一个新游戏", _games(manager), None).intent == "create"
        assert fallback_intent("打开设置", _games(manager), None).intent == "settings"
        assert fallback_intent("打开平台界面", _games(manager), None).intent == "platform"

    def test_resume_with_active_session(self, manager: PlayManager) -> None:
        session = manager.start("moon_chess", "p_black", "easy")
        result = fallback_intent("继续上一局", _games(manager), session)
        assert result.intent == "resume"
        assert result.params["game_id"] == session.game_id

    def test_grid_text_move(self, manager: PlayManager) -> None:
        session = manager.start("moon_chess", "p_black", "easy")
        result = fallback_intent("我下第2行第1列", _games(manager), session)
        assert result.intent == "move"
        assert result.params["action"] == {"cell_index": 3}

    def test_grid_text_move_invalid_clarifies(self, manager: PlayManager) -> None:
        session = manager.start("moon_chess", "p_black", "easy")
        result = fallback_intent("下第9行第9列", _games(manager), session)
        assert result.intent == "clarify"

    def test_gomoku_grid_text_move_uses_board_size(self, manager: PlayManager) -> None:
        """P2-18 回归：随机五子棋是 9×9，文字落子必须按 spec.board_size 解析。

        修复前 ``GRID_BOARD_LEN`` 硬编码 stochastic_gomoku=15（且字典优先于
        spec）—— "下第2行第3列" 落到 (2-1)*15+2=17 格（9×9 盘上越界 → 澄清），
        10-15 行的落子又被正则接受后再判非法。
        """
        session = manager.start("stochastic_gomoku", "p_black", "easy")
        result = fallback_intent("我下第2行第3列", _games(manager), session)
        assert result.intent == "move"
        assert result.params["action"] == {"cell_index": (2 - 1) * 9 + (3 - 1)}
        # 9×9 之外必须澄清（旧 15×15 解析会静默接受越界行号）
        assert fallback_intent("下第10行第1列", _games(manager), session).intent == "clarify"

    def test_gomoku_center_move_in_range(self, manager: PlayManager) -> None:
        """9×9 的天元是第 41 格（5,5）；15×15 的中心是 113 格 —— 修复前
        "下中间" 会落到 113 直接越界。"""
        session = manager.start("stochastic_gomoku", "p_black", "easy")
        result = fallback_intent("下中间", _games(manager), session)
        assert result.intent == "move"
        assert result.params["action"] == {"cell_index": 40}

    def test_help_and_default_chat(self, manager: PlayManager) -> None:
        assert fallback_intent("你能做什么", _games(manager), None).intent == "help"
        assert fallback_intent("你好呀", _games(manager), None).intent == "chat"

    def test_help_topic_routes_specific_feature_questions(self, manager: PlayManager) -> None:
        """具体功能怎么用（不命中动作面板关键词）→ 主题文档确定性回答。

        旧逻辑这些提问落到泛泛默认 chat；现在按最长关键词命中路由到
        ``platform_knowledge`` 的权威主题文档（与 ``get_platform_help``
        同一事实来源）。
        """
        r = fallback_intent("怎么改难度", _games(manager), None)
        assert r.intent == "help"
        assert r.params["topic"] == "settings"
        assert "难度" in r.text
        r2 = fallback_intent("教学对局是什么", _games(manager), None)
        assert r2.intent == "help"
        assert r2.params["topic"] == "teaching"
        assert "教练" in r2.text
        r3 = fallback_intent("视觉识别怎么用", _games(manager), None)
        assert r3.intent == "help"
        assert r3.params["topic"] == "vision"
        assert "视觉识别" in r3.text
        r4 = fallback_intent("LLM 模型在哪配置", _games(manager), None)
        assert r4.intent == "help"
        assert r4.params["topic"] == "llm"
        assert "密钥" in r4.text

    def test_help_generic_keeps_overview_text(self, manager: PlayManager) -> None:
        """泛泛“你能做什么”不命中任何主题关键词 → 维持原总览帮助文案。"""
        r = fallback_intent("你能做什么", _games(manager), None)
        assert r.intent == "help"
        assert "topic" not in r.params
        assert "玩月亮棋" in r.text

    def test_what_is_game_answers_from_registry(self, manager: PlayManager) -> None:
        """“月亮棋是什么？” —— 无 LLM 也有确定性回答（零幻觉路径）。"""
        result = fallback_intent("月亮棋是什么", _games(manager), None)
        assert result.intent == "chat"
        assert result.params["game_id"] == "moon_chess"
        assert "月亮棋" in result.text
        assert "3×3" in result.text  # GameSpec.description / 玩法文档的权威内容
        assert result.params.get("chips")

    def test_what_is_beats_play_verb(self, manager: PlayManager) -> None:
        """“怎么下/怎么打”含开局动词，语义却是问规则 —— 知识分支优先于 play。"""
        result = fallback_intent("月亮棋怎么下", _games(manager), None)
        assert result.intent == "chat"
        assert "3×3" in result.text

    def test_what_is_without_game_falls_through(self, manager: PlayManager) -> None:
        """问句不点名任何已注册游戏 → 维持原兜底（不猜、不编）。"""
        result = fallback_intent("围棋是什么", _games(manager), None)
        assert result.intent == "chat"
        assert "3×3" not in result.text

    def test_uno_short_name_matches_via_alias(self, manager: PlayManager) -> None:
        """短名匹配（audit §5-5）：“UNO 的规则”接不住 display_name
        「UNO（经典）」—— 别名表 + 大小写不敏感补上；play 分支同样受益。"""
        games = _games(manager)
        result = fallback_intent("UNO的规则", games, None)
        assert result.intent == "chat"
        assert result.params["game_id"] == "uno"
        assert "108" in result.text  # 来自注册表 description 的权威内容
        assert result.params["chips"] == ["玩UNO（经典）"]
        # 大小写不敏感：“玩uno” 同样开局（game_id 子串大小写旧逻辑接不住）
        assert fallback_intent("玩uno", games, None).params["game_id"] == "uno"

    def test_uno_variant_longest_alias_wins(self, manager: PlayManager) -> None:
        """“UNO 7-0” 同时命中 uno 的 "UNO" 与 seven_zero 的 "UNO 7-0" —— 最长匹配胜出。"""
        result = fallback_intent("UNO 7-0怎么玩", _games(manager), None)
        assert result.params["game_id"] == "uno_seven_zero"

    def test_alias_matches_play_branch(self, manager: PlayManager) -> None:
        """别名不止服务知识问句：“来一局德扑”（德州别名）同样开局。"""
        result = fallback_intent("来一局德扑", _games(manager), None)
        assert result.intent == "play"
        assert result.params["game_id"] == "texas_holdem"


# ── LLM + function calling ────────────────────────────────────────


class TestChatTurnLLM:
    def test_play_tool_call(self, manager: PlayManager) -> None:
        fake = _FakeLLM(text="好，来一局德州扑克！", tool_calls=(("play_game", {"game_id": "texas_holdem"}),))
        result = chat_turn(manager, "我想玩德州扑克", llm=fake)
        assert result.intent == "play"
        assert result.params["game_id"] == "texas_holdem"

    def test_play_unknown_game_clarifies(self, manager: PlayManager) -> None:
        fake = _FakeLLM(tool_calls=(("play_game", {"game_id": "no_such_game"}),))
        result = chat_turn(manager, "玩神秘游戏", llm=fake)
        assert result.intent == "clarify"
        assert result.params.get("chips")

    def test_make_move_valid(self, manager: PlayManager) -> None:
        session = manager.start("moon_chess", "p_black", "easy")
        fake = _FakeLLM(tool_calls=(("make_move", {"action": {"cell_index": 0}}),))
        result = chat_turn(manager, "我下左上角", llm=fake, game_id=session.game_id)
        assert result.intent == "move"
        assert result.params["action"] == {"cell_index": 0}

    def test_make_move_invalid_clarifies(self, manager: PlayManager) -> None:
        session = manager.start("moon_chess", "p_black", "easy")
        fake = _FakeLLM(tool_calls=(("make_move", {"action": {"cell_index": 99}}),))
        result = chat_turn(manager, "下第10个位置", llm=fake, game_id=session.game_id)
        assert result.intent == "clarify"

    def test_llm_text_only_chat(self, manager: PlayManager) -> None:
        fake = _FakeLLM(text="我在的，你想先玩哪款？")
        result = chat_turn(manager, "在吗", llm=fake)
        assert result.intent == "chat"
        assert "先玩哪款" in result.text

    def test_unknown_tool_falls_to_chat(self, manager: PlayManager) -> None:
        fake = _FakeLLM(tool_calls=(("mystery_tool", {}),))
        result = chat_turn(manager, "随便", llm=fake)
        assert result.intent == "chat"

    def test_resume_tool_call(self, manager: PlayManager) -> None:
        session = manager.start("moon_chess", "p_black", "easy")
        fake = _FakeLLM(tool_calls=(("resume_session", {}),))
        result = chat_turn(manager, "继续", llm=fake, game_id=session.game_id)
        assert result.intent == "resume"


# ── Info tools (describe_game / list_games) ────────────────────────


class TestInfoTools:
    """知识类问题的 function-calling 路径：工具在本地执行、结果以
    ``role:"tool"`` 回传，模型基于权威资料作答（而不是幻觉）。"""

    def test_build_tools_always_exposes_info_tools(self, manager: PlayManager) -> None:
        names = [t["function"]["name"] for t in build_tools(games=_games(manager), session=None, active=[])]
        assert "describe_game" in names
        assert "list_games" in names
        assert "get_platform_help" in names  # 具体功能帮助工具常驻暴露

    def test_describe_game_tool_loop(self, manager: PlayManager) -> None:
        fake = _ScriptedLLM(
            [
                ChatReply(text="", tool_calls=[ToolCall("describe_game", {"game_id": "moon_chess"}, id="call_1")]),
                ChatReply(text="月亮棋是 3×3 的连线棋：每方最多 3 子，最老的会被挤出，三连即胜。"),
            ]
        )
        result = chat_turn(manager, "月亮棋是什么？", llm=fake)
        assert result.intent == "chat"
        assert "3×3" in result.text
        # 第二轮：模型必须收到 role:"tool" 的权威资料（关联 call_id）
        assert len(fake.seen) == 2
        first, second = fake.seen
        assert first[-1] == {"role": "user", "content": "月亮棋是什么？"}
        tool_msgs = [m for m in second if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert "3×3" in tool_msgs[0]["content"]  # 来自 GameSpec.description / 玩法文档
        # 发起调用的 assistant 消息成对出现在 tool 消息之前
        assistant_idx = next(i for i, m in enumerate(second) if m.get("role") == "assistant" and m.get("tool_calls"))
        assert second[assistant_idx]["tool_calls"][0]["id"] == "call_1"
        assert second[assistant_idx]["tool_calls"][0]["function"]["name"] == "describe_game"
        assert second[assistant_idx + 1] is tool_msgs[0]

    def test_list_games_tool_loop(self, manager: PlayManager) -> None:
        fake = _ScriptedLLM(
            [
                ChatReply(text="", tool_calls=[ToolCall("list_games", {})]),
                ChatReply(text="平台有月亮棋、随机五子棋和德州扑克，详见列表。"),
            ]
        )
        result = chat_turn(manager, "你们平台都有什么游戏？", llm=fake)
        assert result.intent == "chat"
        tool_msgs = [m for m in fake.seen[1] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "月亮棋" in tool_msgs[0]["content"]
        assert "德州扑克" in tool_msgs[0]["content"]

    def test_info_tool_budget_exhausted_answers_deterministically(self, manager: PlayManager) -> None:
        """模型一直要查资料（预算耗尽）→ 用工具执行结果直接作答，绝不空转。"""
        looping = ChatReply(text="", tool_calls=[ToolCall("describe_game", {"game_id": "moon_chess"})])
        fake = _ScriptedLLM([looping, looping, looping])
        result = chat_turn(manager, "月亮棋是什么？", llm=fake)
        assert result.intent == "chat"
        assert "3×3" in result.text
        assert len(fake.seen) == 3  # _MAX_TOOL_ROUNDS 上限生效

    def test_system_prompt_carries_game_descriptions(self, manager: PlayManager) -> None:
        """上下文注入：系统提示的游戏目录必须带一句话简介（防模型瞎猜）。"""
        fake = _RecordingLLM(text="嗯。")
        chat_turn(manager, "在吗", llm=fake)
        system = fake.seen[0]["content"]
        assert "月亮棋" in system
        assert "三子连珠" in system  # moon_chess 的 description 关键词
        assert "知识红线" in system

    def test_parallel_info_calls_all_executed_and_fed_back(self, manager: PlayManager) -> None:
        """并行 tool_calls（audit §5-3）：一个回合并行发起的两个信息查询
        都就地执行、都以 ``role:"tool"`` 回传（按 tool_call_id 成对关联），
        不再只取 ``tool_calls[0]``。"""
        fake = _ScriptedLLM(
            [
                ChatReply(
                    text="",
                    tool_calls=[
                        ToolCall("describe_game", {"game_id": "moon_chess"}, id="call_a"),
                        ToolCall("describe_game", {"game_id": "texas_holdem"}, id="call_b"),
                    ],
                ),
                ChatReply(text="月亮棋是 3×3 连线棋；德州是双人扑克。"),
            ]
        )
        result = chat_turn(manager, "月亮棋和德州扑克分别是什么？", llm=fake)
        assert result.intent == "chat"
        assert "3×3" in result.text
        second = fake.seen[1]
        tool_msgs = [m for m in second if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert {m["tool_call_id"] for m in tool_msgs} == {"call_a", "call_b"}
        joined = "\n".join(str(m["content"]) for m in tool_msgs)
        assert "3×3" in joined  # moon_chess 资料
        assert "四轮下注" in joined  # texas_holdem 资料
        # 发起方 assistant 消息携带全部两个 tool_calls（OpenAI 协议形态）
        assistant = next(m for m in second if m.get("role") == "assistant" and m.get("tool_calls"))
        assert [c["id"] for c in assistant["tool_calls"]] == ["call_a", "call_b"]

    def test_parallel_mixed_batch_action_wins(self, manager: PlayManager) -> None:
        """混合批次（信息 + 动作）：动作类只取首个立即映射 intent——
        旧逻辑只看 ``tool_calls[0]``，信息查询排在前时动作被静默丢弃。"""
        fake = _ScriptedLLM(
            [
                ChatReply(
                    text="",
                    tool_calls=[
                        ToolCall("describe_game", {"game_id": "moon_chess"}),
                        ToolCall("play_game", {"game_id": "texas_holdem"}),
                    ],
                ),
            ]
        )
        result = chat_turn(manager, "介绍一下月亮棋，然后来一局德州", llm=fake)
        assert result.intent == "play"
        assert result.params["game_id"] == "texas_holdem"
        assert len(fake.seen) == 1  # 动作立即返回，没有第二轮取数

    def test_platform_help_tool_loop(self, manager: PlayManager) -> None:
        """具体功能提问（LLM 路径）：模型取主题文档再作答，不再泛泛总览。

        旧 ``help`` 工具只回一段总览，面对“在线学习怎么用”这类具体提问
        只能泛泛而谈或编造；``get_platform_help`` 与 ``describe_game``
        同一条信息工具路径：本地执行 + ``role:"tool"`` 回传权威文档。
        """
        fake = _ScriptedLLM(
            [
                ChatReply(text="", tool_calls=[ToolCall("get_platform_help", {"topic": "learning"}, id="call_help")]),
                ChatReply(text="在线学习会收集人机对局里人类的决策，门禁通过后发布给 AI。"),
            ]
        )
        result = chat_turn(manager, "在线学习怎么用？", llm=fake)
        assert result.intent == "chat"
        assert "在线学习" in result.text
        second = fake.seen[1]
        tool_msgs = [m for m in second if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_help"
        assert "门禁" in tool_msgs[0]["content"]  # 主题文档的权威要点
        assert "发布" in tool_msgs[0]["content"]
        assistant = next(m for m in second if m.get("role") == "assistant" and m.get("tool_calls"))
        assert assistant["tool_calls"][0]["function"]["name"] == "get_platform_help"

    def test_platform_help_unknown_topic_returns_index(self, manager: PlayManager) -> None:
        """topic 缺省/未知 → 主题总览（fail-soft），模型可据此继续提问。"""
        fake = _ScriptedLLM(
            [
                ChatReply(text="", tool_calls=[ToolCall("get_platform_help", {"topic": "xx"}, id="call_1")]),
                ChatReply(text="平台功能挺多，我再具体问一下某项怎么用。"),
            ]
        )
        result = chat_turn(manager, "平台都有什么功能？", llm=fake)
        assert result.intent == "chat"
        tool_msgs = [m for m in fake.seen[1] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "在线学习" in tool_msgs[0]["content"]  # 总览索引含各主题


# ── Shared knowledge assembly (game_knowledge) ─────────────────────


class TestGameKnowledge:
    """chat 信息工具与陪伴对话注入共用的资料拼装（单一事实来源）。"""

    def test_builtin_game_assembles_description_and_rules(self) -> None:
        text = game_knowledge_text("moon_chess")
        assert "月亮棋" in text
        assert "3×3" in text  # GameSpec.description
        assert "规则要点" in text  # docs/user/play_moon_chess.md 规则段

    def test_unknown_game_returns_empty(self) -> None:
        """未知 / custom 游戏返回空串 —— 调用方各自 fail-soft。"""
        assert game_knowledge_text("no_such_game") == ""
        assert game_knowledge_text("") == ""


# ── Shared platform help (platform_knowledge) ──────────────────────


class TestPlatformKnowledge:
    """具体功能帮助文档的单一事实来源（``get_platform_help`` / 兜底共用）。"""

    def test_index_covers_all_topic_keys(self) -> None:
        index = platform_help_index()
        assert "在线学习" in index
        assert "创建自定义游戏" in index
        assert "教学对局 / 教练" in index
        assert "视觉识别" in index
        for key in PLATFORM_TOPIC_KEYS:
            topic = next(t for t in PLATFORM_TOPICS if t.key == key)
            assert topic.title in index or topic.title.split(" / ")[0] in index

    def test_topic_text_returns_authoritative_doc(self) -> None:
        text = platform_help_text("learning")
        assert "门禁" in text
        assert "发布" in text
        assert "在线学习" in text
        assert platform_help_text("no_such_topic") == ""
        assert platform_help_text("") == ""

    def test_match_platform_topic_longest_keyword_wins(self) -> None:
        assert match_platform_topic("在线学习功能在哪") == "learning"
        assert match_platform_topic("怎么改难度") == "settings"
        assert match_platform_topic("教学对局是什么") == "teaching"
        assert match_platform_topic("视觉识别怎么用") == "vision"
        assert match_platform_topic("LLM 模型在哪配置") == "llm"
        # 泛泛帮助提问不命中任何具体主题（保持原总览文案）
        assert match_platform_topic("你能做什么") is None
        assert match_platform_topic("你好呀") is None


# ── In-match / post-match info tools (对局信息源修复) ───────────────


def _recorded_moon_match(history: MatchHistory) -> str:
    """写入一局已结束的月亮棋记录（人类 p_black 落败），返回 match_id。"""
    boards = [
        ["p_black", None, None, None, None, None, None, None, None],
        ["p_black", None, "p_white", None, None, None, None, None, None],
        ["p_black", "p_black", "p_white", None, None, None, None, None, None],
        ["p_black", "p_black", "p_white", None, None, "p_white", None, None, None],
        ["p_black", "p_black", "p_white", "p_black", None, "p_white", None, None, None],
        ["p_black", "p_black", "p_white", "p_black", None, "p_white", None, None, "p_white"],
    ]
    moves = []
    for i, board in enumerate(boards):
        over = i == len(boards) - 1
        moves.append(
            {
                "step": i,
                "actor": "human" if i % 2 == 0 else "ai",
                "action": f"cell_{i}",
                "snapshot": {
                    "player_pid": "p_black",
                    "board": board,
                    "turn": "p_white" if i % 2 == 0 else "p_black",
                    "winner": "p_white" if over else None,
                    "over": over,
                    "round": i + 1,
                },
            }
        )
    return history.record(
        {
            "match_id": "mtest0001",
            "game_id": "moon_chess",
            "player_pid": "p_black",
            "ai_pid": "p_white",
            "difficulty": "easy",
            "winner": "p_white",
            "over": True,
            "moves": moves,
        }
    )


@pytest.fixture
def manager_history(tmp_path) -> tuple[PlayManager, MatchHistory]:
    history = MatchHistory(tmp_path)
    return PlayManager(provider=default_provider, history=history, seed=42), history


class TestMatchStateTools:
    """对局中信息源（get_match_state）与提示回流（ask_hint）。"""

    def test_get_match_state_feeds_board_layout(self, manager: PlayManager) -> None:
        """对局信息源：模型能拉到棋盘布局（含占据方），不再反问用户描述棋盘。"""
        session = manager.start("moon_chess", "p_black", "easy")
        manager.move(session.game_id, {"cell_index": 4})  # 双方各落一子
        fake = _ScriptedLLM(
            [
                ChatReply(text="", tool_calls=[ToolCall("get_match_state", {}, id="c1")]),
                ChatReply(text="中心点已经被你占住了，接下来注意边路。"),
            ]
        )
        result = chat_turn(manager, "现在局面怎么样", llm=fake, game_id=session.game_id)
        assert result.intent == "chat"
        assert result.text == "中心点已经被你占住了，接下来注意边路。"
        tool_msgs = [m for m in fake.seen[1] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        content = tool_msgs[0]["content"]
        assert "棋盘" in content
        assert "你" in content and "AI" in content  # 占据布局可见（旧版只有空位索引）

    def test_get_match_state_mahjong_hand_visible(self, manager: PlayManager) -> None:
        """各族投影字段直达模型：麻将的 my_hand（玩家自己的手牌）不再被红线误伤。"""
        session = manager.start("mahjong_guangdong", "p0", "easy")
        fake = _ScriptedLLM(
            [
                ChatReply(text="", tool_calls=[ToolCall("get_match_state", {})]),
                ChatReply(text="你的手牌如上，建议先拆边张。"),
            ]
        )
        result = chat_turn(manager, "我手上有什么牌", llm=fake, game_id=session.game_id)
        assert result.intent == "chat"
        tool_msgs = [m for m in fake.seen[1] if m.get("role") == "tool"]
        assert "my_hand" in tool_msgs[0]["content"]

    def test_ask_hint_carries_level_and_hint(self, manager: PlayManager) -> None:
        """提示回流：level 透传、机械提示全文进 ``role:"tool"``，模型成文讲解。"""
        session = manager.start("moon_chess", "p_black", "easy")
        fake = _ScriptedLLM(
            [
                ChatReply(text="", tool_calls=[ToolCall("ask_hint", {"level": "direction"}, id="c1")]),
                ChatReply(text="这一步建议先占中心，控制两条线。"),
            ]
        )
        result = chat_turn(manager, "这步怎么走", llm=fake, game_id=session.game_id)
        assert result.intent == "hint"
        assert result.params["level"] == "direction"
        assert isinstance(result.params["hint"], dict) and result.params["hint"]
        assert result.text == "这一步建议先占中心，控制两条线。"
        tool_msgs = [m for m in fake.seen[1] if m.get("role") == "tool"]
        assert "机械提示" in tool_msgs[0]["content"]

    def test_ask_hint_budget_exhausted_returns_mechanical_hint(self, manager: PlayManager) -> None:
        """预算耗尽也落在 hint 意图上（带 params），不退回纯 chat 丢参数。"""
        session = manager.start("moon_chess", "p_black", "easy")
        looping = ChatReply(text="", tool_calls=[ToolCall("ask_hint", {"level": "direction"})])
        fake = _ScriptedLLM([looping, looping, looping])
        result = chat_turn(manager, "这步怎么走", llm=fake, game_id=session.game_id)
        assert result.intent == "hint"
        assert result.params["level"] == "direction"
        assert result.text  # 机械提示文本兜底


class TestMatchReviewTools:
    """赛后复盘（get_match_review）：时间线 + 关键节点喂给模型讲解。"""

    def test_get_match_review_narrates_with_report(self, manager_history) -> None:
        manager, history = manager_history
        match_id = _recorded_moon_match(history)
        fake = _ScriptedLLM(
            [
                ChatReply(text="", tool_calls=[ToolCall("get_match_review", {}, id="c1")]),
                ChatReply(text="这局的关键在第 3 手：你在边路落子后局势开始倾斜。"),
            ]
        )
        result = chat_turn(manager, "复盘一下关键手和失误点", llm=fake, match_history=history)
        assert result.intent == "review"
        assert result.params["match_id"] == match_id
        report = result.params["report"]
        assert report["key_nodes"]
        assert report["key_nodes"][0]["what"]  # 动作内容进了报告（不再是纯序号）
        assert result.text == "这局的关键在第 3 手：你在边路落子后局势开始倾斜。"
        tool_msgs = [m for m in fake.seen[1] if m.get("role") == "tool"]
        content = tool_msgs[0]["content"]
        assert "走子时间线" in content
        assert "你:" in content  # 动作时间线（moves[].action）

    def test_get_match_review_without_history_answers_deterministically(self, manager_history) -> None:
        manager, history = manager_history  # 空历史
        fake = _ScriptedLLM(
            [
                ChatReply(text="", tool_calls=[ToolCall("get_match_review", {})]),
                ChatReply(text="还没有打完的局，先来一局吧。"),
            ]
        )
        result = chat_turn(manager, "复盘一下", llm=fake, match_history=history)
        assert result.intent == "chat"
        tool_msgs = [m for m in fake.seen[1] if m.get("role") == "tool"]
        assert "还没有已结束的对局记录" in tool_msgs[0]["content"]

    def test_system_prompt_mentions_latest_match(self, manager_history) -> None:
        """终局后 session 移除 → system prompt 用最近一局补上下文（不再一片空白）。"""
        manager, history = manager_history
        _recorded_moon_match(history)
        fake = _RecordingLLM(text="嗯")
        chat_turn(manager, "在吗", llm=fake, match_history=history)
        system = fake.seen[0]["content"]
        assert "最近一局" in system
        assert "get_match_review" in system

    def test_build_tools_gates_state_tool_on_session(self, manager: PlayManager) -> None:
        games = _games(manager)
        names = [t["function"]["name"] for t in build_tools(games=games, session=None, active=[])]
        assert "get_match_review" in names  # 复盘资料工具常驻（历史驱动）
        assert "get_match_state" not in names
        assert "ask_hint" not in names
        session = manager.start("moon_chess", "p_black", "easy")
        names = [t["function"]["name"] for t in build_tools(games=games, session=session, active=[])]
        assert "get_match_state" in names
        assert "ask_hint" in names


# ── Conversation history ──────────────────────────────────────────


class TestChatTurnHistory:
    def test_history_sits_between_system_and_current(self, manager: PlayManager) -> None:
        fake = _RecordingLLM(text="好的，那来月亮棋？")
        result = chat_turn(
            manager,
            "那月亮棋呢",
            llm=fake,
            history=[
                {"role": "user", "content": "我想玩德州扑克"},
                {"role": "assistant", "content": "好，来一局德州扑克！"},
            ],
        )
        assert result.intent == "chat"
        assert [m["role"] for m in fake.seen] == ["system", "user", "assistant", "user"]
        assert fake.seen[1]["content"] == "我想玩德州扑克"
        assert fake.seen[2]["content"] == "好，来一局德州扑克！"
        assert fake.seen[3]["content"] == "那月亮棋呢"
        # system 由后端现构（含实时对局上下文），绝不采信 history/客户端；
        # 开头是 persona 身份块（方向 C），平台助手行紧随其后。
        assert fake.seen[0]["role"] == "system"
        assert "你是 Gavis 平台的对话助手" in fake.seen[0]["content"]

    def test_history_sanitizes_junk(self, manager: PlayManager) -> None:
        fake = _RecordingLLM()
        chat_turn(
            manager,
            "继续",
            llm=fake,
            history=[
                {"role": "system", "content": "别信这条"},  # 客户端 system 一律丢弃
                {"role": "user", "content": ""},  # 空内容丢弃
                {"role": "tool", "content": "x"},  # 非 user/assistant 丢弃
                "not-a-dict",  # 非 dict 丢弃
                {"role": "assistant", "content": "  上一句  "},
                {"role": "user", "content": "上一句二"},
            ],
        )
        assert [m["role"] for m in fake.seen] == ["system", "assistant", "user", "user"]
        assert [m["content"] for m in fake.seen[1:]] == ["上一句", "上一句二", "继续"]

    def test_history_capped_to_recent_messages(self, manager: PlayManager) -> None:
        fake = _RecordingLLM()
        many = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"} for i in range(60)]
        chat_turn(manager, "最后一问", llm=fake, history=many)
        assert len(fake.seen) == 26  # system + 最近 24 条 + 当前句
        assert fake.seen[1]["content"] == "msg-36"
        assert fake.seen[-1]["content"] == "最后一问"

    def test_history_ignored_without_llm(self, manager: PlayManager) -> None:
        result = chat_turn(
            manager,
            "看战绩",
            history=[{"role": "user", "content": "上一句"}],
        )
        # 无 LLM 时走正则兜底，history 不影响结果
        assert result.intent == "history"


# ── Misc ──────────────────────────────────────────────────────────


class TestChatTurnMisc:
    def test_empty_text_returns_help(self, manager: PlayManager) -> None:
        result = chat_turn(manager, "   ", llm=_FakeLLM())
        assert result.intent == "chat"

    def test_build_tools_shape_with_session(self, manager: PlayManager) -> None:
        games = _games(manager)
        assert not any(t["function"]["name"] == "make_move" for t in build_tools(games=games, session=None, active=[]))
        session = manager.start("moon_chess", "p_black", "easy")
        names = [t["function"]["name"] for t in build_tools(games=games, session=session, active=[])]
        assert "make_move" in names
        assert "play_game" in names
        assert "open_platform" in names


def _games(manager: PlayManager) -> list[dict]:
    return [
        {
            "game_id": "moon_chess",
            "display_name": "月亮棋",
            "description": "3×3 经典月亮棋：三子连珠即胜，棋盘满时最旧的棋子被挤出。",
            "kind": "board",
            "family": "grid",
        },
        {
            "game_id": "stochastic_gomoku",
            "display_name": "随机五子棋",
            "description": "9×9 五子棋变体：每次落子后棋子有 50% 概率被随机抹去。",
            "kind": "board",
            "family": "grid",
        },
        {
            "game_id": "texas_holdem",
            "display_name": "德州扑克",
            "description": "双人德州扑克：翻前/翻牌/转牌/河牌四轮下注，AI 使用混合求解器。",
            "kind": "poker",
            "family": "poker",
        },
        # UNO（display_name 与真实注册表一致——带括注/空格，短名匹配的
        # 问题形态；别名表见 game_knowledge.GAME_ALIASES）。
        {
            "game_id": "uno",
            "display_name": "UNO（经典）",
            "description": "四人经典 UNO：108 张牌，同色或同符号接牌，先清空手牌者胜。",
            "kind": "uno",
            "family": "uno",
        },
        {
            "game_id": "uno_seven_zero",
            "display_name": "UNO 7-0（换手/移交）",
            "description": "UNO 7-0 变体：打出 7 可与任一玩家换手，打出 0 全场手牌按方向移交。",
            "kind": "uno",
            "family": "uno",
        },
    ]


class _StreamFakeLLM:
    """流式假 LLM：跨轮按序弹出脚本化的 ``StreamChunk``（无网络）。"""

    def __init__(self, chunks: list[StreamChunk]) -> None:
        self._chunks = list(chunks)
        self.seen: list[list[dict]] = []

    def complete_stream(self, messages: list[dict], tools: list[dict], **_: object):
        self.seen.append(list(messages))
        while self._chunks:
            chunk = self._chunks.pop(0)
            yield chunk
            if chunk.done:
                return
        yield StreamChunk(done=True)


class _RaisingStreamLLM:
    """流式假 LLM：transport 中途抛错（fail-hard 路径）。"""

    def complete_stream(self, messages: list[dict], tools: list[dict], **_: object):
        raise RuntimeError("connection refused")


class _StrictStreamLLM:
    """流式假 LLM：签名与真实 ``LLMClient.complete_stream`` 一致 —— ``tools``
    是 keyword-only（真实客户端如此声明）。调用方若把 ``tools`` 当位置参数传，
    此处会立刻 ``TypeError``，正好复现线上「LLM 流式调用中断 → 正则兜底」的
    回归；同时把每次收到的 ``tools`` 记下来供断言。"""

    def __init__(self, chunks: list[StreamChunk]) -> None:
        self._chunks = list(chunks)
        self.tools_seen: list[list[dict] | None] = []

    def complete_stream(self, messages: list[dict], *, tools: list[dict] | None = None, **_: object):
        self.tools_seen.append(tools)
        while self._chunks:
            chunk = self._chunks.pop(0)
            yield chunk
            if chunk.done:
                return
        yield StreamChunk(done=True)


def _event_tuples(events: list[dict]) -> list[tuple[str, dict]]:
    return [(e["event"], e["data"]) for e in events]


class TestChatTurnStream:
    """chat_turn_stream（SSE 事件生成器）：增量正文/思维链、工具循环、失败兜底."""

    def test_text_deltas_then_intent(self, manager: PlayManager) -> None:
        fake = _StreamFakeLLM(
            [
                StreamChunk(text="你"),
                StreamChunk(text="好"),
                StreamChunk(text="，这步建议占中心。", done=True),
            ]
        )
        events = list(chat_turn_stream(manager, "我该怎么走？", llm=fake))
        tuples = _event_tuples(events)
        assert tuples[0] == ("text", {"delta": "你"})
        assert tuples[1] == ("text", {"delta": "好"})
        assert tuples[2][0] == "text"
        assert tuples[3][0] == "intent"
        assert tuples[3][1]["intent"] == "chat"
        assert tuples[3][1]["text"] == "你好，这步建议占中心。"
        assert tuples[4] == ("done", {})
        assert len(fake.seen) == 1

    def test_reasoning_deltas_stream(self, manager: PlayManager) -> None:
        fake = _StreamFakeLLM(
            [
                StreamChunk(reasoning="先看中心"),
                StreamChunk(reasoning="再判断危险"),
                StreamChunk(text="建议占中心。", done=True),
            ]
        )
        events = list(chat_turn_stream(manager, "我该怎么走？", llm=fake))
        tuples = _event_tuples(events)
        assert tuples[0] == ("reasoning", {"delta": "先看中心"})
        assert tuples[1] == ("reasoning", {"delta": "再判断危险"})
        assert tuples[2] == ("text", {"delta": "建议占中心。"})
        assert tuples[3][0] == "intent"
        assert tuples[3][1]["text"] == "建议占中心。"

    def test_action_tool_round_emits_intent_with_model_text(self, manager: PlayManager) -> None:
        fake = _StreamFakeLLM(
            [
                StreamChunk(
                    text="好，来一局月亮棋！",
                    tool_calls=[ToolCall("play_game", {"game_id": "moon_chess"})],
                    done=True,
                )
            ]
        )
        events = list(chat_turn_stream(manager, "我想玩月亮棋", llm=fake))
        tuples = _event_tuples(events)
        intent = tuples[-2][1]
        assert intent["intent"] == "play"
        assert intent["params"] == {"game_id": "moon_chess"}
        assert intent["text"] == "好，来一局月亮棋！"

    def test_info_tool_loop_feeds_back_then_answers(self, manager: PlayManager) -> None:
        fake = _StreamFakeLLM(
            [
                StreamChunk(tool_calls=[ToolCall("describe_game", {"game_id": "moon_chess"})], done=True),
                StreamChunk(text="月亮棋是双人策略棋盘游戏，落子占中心。", done=True),
            ]
        )
        events = list(chat_turn_stream(manager, "月亮棋是什么？", llm=fake))
        tuples = _event_tuples(events)
        assert len(fake.seen) == 2  # 工具调用轮 + 带 role:"tool" 结果的成文轮
        assert fake.seen[1][-1]["role"] == "tool"
        assert tuples[-2][0] == "intent"
        assert tuples[-2][1]["intent"] == "chat"
        assert "月亮棋是双人" in tuples[-2][1]["text"]

    def test_no_llm_falls_back_with_single_intent(self, manager: PlayManager) -> None:
        events = list(chat_turn_stream(manager, "我想玩月亮棋"))
        tuples = _event_tuples(events)
        assert tuples[0][0] == "intent"
        assert tuples[0][1]["intent"] == "play"
        assert tuples[0][1]["params"]["game_id"] == "moon_chess"
        assert tuples[1] == ("done", {})

    def test_error_chunk_then_fallback_intent(self, manager: PlayManager) -> None:
        fake = _StreamFakeLLM([StreamChunk(error="LLM 端点不可达/超时", done=True)])
        events = list(chat_turn_stream(manager, "我想玩月亮棋", llm=fake))
        tuples = _event_tuples(events)
        assert tuples[0][0] == "error"
        assert "不可达" in tuples[0][1]["error"]
        assert tuples[1][0] == "intent"  # 兜底 intent 仍给出（error 不中断流）
        assert tuples[-1] == ("done", {})

    def test_stream_exception_falls_back_with_error_event(self, manager: PlayManager) -> None:
        events = list(chat_turn_stream(manager, "我想玩月亮棋", llm=_RaisingStreamLLM()))
        tuples = _event_tuples(events)
        assert tuples[0][0] == "error"
        assert "流式" in tuples[0][1]["error"]  # last_error 空 → 通用文案
        assert tuples[1][0] == "intent"

    def test_stream_passes_tools_keyword_only(self, manager: PlayManager) -> None:
        """回归：``chat_turn_stream`` 必须用关键字传 ``tools``（真实客户端
        ``LLMClient.complete_stream`` 声明为 keyword-only）。

        曾以位置传参 → ``TypeError`` 立即抛出（尚未发任何 HTTP 请求，
        ``last_error`` 为空）→ 线上表现为「LLM 流式调用中断 → 正则兜底」。
        用与真实签名一致的假 LLM 复现该调用形态，确保增量正文正常上浮。
        """
        fake = _StrictStreamLLM([StreamChunk(text="你好，我在！", done=True)])
        events = list(chat_turn_stream(manager, "你好", llm=fake))
        tuples = _event_tuples(events)
        assert len(fake.tools_seen) == 1  # 每轮都以 keyword 收到工具
        assert isinstance(fake.tools_seen[0], list) and fake.tools_seen[0]  # 非 None 且非空
        assert tuples[0] == ("text", {"delta": "你好，我在！"})
        assert tuples[-2][0] == "intent"
        assert tuples[-2][1]["intent"] == "chat"
        assert tuples[-2][1]["text"] == "你好，我在！"
        assert tuples[-1] == ("done", {})
