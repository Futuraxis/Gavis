"""陪伴发言异步化：走子请求不再被陪伴 Agent 的 LLM 调用拖住。

实测（真实平台 + 云端推理模型）每次走子 16-33s **全部**花在陪伴 Agent 的
两次成文调用上（对手反应 + 读人）：玩家点一下等半分钟，体感就是「创建好了
玩不了」。``agent_async=True`` 时成文在后台线程完成，走子立刻返回，发言由
前端轮询 ``/api/match/state`` 收取（快照的 ``chat`` 增量）。

默认 ``agent_async=False`` 保持既有同步契约（测试与非平台调用方）。
"""

from __future__ import annotations

import time

from layer4_interface.agent import PERSONAS, DialogueEngine
from layer4_interface.frontend.platform.history import MatchHistory
from layer4_interface.frontend.platform.session import PlayManager
from train_cli import default_provider


class _SlowAgent:
    """陪伴 Agent 双：按 ``delay`` 模拟一次 LLM 成文的耗时。"""

    def __init__(self, delay: float = 0.3) -> None:
        self.delay = delay
        self.persona = PERSONAS["gentle"]
        self.calls = 0
        self._engine = DialogueEngine(PERSONAS["gentle"])

    def reply(self, ctx, scenario, game_id=None):  # noqa: ANN001, ANN202
        self.calls += 1
        time.sleep(self.delay)
        return self._engine.reply(ctx, scenario, game_id=game_id)


def _manager(tmp_path, delay: float, *, async_agent: bool) -> tuple[PlayManager, _SlowAgent]:
    agent = _SlowAgent(delay)
    manager = PlayManager(
        provider=default_provider,
        history=MatchHistory(tmp_path / "matches"),
        seed=42,
        agent_factory=lambda _key: agent,  # type: ignore[arg-type,return-value]
        agent_async=async_agent,
    )
    return manager, agent


def _wait_for_chat(session, timeout: float = 6.0) -> list[dict]:
    deadline = time.time() + timeout
    while not session.pending_chat and time.time() < deadline:
        time.sleep(0.05)
    return session.snapshot()["chat"]


def _first_cell(session) -> int:
    """玩家可落子的第一个格位（走子 payload 用）。"""
    action = next(iter(session.engine.get_legal_actions(session.state)))
    cell = action.params.get("cell", {})
    index = cell.get("_index") if isinstance(cell, dict) else None
    return int(index) if isinstance(index, int) and index >= 0 else 0


class TestAsyncAgentReplies:
    def test_start_does_not_wait_for_agent(self, tmp_path) -> None:
        manager, _agent = _manager(tmp_path, delay=0.6, async_agent=True)
        started = time.time()
        session = manager.start("moon_chess", "p_black", "easy")
        elapsed = time.time() - started
        assert elapsed < 0.5, f"开局不该等陪伴成文（已等 {elapsed:.2f}s）"
        assert session.snapshot()["chat"] == []  # 开场白还没生成

        messages = _wait_for_chat(session)
        assert messages, "后台线程生成的发言必须能被下一次快照取走"
        assert messages[0]["speaker"]

    def test_move_returns_before_agent_finishes(self, tmp_path) -> None:
        manager, _agent = _manager(tmp_path, delay=0.4, async_agent=True)
        # 9×9 五子棋：一手不会终局（终局播报刻意保持同步，见 PlayManager.move）。
        session = manager.start("stochastic_gomoku", "p_black", "easy")
        _wait_for_chat(session)  # 收走开场白，避免与走子后的消息混淆
        session.drain_chat()

        manager.move(session.game_id, {"cell_index": _first_cell(session)})
        assert not session.over
        # 走子返回时 AI 反应/读人还没生成完 —— 这正是「不被 LLM 拖住」的证据。
        assert session.pending_chat == []
        assert _wait_for_chat(session), "走子后的 AI 反应应随后台线程到达"

    def test_async_is_faster_than_sync_for_slow_agent(self, tmp_path) -> None:
        """同种子同一步：异步不等待陪伴成文，因此明显快于同步路径。"""
        async_manager, _ = _manager(tmp_path / "async", delay=0.4, async_agent=True)
        sync_manager, _ = _manager(tmp_path / "sync", delay=0.4, async_agent=False)

        async_session = async_manager.start("stochastic_gomoku", "p_black", "easy")
        sync_session = sync_manager.start("stochastic_gomoku", "p_black", "easy")
        _wait_for_chat(async_session)
        async_session.drain_chat()

        started = time.time()
        async_manager.move(async_session.game_id, {"cell_index": _first_cell(async_session)})
        async_elapsed = time.time() - started

        started = time.time()
        sync_manager.move(sync_session.game_id, {"cell_index": _first_cell(sync_session)})
        sync_elapsed = time.time() - started

        assert async_elapsed < sync_elapsed - 0.4, (
            f"异步走子应省下陪伴成文的等待（async={async_elapsed:.2f}s sync={sync_elapsed:.2f}s）"
        )

    def test_sync_default_unchanged(self, tmp_path) -> None:
        """默认（同步）路径：调用返回时消息已在队列里（既有契约不变）。"""
        manager, _agent = _manager(tmp_path, delay=0.0, async_agent=False)
        session = manager.start("moon_chess", "p_black", "easy")
        assert session.snapshot()["chat"], "同步模式开局即带开场白"
