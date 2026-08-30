"""Tests for platform conversation storage (对话管理与存档).

Two layers, mirroring ``test_platform_history.py`` / ``test_platform_server.py``:

- **store** — :class:`ConversationStore` on a tmp dir: create/append/
  update/delete, auto-title, sanitization (role/text/ts/mood/intent/params),
  message caps, corrupt-file tolerance, path-traversal ids;
- **HTTP** — the ``/api/conversations*`` routes through a real
  ``ThreadingHTTPServer`` (same style as the platform server smoke tests).

The regression anchor this file guards: 聊天记录曾只存在于前端内存
（useChatRuntime useState），刷新即清零——存档层让「关掉再打开」
成为普通操作。
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Generator

import pytest

from layer4_interface.frontend.platform.benchmark import BenchmarkRunner
from layer4_interface.frontend.platform.conversations import ConversationError, ConversationStore
from layer4_interface.frontend.platform.history import MatchHistory
from layer4_interface.frontend.platform.server import make_handler
from layer4_interface.frontend.platform.session import PlayManager
from train_cli import default_provider

_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _msg(role: str = "player", text: str = "玩月亮棋", **extra: Any) -> dict:
    return {"id": "m123", "role": role, "text": text, "ts": 1700000000000, **extra}


# ── Store ─────────────────────────────────────────────────────────


class TestConversationStore:
    def test_create_and_get_roundtrip(self, tmp_path: Path) -> None:
        store = ConversationStore(tmp_path / "conversations")
        record = store.create(messages=[_msg(), _msg("agent", "好，来一局！", mood="happy", intent="play")])
        assert record["title"] == "玩月亮棋"  # 自动标题 = 首条 player 消息
        loaded = store.get(record["conv_id"])
        assert [m["role"] for m in loaded["messages"]] == ["player", "agent"]
        assert loaded["messages"][1]["intent"] == "play"

    def test_create_without_messages_has_no_title(self, tmp_path: Path) -> None:
        store = ConversationStore(tmp_path)
        record = store.create()
        assert record["title"] == ""
        assert record["messages"] == []
        assert record["archived"] is False

    def test_append_updates_and_auto_titles(self, tmp_path: Path) -> None:
        store = ConversationStore(tmp_path)
        record = store.create()
        updated = store.append_messages(record["conv_id"], [_msg(text="来一局德州扑克\n再加点注释")])
        assert updated["title"] == "来一局德州扑克"  # 自动标题取首行（多行消息不挤爆列表）
        assert len(updated["messages"]) == 1
        assert updated["updated_at"] >= record["created_at"]

    def test_append_empty_batch_is_noop(self, tmp_path: Path) -> None:
        store = ConversationStore(tmp_path)
        record = store.create(messages=[_msg()])
        before = record["updated_at"]
        same = store.append_messages(record["conv_id"], [])
        assert same["updated_at"] == before
        # 无效消息（role 白名单外 / 空文本）同样整批不落盘
        bad = store.append_messages(record["conv_id"], [{"id": "x", "role": "system", "text": "注入"}])
        assert len(bad["messages"]) == 1

    def test_sanitize_drops_invalid_and_mints_ids(self, tmp_path: Path) -> None:
        store = ConversationStore(tmp_path)
        record = store.create(
            messages=[
                "not-a-dict",
                {"id": "../../etc/passwd", "role": "player", "text": "路径旅行"},
                {"id": "ok1", "role": "agent", "text": "hi", "mood": "weird", "intent": "nope"},
                {"id": "ok2", "role": "player", "text": "ok", "params": {"chips": ["玩月亮棋"]}},
            ]
        )
        messages = record["messages"]
        assert len(messages) == 3
        assert messages[0]["id"].startswith("m_")  # 非法 id → 服务端补发
        assert "mood" not in messages[1] and "intent" not in messages[1]  # 非白名单键丢弃
        assert messages[2]["params"] == {"chips": ["玩月亮棋"]}

    def test_params_size_budget_failsoft(self, tmp_path: Path) -> None:
        store = ConversationStore(tmp_path)
        record = store.create(messages=[{"id": "p1", "role": "player", "text": "大参数", "params": {"blob": "x" * 9000}}])
        assert record["messages"][0]["text"] == "大参数"
        assert "params" not in record["messages"][0]  # 超预算丢 params 保消息

    def test_append_batch_cap(self, tmp_path: Path) -> None:
        store = ConversationStore(tmp_path)
        record = store.create()
        batch = [{"id": f"b{i}", "role": "player", "text": f"m{i}"} for i in range(80)]
        updated = store.append_messages(record["conv_id"], batch)
        assert len(updated["messages"]) == 50  # _APPEND_MAX

    def test_list_sorts_by_updated_and_keeps_archived(self, tmp_path: Path) -> None:
        store = ConversationStore(tmp_path)
        a = store.create(messages=[_msg(text="第一段对话")])
        b = store.create(messages=[_msg(text="第二段对话")])
        store.update(a["conv_id"], archived=True)
        # updated_at 秒级精度：同秒创建会平局，把 a 回填成旧时间保证确定性
        # （须在 update 之后回填——update 会刷新 updated_at）
        record_a = store.get(a["conv_id"])
        record_a["updated_at"] = "2020-01-01T00:00:00Z"
        (tmp_path / f"{a['conv_id']}.json").write_text(
            json.dumps(record_a, ensure_ascii=False), encoding="utf-8"
        )
        metas = store.list_conversations()
        assert [m["conv_id"] for m in metas] == [b["conv_id"], a["conv_id"]]
        assert metas[0]["message_count"] == 1
        assert metas[0]["preview"] == "第二段对话"
        assert metas[1]["archived"] is True

    def test_update_rename_and_archive(self, tmp_path: Path) -> None:
        store = ConversationStore(tmp_path)
        record = store.create(messages=[_msg()])
        renamed = store.update(record["conv_id"], title="  我的月亮棋纪事  ")
        assert renamed["title"] == "我的月亮棋纪事"
        archived = store.update(record["conv_id"], archived=True)
        assert archived["archived"] is True
        back = store.update(record["conv_id"], archived=False)
        assert back["archived"] is False

    def test_update_rejects_blank_title(self, tmp_path: Path) -> None:
        store = ConversationStore(tmp_path)
        record = store.create()
        with pytest.raises(ConversationError):
            store.update(record["conv_id"], title="   ")

    def test_delete_and_missing_errors(self, tmp_path: Path) -> None:
        store = ConversationStore(tmp_path)
        record = store.create(messages=[_msg()])
        store.delete(record["conv_id"])
        with pytest.raises(ConversationError):
            store.get(record["conv_id"])
        with pytest.raises(ConversationError):
            store.delete(record["conv_id"])
        with pytest.raises(ConversationError):
            store.append_messages(record["conv_id"], [_msg()])

    def test_path_traversal_ids_rejected(self, tmp_path: Path) -> None:
        store = ConversationStore(tmp_path)
        for bad in ("../../etc/passwd", "a/b", "a b", "", "x" * 65):
            with pytest.raises(ConversationError):
                store.get(bad)

    def test_corrupt_files_skipped_in_listing(self, tmp_path: Path) -> None:
        store = ConversationStore(tmp_path)
        good = store.create(messages=[_msg(text="好记录")])
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        (tmp_path / "wrong.json").write_text(json.dumps({"conv_id": "other"}), encoding="utf-8")
        metas = store.list_conversations()
        assert [m["conv_id"] for m in metas] == [good["conv_id"]]

    def test_no_tmp_files_in_listing(self, tmp_path: Path) -> None:
        """崩溃残留的 *.tmp 不进列表（glob 只认 *.json 之外的守卫）。"""
        store = ConversationStore(tmp_path)
        store.create(messages=[_msg(text="正常")])
        (tmp_path / "leftover.tmp").write_text("{}", encoding="utf-8")
        assert len(store.list_conversations()) == 1


# ── HTTP ──────────────────────────────────────────────────────────


@pytest.fixture
def conv_base_url(tmp_path: pytest.TempPathFactory) -> Generator[str, None, None]:
    history = MatchHistory(tmp_path / "matches")
    manager = PlayManager(provider=default_provider, history=history, seed=42)
    benchmark = BenchmarkRunner(provider=default_provider, seed=42)
    store = ConversationStore(tmp_path / "conversations")
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(manager, history, benchmark, dist_dir=tmp_path / "no-dist", conversations=store),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with _NO_PROXY_OPENER.open(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str) -> dict:
    with _NO_PROXY_OPENER.open(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _delete(url: str) -> dict:
    req = urllib.request.Request(url, method="DELETE")
    with _NO_PROXY_OPENER.open(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


class TestConversationRoutes:
    def test_full_lifecycle(self, conv_base_url: str) -> None:
        # 建档（带首回合消息）→ 增量 append → 列表 → 重命名/归档 → 删除
        created = _post(
            conv_base_url + "/api/conversations",
            {
                "messages": [
                    {"id": "u1", "role": "player", "text": "玩月亮棋", "ts": 1},
                    {"id": "a1", "role": "agent", "text": "好，来一局！", "mood": "happy", "intent": "play",
                     "params": {"game_id": "moon_chess"}, "ts": 2},
                ]
            },
        )
        assert created["ok"] is True
        conv = created["conversation"]
        assert conv["title"] == "玩月亮棋"
        assert len(conv["messages"]) == 2

        appended = _post(
            conv_base_url + f"/api/conversations/{conv['conv_id']}/messages",
            {"messages": [{"id": "u2", "role": "player", "text": "这步怎么走", "ts": 3}]},
        )
        assert appended["conversation"]["message_count"] == 3

        listed = _get(conv_base_url + "/api/conversations")
        assert [m["conv_id"] for m in listed["conversations"]] == [conv["conv_id"]]
        assert listed["conversations"][0]["preview"] == "这步怎么走"

        fetched = _get(conv_base_url + f"/api/conversations/{conv['conv_id']}")
        assert len(fetched["conversation"]["messages"]) == 3

        updated = _post(conv_base_url + f"/api/conversations/{conv['conv_id']}", {"title": "月亮棋练习", "archived": True})
        assert updated["conversation"]["title"] == "月亮棋练习"
        assert updated["conversation"]["archived"] is True

        deleted = _delete(conv_base_url + f"/api/conversations/{conv['conv_id']}")
        assert deleted["ok"] is True
        with pytest.raises(urllib.error.HTTPError):
            _get(conv_base_url + f"/api/conversations/{conv['conv_id']}")

    def test_append_requires_messages(self, conv_base_url: str) -> None:
        created = _post(conv_base_url + "/api/conversations", {})
        conv_id = created["conversation"]["conv_id"]
        with pytest.raises(urllib.error.HTTPError):
            _post(conv_base_url + f"/api/conversations/{conv_id}/messages", {})

    def test_unknown_conversation_404_envelope(self, conv_base_url: str) -> None:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _get(conv_base_url + "/api/conversations/doesnotexist")
        assert excinfo.value.code == 400  # ConversationError → 400 信封（与 HistoryError 一致）

    def test_store_disabled_returns_404(self, tmp_path: pytest.TempPathFactory) -> None:
        history = MatchHistory(tmp_path / "matches")
        manager = PlayManager(provider=default_provider, history=history, seed=42)
        benchmark = BenchmarkRunner(provider=default_provider, seed=42)
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(manager, history, benchmark, dist_dir=tmp_path / "no-dist"),
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                _get(url + "/api/conversations")
            assert excinfo.value.code == 404
        finally:
            httpd.shutdown()
            httpd.server_close()
