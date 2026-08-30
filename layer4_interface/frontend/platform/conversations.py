"""Conversation storage for the platform chat (对话管理与存档).

One conversation = one JSON file under ``data/conversations/<conv_id>.json``.
Writes are atomic (temp file + ``os.replace``) — the same discipline as
``history.MatchHistory`` so a crash never leaves a partial record on disk.

The store persists the *rendered* chat messages (role / text / mood /
intent / params) rather than raw LLM turns, so the frontend can restore
the full conversation after a refresh or a reopen — inline cards (开局卡 /
战绩卡 / 复盘卡 / chips) included.  Sanitization is fail-soft per message:
bad entries are dropped, never rejected wholesale (one malformed message
must not cost the user their whole archive).

Conversation lifecycle:

- ``create`` — new record (optionally seeded with the first messages);
- ``append_messages`` — incremental sync from the frontend (one batch per
  chat turn / drained coach messages);
- ``update`` — rename / archive toggle (``archived`` conversations stay
  on disk but leave the active list);
- ``delete`` — permanent removal.

Auto-title: a conversation created without an explicit title takes its
first player message (truncated) as the title — the common「新对话」flow
needs no extra round-trip.

Message contract (mirrors ``platform-frontend/src/types.ts`` ``ChatMessage``):

========  ==========================================================
key       rule
========  ==========================================================
id        str, ``[A-Za-z0-9_-]{1,64}``（缺失时服务端补发）
role      ``agent`` | ``player``
text      non-empty str（strip 后）
ts        int（毫秒 epoch；缺失时补当前时间）
mood      happy | thinking | sorry | neutral（可选）
intent    15 个平台意图白名单（可选）
params    dict，序列化 ≤ 8KB（超限整体丢弃，fail-soft）
reasoning 思维链文本（可选；剔控制字符 + ≤ 4000 字符，非字符串丢弃）
========  ==========================================================
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from layer2_engine.core.llm import sanitize_text

#: conv_id 白名单：仅字母数字、下划线、连字符（与 history.match_id 同款，
#: 杜绝路径分隔符/`..` 逃逸出 data 目录）。
_CONV_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: 消息 id 白名单（前端 uid() 产物天然满足）。
_MSG_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_MSG_ROLES = frozenset({"agent", "player"})
_MOODS = frozenset({"happy", "thinking", "sorry", "neutral"})

#: intent 白名单 — 与 chat.py 的意图契约、前端 ChatIntent 一致。
_INTENTS = frozenset(
    {
        "play",
        "resume",
        "move",
        "hint",
        "restart",
        "history",
        "review",
        "create",
        "settings",
        "platform",
        "benchmark",
        "learning",
        "help",
        "chat",
        "clarify",
    }
)

#: 单条消息 params 的序列化预算（字符）。超限丢 params 保消息。
_PARAMS_MAX_CHARS = 8192
#: 单条消息 reasoning（思维链）的存档上限（字符）。超限截断、非字符串丢弃（fail-soft）。
_REASONING_MAX = 4000
#: 单次 append 的消息条数上限（防一次请求撑爆文件）。
_APPEND_MAX = 50
#: 单个对话的消息条数上限（超限从头裁剪——对话档是可再生的展示记录）。
_CONV_MAX_MESSAGES = 2000
#: 自动标题 / 手动重命名的长度上限（字符）。
_TITLE_MAX = 40
#: 列表 preview 截断长度（字符）。
_PREVIEW_MAX = 60


class ConversationError(Exception):
    """Missing, unreadable, or invalid conversation record."""


def _now_iso() -> str:
    """UTC ISO-8601 timestamp (second precision, ``Z`` suffix)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_conv_id(conv_id: str) -> None:
    """Validate conv_id against the path-traversal-safe whitelist."""
    if not isinstance(conv_id, str) or not _CONV_ID_RE.fullmatch(conv_id):
        raise ConversationError(f"invalid conv_id: {conv_id!r}")


def _sanitize_message(item: Any) -> dict[str, Any] | None:
    """Validate one client-supplied message (fail-soft → ``None``)."""
    if not isinstance(item, dict):
        return None
    role = item.get("role")
    text = item.get("text")
    if role not in _MSG_ROLES or not isinstance(text, str) or not text.strip():
        return None
    msg: dict[str, Any] = {
        "id": _sanitize_msg_id(item.get("id")),
        "role": str(role),
        "text": text.strip(),
    }
    ts = item.get("ts")
    msg["ts"] = int(ts) if isinstance(ts, (int, float)) and int(ts) > 0 else int(time.time() * 1000)
    mood = item.get("mood")
    if mood in _MOODS:
        msg["mood"] = str(mood)
    intent = item.get("intent")
    if intent in _INTENTS:
        msg["intent"] = str(intent)
    reasoning = item.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        msg["reasoning"] = sanitize_text(reasoning, _REASONING_MAX).strip()
    params = item.get("params")
    if isinstance(params, dict) and params:
        try:
            if len(json.dumps(params, ensure_ascii=False)) <= _PARAMS_MAX_CHARS:
                msg["params"] = params
        except (TypeError, ValueError):
            pass  # 不可序列化的 params 整体丢弃，消息本体保留
    return msg


def _sanitize_msg_id(raw: Any) -> str:
    """Keep client message ids that fit the whitelist; else mint one."""
    if isinstance(raw, str) and _MSG_ID_RE.fullmatch(raw):
        return raw
    return "m_" + uuid.uuid4().hex[:12]


def _title_from(text: str) -> str:
    """Auto-title: first line, whitespace collapsed, truncated."""
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    first = re.sub(r"\s+", " ", first)
    return first[:_TITLE_MAX]


class ConversationStore:
    """Filesystem-backed store of chat conversations (对话存档)."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Reading ──────────────────────────────────────────────────

    def list_conversations(self, limit: int = 100) -> list[dict]:
        """Conversation metadata, newest (``updated_at``) first.

        Archived conversations are included with ``archived: True`` —
        the frontend splits the active/archived sections client-side.
        Corrupt or unreadable files are skipped silently (fail-soft).
        """
        metas: list[dict] = []
        for path in self.data_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    record = json.load(f)
            except (OSError, ValueError):
                continue
            meta = self._meta(record) if isinstance(record, dict) else None
            if meta is not None:
                metas.append(meta)
        metas.sort(key=lambda m: (m.get("updated_at") or "", m.get("conv_id") or ""), reverse=True)
        return metas[:limit]

    def get(self, conv_id: str) -> dict:
        """Full conversation record including the message log."""
        record = self._load(conv_id)
        return record

    # ── Writing ──────────────────────────────────────────────────

    def create(self, title: str = "", messages: list[Any] | None = None) -> dict:
        """Create a new conversation; returns the stored record.

        ``messages`` may seed the record (the frontend creates a
        conversation lazily with the first turn's messages).  An empty
        ``title`` is auto-filled from the first player message.
        """
        now = _now_iso()
        clean = [m for m in (_sanitize_message(i) for i in (messages or [])) if m is not None][:_APPEND_MAX]
        record = {
            "conv_id": uuid.uuid4().hex[:12],
            "title": (title or "").strip()[:_TITLE_MAX],
            "archived": False,
            "created_at": now,
            "updated_at": now,
            "messages": clean,
        }
        if not record["title"]:
            record["title"] = self._auto_title(clean)
        self._atomic_write(record)
        return record

    def append_messages(self, conv_id: str, messages: list[Any]) -> dict:
        """Append sanitized messages; returns the updated record.

        Batch is capped at ``_APPEND_MAX``; the conversation keeps at
        most ``_CONV_MAX_MESSAGES`` messages (oldest dropped first).
        Auto-title kicks in only when the record has no title yet.
        """
        record = self._load(conv_id)
        batch = [m for m in (_sanitize_message(i) for i in (messages or [])) if m is not None][:_APPEND_MAX]
        if not batch:
            return record  # 无有效消息 → 不动文件（也不刷新 updated_at）
        record["messages"] = (record.get("messages") or []) + batch
        if len(record["messages"]) > _CONV_MAX_MESSAGES:
            record["messages"] = record["messages"][-_CONV_MAX_MESSAGES:]
        if not str(record.get("title") or "").strip():
            record["title"] = self._auto_title(batch)
        record["updated_at"] = _now_iso()
        self._atomic_write(record)
        return record

    def update(self, conv_id: str, *, title: str | None = None, archived: bool | None = None) -> dict:
        """Rename and/or (un)archive a conversation; returns the record."""
        record = self._load(conv_id)
        if title is not None:
            title = str(title).strip()
            if not title:
                raise ConversationError("标题不能为空")
            record["title"] = title[:_TITLE_MAX]
        if archived is not None:
            record["archived"] = bool(archived)
        record["updated_at"] = _now_iso()
        self._atomic_write(record)
        return record

    def delete(self, conv_id: str) -> None:
        """Remove a stored conversation record."""
        _check_conv_id(conv_id)
        path = self.data_dir / f"{conv_id}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            raise ConversationError(f"conversation not found: {conv_id}") from None

    # ── Internals ────────────────────────────────────────────────

    def _load(self, conv_id: str) -> dict:
        """Read and structurally validate one record."""
        _check_conv_id(conv_id)
        path = self.data_dir / f"{conv_id}.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except OSError:
            raise ConversationError(f"conversation not found: {conv_id}") from None
        except ValueError:
            raise ConversationError(f"conversation record is corrupt: {conv_id}") from None
        if not isinstance(record, dict) or record.get("conv_id") != conv_id:
            raise ConversationError(f"conversation record is corrupt: {conv_id}")
        if not isinstance(record.get("messages"), list):
            record["messages"] = []
        return record

    def _atomic_write(self, record: dict) -> None:
        """Write via a temp file in the same directory, then rename."""
        path = self.data_dir / f"{record['conv_id']}.json"
        fd, tmp_name = tempfile.mkstemp(dir=self.data_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _auto_title(messages: list[dict]) -> str:
        """First player message (fallback: any message) → title text."""
        for msg in messages:
            if msg.get("role") == "player":
                return _title_from(str(msg.get("text") or ""))
        for msg in messages:
            return _title_from(str(msg.get("text") or ""))
        return ""

    @staticmethod
    def _meta(record: dict) -> dict | None:
        """List metadata for a loaded record (``None`` when unusable)."""
        conv_id = record.get("conv_id")
        if not isinstance(conv_id, str) or not conv_id:
            return None
        messages = record.get("messages")
        if not isinstance(messages, list):
            return None  # 非本存储产物的杂散 JSON 不进列表
        preview = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and str(msg.get("text") or "").strip():
                preview = re.sub(r"\s+", " ", str(msg["text"]).strip())[:_PREVIEW_MAX]
                break
        return {
            "conv_id": conv_id,
            "title": str(record.get("title") or ""),
            "archived": bool(record.get("archived")),
            "created_at": str(record.get("created_at") or ""),
            "updated_at": str(record.get("updated_at") or ""),
            "message_count": len(messages),
            "preview": preview,
        }
