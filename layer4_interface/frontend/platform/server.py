"""Platform frontend server — unified hub for games, benchmark and history.

Serves the built React app from ``platform-frontend/dist/`` and exposes
the platform API under ``/api/*``.  Uses only the stdlib, mirroring the
per-game servers under ``layer4_interface/frontend/``.

Run::

    python -m layer4_interface.frontend.platform.server [--host H] [--port P]
"""

from __future__ import annotations

import argparse
import time
import urllib.parse
from dataclasses import asdict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from layer1_translator import translate_rules_json
from layer2_engine.core.llm import LLMClient

from ...agent import PERSONAS, DialogueEngine
from ...difficulty.adaptive import AdaptiveController
from ...online_learning import LearningManager, LearningStore, OnlineModelStore
from ...profile.store import ProfileStore
from ...review import analyze as review_analyze
from ..common.http_utils import BodyTooLargeError, read_json_body, send_error_json, send_json
from .benchmark import SOLVER_OPTIONS, BenchmarkRunner
from .chat import chat_turn
from .conversations import ConversationError, ConversationStore
from .custom_games import CustomGameError, CustomGameRegistry, CustomGameStore
from .game_knowledge import GAME_ALIASES
from .games import GAMES, PlayError
from .history import HistoryError, MatchHistory
from .session import _BUILTIN_FAMILY, PlayManager

ROOT = Path(__file__).resolve().parents[3]
DIST_DIR = ROOT / "platform-frontend" / "dist"
DEFAULT_DATA_DIR = ROOT / "data" / "matches"
PORT = 8770

#: 聊天编排共享的统一 LLM 客户端（懒加载单例；不可用时自动走正则兜底）。
_CHAT_LLM: LLMClient | None = None
#: 未探测到端点时重探测的最小间隔（秒）——避免每个请求都打 /v1/models，
#: 同时让"平台启动后 ollama 才拉起"的场景能自动接上 LLM。
_CHAT_LLM_PROBE_INTERVAL_S = 5.0
_last_chat_probe = 0.0

#: dist 未构建时的浏览器引导页（503）：三步自救，替代裸 JSON 报错。
_DIST_MISSING_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gavis 平台 · 前端未构建</title>
<style>
  body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0;
         max-width: 640px; margin: 12vh auto; padding: 0 24px; line-height: 1.7; }
  h1 { font-size: 22px; } code { background: #1e293b; padding: 2px 8px; border-radius: 6px;
         color: #7dd3fc; font-size: 14px; }
  ol li { margin: 10px 0; } .muted { color: #94a3b8; font-size: 14px; }
</style>
</head>
<body>
<h1>⚡ 前端尚未构建</h1>
<p>平台服务已在运行，但 <code>platform-frontend/dist/</code> 不存在。在项目根目录依次执行：</p>
<ol>
  <li><code>cd platform-frontend</code></li>
  <li><code>npm install</code>（首次需要，需已安装 Node.js）</li>
  <li><code>npm run build</code></li>
</ol>
<p>构建完成后<a href="/">刷新本页</a>即可进入平台。</p>
<p class="muted">提示：API（/api/*）不受影响；开发模式可改用 <code>npm run dev</code>（5173 端口，/api 自动代理到本服务）。</p>
</body>
</html>
"""


def _get_chat_llm() -> LLMClient | None:
    """共享聊天 LLM 客户端；未探测到时周期性重探测（见 _CHAT_LLM_PROBE_INTERVAL_S）。

    Fail-soft 客户端：中途 API 错误由 ``chat_turn`` 落到正则兜底并在
    ``LLMClient`` 内记录 + 告警日志（可观测，不静默）。
    """
    global _CHAT_LLM, _last_chat_probe
    if _CHAT_LLM is None:
        now = time.monotonic()
        if now - _last_chat_probe < _CHAT_LLM_PROBE_INTERVAL_S:
            return None
        _last_chat_probe = now
        if LLMClient.available():
            _CHAT_LLM = LLMClient()
    return _CHAT_LLM


def make_handler(
    manager: PlayManager,
    history: MatchHistory,
    benchmark: BenchmarkRunner,
    dist_dir: Path = DIST_DIR,
    learning: LearningManager | None = None,
    profile_store: ProfileStore | None = None,
    custom: CustomGameRegistry | None = None,
    conversations: ConversationStore | None = None,
) -> type:
    """Build a handler class bound to the given platform services.

    The handler is produced by a factory (rather than holding module-level
    state like the play apps) so tests can mount their own services.
    """

    class PlatformHandler(SimpleHTTPRequestHandler):
        server_version = "GavisPlatform/0.1"

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, directory=str(dist_dir), **kwargs)

        # ── CORS ──────────────────────────────────────────────────
        # 决策更新（审计 3.6 修复，2026-08）：CORS 通配已移除——本服务默认
        # 同源（Python 托管 dist，页面与 /api 同源；开发模式经 Vite
        # proxy 转发也不跨域）。通配 ACAO 曾允许本机浏览器里任意网页跨域
        # 读取对局史/画像并触发写操作。LAN 访问（--host 0.0.0.0）下页面
        # 仍由本服务托管，同源成立。跨域消费方不存在（前端仅两种模式）。
        # 对外网暴露前仍需加鉴权（见 docs/design/security-notes.md）。

        def end_headers(self) -> None:
            super().end_headers()  # 不再发任何 CORS 头：默认同源策略生效

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

        def _send_dist_missing(self) -> None:
            """dist 未构建：给浏览器一页可读的自救指引（而非裸 JSON）。"""
            payload = _DIST_MISSING_HTML.encode("utf-8")
            self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        # ── Routing ───────────────────────────────────────────────

        def do_GET(self) -> None:
            path = urllib.parse.urlsplit(self.path).path
            try:
                if path == "/api/games":
                    self._handle_games()
                elif path == "/api/custom/games":
                    self._handle_custom_games_list()
                elif path == "/api/history":
                    self._handle_history_list()
                elif path.startswith("/api/history/"):
                    self._handle_history_get(path[len("/api/history/") :])
                elif path == "/api/profile":
                    self._handle_profile_get()
                elif path.startswith("/api/review/"):
                    self._handle_review_get(path[len("/api/review/") :])
                elif path == "/api/match/active":
                    self._handle_match_active()
                elif path == "/api/conversations":
                    self._handle_conversations_list()
                elif path.startswith("/api/conversations/"):
                    self._handle_conversation_get(path[len("/api/conversations/") :])
                elif path == "/api/benchmark":
                    self._handle_benchmark_list()
                elif path == "/api/benchmark/status":
                    self._handle_benchmark_status()
                elif path == "/api/learning/status":
                    self._handle_learning_status()
                elif path == "/api/learning/apply":
                    self._handle_learning_apply()
                elif path == "/api/learning/config":
                    self._handle_learning_config()
                elif path.startswith("/api/"):
                    send_error_json(self, HTTPStatus.NOT_FOUND, f"未知接口: {path}")
                else:
                    if not dist_dir.is_dir():
                        self._send_dist_missing()
                        return
                    super().do_GET()
            except (PlayError, HistoryError, CustomGameError, ConversationError) as exc:
                send_error_json(self, HTTPStatus.BAD_REQUEST, str(exc))
            except (KeyError, TypeError, ValueError) as exc:
                send_error_json(self, HTTPStatus.BAD_REQUEST, f"参数错误: {exc}")
            except Exception as exc:  # noqa: BLE001 - last-resort envelope for the client
                self.log_error("internal error: %s", exc)
                send_error_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, "服务器内部错误")
                if not dist_dir.is_dir():
                    self._send_dist_missing()
                    return
                super().do_GET()

        def do_POST(self) -> None:
            path = urllib.parse.urlsplit(self.path).path
            try:
                if path == "/api/custom/games":
                    self._handle_custom_games_create()
                elif path == "/api/match/start":
                    self._handle_match_start()
                elif path == "/api/match/move":
                    self._handle_match_move()
                elif path == "/api/match/state":
                    self._handle_match_state()
                elif path == "/api/benchmark/start":
                    self._handle_benchmark_start()
                elif path == "/api/rules/translate":
                    self._handle_rules_translate()
                elif path == "/api/learning/apply":
                    self._handle_learning_apply()
                elif path == "/api/learning/config":
                    self._handle_learning_config()
                elif path == "/api/match/hint":
                    self._handle_match_hint()
                elif path == "/api/agent/say":
                    self._handle_agent_say()
                elif path == "/api/chat":
                    self._handle_chat()
                elif path == "/api/conversations":
                    self._handle_conversations_create()
                elif path.startswith("/api/conversations/") and path.endswith("/messages"):
                    self._handle_conversation_append(path[len("/api/conversations/") : -len("/messages")])
                elif path.startswith("/api/conversations/"):
                    self._handle_conversation_update(path[len("/api/conversations/") :])
                elif path == "/api/profile":
                    self._handle_profile_save()
                elif path == "/api/profile/clear":
                    self._handle_profile_clear()
                else:
                    send_error_json(self, HTTPStatus.NOT_FOUND, f"未知接口: {path}")
            except BodyTooLargeError as exc:
                send_error_json(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(exc))
            except (PlayError, HistoryError, CustomGameError, ConversationError) as exc:
                send_error_json(self, HTTPStatus.BAD_REQUEST, str(exc))
            except (KeyError, TypeError, ValueError) as exc:
                send_error_json(self, HTTPStatus.BAD_REQUEST, f"参数错误: {exc}")
            except Exception as exc:  # noqa: BLE001 - last-resort envelope for the client
                self.log_error("internal error: %s", exc)
                send_error_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, "服务器内部错误")

        def do_DELETE(self) -> None:
            path = urllib.parse.urlsplit(self.path).path
            try:
                if path.startswith("/api/custom/games/"):
                    self._handle_custom_game_delete(path[len("/api/custom/games/") :])
                elif path.startswith("/api/conversations/"):
                    self._handle_conversation_delete(path[len("/api/conversations/") :])
                else:
                    send_error_json(self, HTTPStatus.NOT_FOUND, f"未知接口: {path}")
            except (PlayError, HistoryError, CustomGameError, ConversationError) as exc:
                send_error_json(self, HTTPStatus.BAD_REQUEST, str(exc))
            except (KeyError, TypeError, ValueError) as exc:
                send_error_json(self, HTTPStatus.BAD_REQUEST, f"参数错误: {exc}")
            except Exception as exc:  # noqa: BLE001 - last-resort envelope for the client
                self.log_error("internal error: %s", exc)
                send_error_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, "服务器内部错误")

        def do_PUT(self) -> None:
            path = urllib.parse.urlsplit(self.path).path
            try:
                if path == "/api/profile":
                    self._handle_profile_save()
                else:
                    send_error_json(self, HTTPStatus.NOT_FOUND, f"未知接口: {path}")
            except BodyTooLargeError as exc:
                send_error_json(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(exc))
            except (PlayError, HistoryError, CustomGameError, ConversationError) as exc:
                send_error_json(self, HTTPStatus.BAD_REQUEST, str(exc))
            except (KeyError, TypeError, ValueError) as exc:
                send_error_json(self, HTTPStatus.BAD_REQUEST, f"参数错误: {exc}")
            except Exception as exc:  # noqa: BLE001 - last-resort envelope for the client
                self.log_error("internal error: %s", exc)
                send_error_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, "服务器内部错误")

        # ── API handlers ──────────────────────────────────────────

        def _handle_games(self) -> None:
            games = []
            for spec in GAMES.values():
                games.append(
                    {
                        "game_id": spec.game_id,
                        "display_name": spec.display_name,
                        "description": spec.description,
                        "kind": spec.kind,
                        "family": _BUILTIN_FAMILY.get(spec.game_id),
                        "custom": False,
                        "board_size": spec.board_size,
                        "seat_options": list(spec.seat_options),
                        "seat_label": spec.seat_label,
                        "player_counts": list(spec.player_counts),
                        "difficulties": list(spec.difficulty_budgets),
                        "solver_options": list(SOLVER_OPTIONS.get(spec.game_id, ())),
                        "aliases": list(GAME_ALIASES.get(spec.game_id, ())),
                    }
                )
            if custom is not None:
                games.extend(custom.list_games())
            send_json(self, HTTPStatus.OK, {"ok": True, "games": games})

        # ── Custom games ─────────────────────────────────────────

        def _handle_custom_games_list(self) -> None:
            """List persisted custom games (``{}`` when the registry is off)."""
            if custom is None:
                send_json(self, HTTPStatus.OK, {"ok": True, "games": []})
                return
            send_json(self, HTTPStatus.OK, {"ok": True, "games": custom.list_games()})

        def _handle_custom_games_create(self) -> None:
            """Create a custom game from a translation (from-scratch / variant)."""
            if custom is None:
                send_error_json(self, HTTPStatus.SERVICE_UNAVAILABLE, "自定义游戏注册表未启用")
                return
            payload = read_json_body(self)
            try:
                entry = custom.create(
                    mode=str(payload.get("mode", "from_scratch")),
                    rule_text=payload.get("rule_text"),
                    base_game_id=payload.get("base_game_id"),
                    change_text=payload.get("change_text"),
                    game_name=payload.get("game_name"),
                    source_lang=str(payload.get("source_lang", "zh")),
                    use_llm=bool(payload.get("use_llm", False)),
                )
            except CustomGameError as exc:
                validation = (
                    {
                        "valid": exc.validation.valid,
                        "errors": list(exc.validation.errors),
                        "warnings": list(exc.validation.warnings),
                    }
                    if exc.validation is not None
                    else None
                )
                send_json(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {
                        "ok": False,
                        "error": str(exc),
                        "validation": validation,
                        "diff_summary": exc.diff_summary,
                    },
                )
                return
            send_json(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "game_id": entry["game_id"],
                    "game": entry,
                    "confidence": entry["confidence"],
                    "family": entry["family"],
                    "diff_summary": entry.get("diff_summary"),
                    "validation": entry["validation"],
                },
            )

        def _handle_custom_game_delete(self, game_id: str) -> None:
            """Delete one custom game; 404 when it does not exist."""
            if custom is None:
                send_error_json(self, HTTPStatus.SERVICE_UNAVAILABLE, "自定义游戏注册表未启用")
                return
            deleted = custom.delete(urllib.parse.unquote(game_id))
            if not deleted:
                send_error_json(self, HTTPStatus.NOT_FOUND, f"自定义游戏不存在: {game_id}")
                return
            send_json(self, HTTPStatus.OK, {"ok": True})

        def _handle_match_start(self) -> None:
            payload = read_json_body(self)
            # 未传 player_count → None，由 session 按注册表默认解析（麻将 4 人）。
            raw_count = payload.get("player_count")
            player_count = int(raw_count) if raw_count is not None else None
            # 教学对局：teaching=true 或 mode="teaching"（教练能看到玩家
            # 自己的牌并推理；见 session.py / agent/coach.py）。
            teaching = bool(payload.get("teaching", False)) or str(payload.get("mode", "")) == "teaching"
            session = manager.start(
                payload["game_id"],
                str(payload.get("player_pid", "random")),
                str(payload.get("difficulty", "normal")),
                player_count,
                persona=str(payload["persona"]) if payload.get("persona") else None,
                hint_level=str(payload.get("hint_level", "off")),
                pacing=str(payload.get("pacing", "standard")),
                adaptive_enabled=bool(payload.get("adaptive", False)),
                teaching=teaching,
            )
            send_json(self, HTTPStatus.OK, {"ok": True, "session": session.snapshot()})

        def _handle_match_move(self) -> None:
            payload = read_json_body(self)
            action = payload.get("action")
            if not isinstance(action, dict):
                raise KeyError("缺少 action")
            snapshot = manager.move(payload["game_id"], action)
            send_json(self, HTTPStatus.OK, {"ok": True, "session": snapshot})

        def _handle_match_state(self) -> None:
            payload = read_json_body(self)
            send_json(self, HTTPStatus.OK, {"ok": True, "session": manager.get(payload["game_id"]).snapshot()})

        def _handle_match_active(self) -> None:
            """List in-flight sessions (support 继续上一局 / resume on refresh)."""
            send_json(self, HTTPStatus.OK, {"ok": True, "sessions": manager.active_sessions()})

        # ── Companion agent / profile / review (D 节接线) ────────

        def _handle_agent_say(self) -> None:
            payload = read_json_body(self)
            message = manager.say(payload["game_id"], str(payload.get("scenario", "idle")))
            send_json(self, HTTPStatus.OK, {"ok": True, "message": message})

        def _handle_chat(self) -> None:
            """Chat-first entry: one sentence → validated intent (LLM + tools, regex fallback).

            ``history`` is optional prior ``user``/``assistant`` turns
            (newest last) that the frontend sends for conversation
            context; ``chat_turn`` sanitizes and bounds it.  The
            platform ``MatchHistory`` (closure ``history``) is passed
            through so the ``get_match_review`` info tool and the
            “最近一局” prompt line have a data source.
            """
            payload = read_json_body(self)
            text = str(payload.get("text", ""))
            game_id = payload.get("game_id")
            if game_id is not None:
                game_id = str(game_id)
            chat_history = payload.get("history")
            result = chat_turn(
                manager,
                text,
                llm=_get_chat_llm(),
                game_id=game_id,
                custom=custom,
                history=chat_history if isinstance(chat_history, list) else None,
                match_history=history,
            )
            send_json(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "intent": result.intent,
                    "text": result.text,
                    "mood": result.mood,
                    "params": result.params,
                },
            )

        def _handle_match_hint(self) -> None:
            payload = read_json_body(self)
            hint = manager.hint(payload["game_id"], str(payload.get("level", "direction")))
            send_json(self, HTTPStatus.OK, {"ok": True, "hint": hint})

        # ── Conversations (对话管理与存档) ────────────────────────
        # 存档随每回合增量 append（前端懒建会话、批式同步）；/api/chat 本身
        # 保持无状态——LLM 上下文仍由前端每轮携带，持久化与编排解耦。

        def _handle_conversations_list(self) -> None:
            """List conversation metadata (archived included, newest first)."""
            if conversations is None:
                send_error_json(self, HTTPStatus.NOT_FOUND, "对话存储未启用")
                return
            send_json(self, HTTPStatus.OK, {"ok": True, "conversations": conversations.list_conversations()})

        def _handle_conversation_get(self, conv_id: str) -> None:
            """Full record (messages included) for one conversation."""
            if conversations is None:
                send_error_json(self, HTTPStatus.NOT_FOUND, "对话存储未启用")
                return
            send_json(
                self,
                HTTPStatus.OK,
                {"ok": True, "conversation": conversations.get(urllib.parse.unquote(conv_id))},
            )

        def _handle_conversations_create(self) -> None:
            """Create a conversation (optionally seeded with messages)."""
            if conversations is None:
                send_error_json(self, HTTPStatus.NOT_FOUND, "对话存储未启用")
                return
            payload = read_json_body(self)
            messages = payload.get("messages")
            record = conversations.create(
                title=str(payload.get("title", "")),
                messages=messages if isinstance(messages, list) else None,
            )
            send_json(self, HTTPStatus.OK, {"ok": True, "conversation": record})

        def _handle_conversation_append(self, conv_id: str) -> None:
            """Append one batch of rendered messages (incremental sync)."""
            if conversations is None:
                send_error_json(self, HTTPStatus.NOT_FOUND, "对话存储未启用")
                return
            payload = read_json_body(self)
            messages = payload.get("messages")
            if not isinstance(messages, list):
                raise KeyError("缺少 messages")
            record = conversations.append_messages(urllib.parse.unquote(conv_id), messages)
            meta = {k: v for k, v in record.items() if k != "messages"}
            meta["message_count"] = len(record.get("messages", []))
            send_json(self, HTTPStatus.OK, {"ok": True, "conversation": meta})

        def _handle_conversation_update(self, conv_id: str) -> None:
            """Rename and/or (un)archive a conversation."""
            if conversations is None:
                send_error_json(self, HTTPStatus.NOT_FOUND, "对话存储未启用")
                return
            payload = read_json_body(self)
            title = payload.get("title")
            archived = payload.get("archived")
            record = conversations.update(
                urllib.parse.unquote(conv_id),
                title=str(title) if title is not None else None,
                archived=bool(archived) if archived is not None else None,
            )
            meta = {k: v for k, v in record.items() if k != "messages"}
            meta["message_count"] = len(record.get("messages", []))
            send_json(self, HTTPStatus.OK, {"ok": True, "conversation": meta})

        def _handle_conversation_delete(self, conv_id: str) -> None:
            """Delete one conversation record permanently."""
            if conversations is None:
                send_error_json(self, HTTPStatus.NOT_FOUND, "对话存储未启用")
                return
            conversations.delete(urllib.parse.unquote(conv_id))
            send_json(self, HTTPStatus.OK, {"ok": True})

        def _handle_profile_get(self) -> None:
            if profile_store is None:
                send_error_json(self, HTTPStatus.NOT_FOUND, "档案存储未启用")
                return
            send_json(self, HTTPStatus.OK, {"ok": True, "profile": profile_store.load()})

        def _handle_profile_save(self) -> None:
            if profile_store is None:
                send_error_json(self, HTTPStatus.NOT_FOUND, "档案存储未启用")
                return
            payload = read_json_body(self)
            patch = payload.get("profile")
            if not isinstance(patch, dict):
                raise KeyError("缺少 profile 字段")
            merged = {**profile_store.load(), **patch}
            profile_store.save(merged)
            send_json(self, HTTPStatus.OK, {"ok": True, "profile": merged})

        def _handle_profile_clear(self) -> None:
            if profile_store is None:
                send_error_json(self, HTTPStatus.NOT_FOUND, "档案存储未启用")
                return
            profile_store.clear()
            send_json(self, HTTPStatus.OK, {"ok": True, "profile": profile_store.load()})

        def _handle_review_get(self, match_id: str) -> None:
            report = review_analyze(history.get(urllib.parse.unquote(match_id)))
            send_json(self, HTTPStatus.OK, {"ok": True, "report": asdict(report)})

        def _handle_history_list(self) -> None:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            limit = int(query.get("limit", ["100"])[0])
            game_id = query.get("game_id", [None])[0]
            send_json(self, HTTPStatus.OK, {"ok": True, "matches": history.list_matches(limit=limit, game_id=game_id)})

        def _handle_history_get(self, match_id: str) -> None:
            send_json(self, HTTPStatus.OK, {"ok": True, "match": history.get(urllib.parse.unquote(match_id))})

        def _handle_benchmark_list(self) -> None:
            send_json(self, HTTPStatus.OK, {"ok": True, "jobs": [asdict(j) for j in benchmark.list_jobs()]})

        def _handle_benchmark_status(self) -> None:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            job_id = query.get("job_id", [""])[0]
            job = benchmark.status(job_id)
            if job is None:
                send_error_json(self, HTTPStatus.NOT_FOUND, f"未知评测任务: {job_id}")
                return
            send_json(self, HTTPStatus.OK, {"ok": True, "job": asdict(job)})

        def _handle_benchmark_start(self) -> None:
            payload = read_json_body(self)
            budget = payload.get("budget")
            job = benchmark.start(
                payload["game_id"],
                str(payload["solver_a"]),
                str(payload["solver_b"]),
                int(payload.get("iterations", 10)),
                budget=int(budget) if budget is not None else None,
            )
            send_json(self, HTTPStatus.OK, {"ok": True, "job_id": job.job_id, "job": asdict(job)})

        def _handle_rules_translate(self) -> None:
            payload = read_json_body(self)
            # use_llm=True → strict_llm=True：显式要求 LLM 翻译时，API 错误/
            # 传输失败如实上报（validation.errors 带真实原因），不静默模板兜底。
            use_llm = bool(payload.get("use_llm", False))
            response = translate_rules_json(
                str(payload.get("rule_text", "")),
                source_lang=str(payload.get("source_lang", "zh")),
                game_name=payload.get("game_name"),
                external_frontend=payload.get("external_frontend"),
                run_engine_validation=bool(payload.get("run_engine_validation", True)),
                use_llm=use_llm,
                strict_llm=use_llm,
                llm_model=payload.get("llm_model"),
                llm_model_path=payload.get("llm_model_path"),
            )
            validation = response.validation
            send_json(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "rules_json": response.rules_json,
                    "confidence": response.confidence,
                    "validation": asdict(validation) if validation is not None else None,
                },
            )

        # ── Online learning ────────────────────────────────────
        # 在线学习：捕获在 PlayManager 内进行（learning 钩子包裹求解器
        # 句柄并记录每步决策）；本组接口只读状态与触发 apply（建表 +
        # 门禁评估 + 发布），apply 在 HTTP 请求线程内同步执行（默认
        # 局数/预算较小），失败不破坏既有模型（保留上一版本）。

        def _handle_learning_status(self) -> None:
            send_json(self, HTTPStatus.OK, {"ok": True, "learning": learning.status_all()})

        def _handle_learning_apply(self) -> None:
            payload = read_json_body(self)
            game_id = payload.get("game_id")
            if game_id:
                send_json(self, HTTPStatus.OK, {"ok": True, "result": asdict(learning.apply(str(game_id)))})
                return
            results = [asdict(learning.apply(item["game_id"])) for item in learning.status_all()]
            send_json(self, HTTPStatus.OK, {"ok": True, "results": results})

        def _handle_learning_config(self) -> None:
            payload = read_json_body(self)
            game_id = str(payload.get("game_id", ""))
            if game_id not in GAMES:
                raise PlayError(f"未知游戏: {game_id}")
            learning.set_enabled(game_id, bool(payload.get("enabled", True)))
            send_json(self, HTTPStatus.OK, {"ok": True, "learning": learning.status(game_id)})

    return PlatformHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Gavis 平台前端服务 (游戏大厅 / 对战 / 评测 / 历史 / 在线学习)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--learning-min-samples", type=int, default=30, help="apply 的最低人类决策样本数")
    parser.add_argument("--learning-gate-episodes", type=int, default=20, help="门禁短赛局数")
    parser.add_argument("--learning-gate-budget", type=int, default=300, help="门禁搜索预算")
    parser.add_argument(
        "--learning-interval", type=float, default=0.0, help="后台 auto-apply 间隔秒（0 = 仅手动 apply）"
    )
    args = parser.parse_args()

    # 求解器由注册表装配（SolverProvider 注入，L4 不 import L3）
    from train_cli import default_provider

    history = MatchHistory(args.data_dir)
    # 在线学习：捕获数据在 data/online_learning/（已 gitignore），
    # 与用户可见的 data/matches/ 严格分离。
    learning_dir = args.data_dir.parent / "online_learning"
    learning_store = LearningStore(learning_dir)
    model_store = OnlineModelStore(learning_dir / "models")
    default_provider.online_models = model_store  # 新对局读取已发布模型
    learning = LearningManager(
        store=learning_store,
        model_store=model_store,
        provider=default_provider,
        seed=42,
        min_samples=args.learning_min_samples,
        gate_episodes=args.learning_gate_episodes,
        gate_budget=args.learning_gate_budget,
    )
    profiles = ProfileStore(args.data_dir.parent)
    adaptive = AdaptiveController()
    # 自定义游戏注册表（data/custom_games/，与 matches/online_learning 并列）。
    custom_registry = CustomGameRegistry(CustomGameStore(args.data_dir.parent / "custom_games"))
    # 对话存档（data/conversations/，与 matches/online_learning/custom_games 并列）。
    conversation_store = ConversationStore(args.data_dir.parent / "conversations")

    def _make_agent(persona_key: str) -> DialogueEngine | None:
        persona = PERSONAS.get(persona_key)
        if persona is None:
            return None
        llm = LLMClient() if LLMClient.available() else None
        return DialogueEngine(persona, llm=llm)

    manager = PlayManager(
        provider=default_provider,
        history=history,
        seed=42,
        learning=learning,
        profiles=profiles,
        adaptive=adaptive,
        agent_factory=_make_agent,
        custom=custom_registry,
    )
    benchmark = BenchmarkRunner(provider=default_provider, seed=42)
    if args.learning_interval > 0:
        learning.start_auto(interval_seconds=args.learning_interval)
        print(f"在线学习 auto-apply 已启动: 每 {args.learning_interval}s 检查一次")
    handler = make_handler(
        manager,
        history,
        benchmark,
        learning=learning,
        profile_store=profiles,
        custom=custom_registry,
        conversations=conversation_store,
    )
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Gavis 平台服务: http://{args.host}:{args.port}  (API 前缀 /api, 静态目录 {DIST_DIR})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if args.learning_interval > 0:
            learning.stop_auto()
        httpd.server_close()


if __name__ == "__main__":
    main()
