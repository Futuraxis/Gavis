"""HTTP smoke tests for the platform server endpoints.

A real ``ThreadingHTTPServer`` on an ephemeral port is exercised with
``urllib`` — the same style the play apps' tests use.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Generator

import pytest

from layer4_interface.frontend.platform.benchmark import BenchmarkRunner
from layer4_interface.frontend.platform.history import MatchHistory
from layer4_interface.frontend.platform.llm_settings import LLMSettingsStore
from layer4_interface.frontend.platform.server import make_handler
from layer4_interface.frontend.platform.session import PlayManager
from train_cli import default_provider

_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@pytest.fixture
def base_url(tmp_path: pytest.TempPathFactory) -> Generator[str, None, None]:
    history = MatchHistory(tmp_path / "matches")
    manager = PlayManager(provider=default_provider, history=history, seed=42)
    benchmark = BenchmarkRunner(provider=default_provider, seed=42)
    # dist_dir points at a missing directory → the 503 path is testable
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(manager, history, benchmark, dist_dir=tmp_path / "no-dist"),
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


def _put(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with _NO_PROXY_OPENER.open(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str) -> dict:
    with _NO_PROXY_OPENER.open(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_stream(url: str, payload: dict, *, accept: bool = False) -> tuple[str, bytes]:
    """POST and return ``(content-type, raw body)`` (SSE mode is opt-in)."""
    headers = {"Content-Type": "application/json"}
    if accept:
        headers["Accept"] = "text/event-stream"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with _NO_PROXY_OPENER.open(req) as resp:
        return (resp.headers.get("Content-Type") or ""), resp.read()


def _parse_sse(body: bytes) -> list[tuple[str, str]]:
    """Split an SSE body into ``(event, data-json)`` frames (no runtime deps)."""
    frames: list[tuple[str, str]] = []
    for block in body.decode("utf-8").split("\n\n"):
        if not block.strip():
            continue
        event = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: ") :])
        if event or data_lines:
            frames.append((event, "\n".join(data_lines)))
    return frames


class TestGames:
    def test_list_games(self, base_url: str):
        data = _get(base_url + "/api/games")
        assert data["ok"] is True
        by_id = {g["game_id"]: g for g in data["games"]}
        # 17 款 = 月亮棋/随机五子棋/德州 + 麻将六变种（v5.2 variants）+ UNO
        # 六变体 + 谁是卧底（undercover, social 族）+ 狼人杀（werewolf,
        # social 族）—— 与 test_platform_session.py::TestGameSpecRegistry 的
        # 17 游戏契约一致；新增/移除必须同步两处断言与用户文档。
        assert set(by_id) == {
            "moon_chess",
            "stochastic_gomoku",
            "texas_holdem",
            "mahjong_guangdong",
            "mahjong_hongzhong",
            "mahjong_blood",
            "mahjong_sichuan",
            "mahjong_changsha",
            "mahjong_taiwan",
            "mahjong_international",
            "uno",
            "uno_seven_zero",
            "uno_jump_in",
            "uno_stacking",
            "uno_draw_until",
            "uno_strict_wild4",
            "undercover",
            "werewolf",
        }
        assert by_id["moon_chess"]["board_size"] == 3
        assert by_id["stochastic_gomoku"]["board_size"] == 9
        assert by_id["texas_holdem"]["kind"] == "poker"
        assert by_id["uno"]["kind"] == "uno"
        assert by_id["undercover"]["family"] == "social"
        assert by_id["undercover"]["player_counts"] == [8, 4, 5, 6, 7, 9, 10, 11, 12]
        assert by_id["werewolf"]["family"] == "social"
        assert by_id["werewolf"]["player_counts"] == [9]
        assert "cfr" not in by_id["texas_holdem"]["solver_options"]
        assert by_id["moon_chess"]["solver_options"] == ["mcts", "cfr", "hybrid", "random"]

    def test_every_game_info_carries_family(self, base_url: str):
        """/api/games 每个条目必须携带非空 family —— 前端按 family 分发棋盘组件。

        回归锚：mahjong_sichuan/changsha/taiwan 曾因 `_BUILTIN_FAMILY` 缺项而
        family 为 null，前端 InlineBoard 把麻将快照误路由到 grid 棋盘（读不到
        board）崩掉整个对话页。此断言让「注册表游戏 ⇒ 非空 family」成为线上
        契约；与 test_platform_session.py::TestGameSpecRegistry::
        test_builtin_family_covers_every_registry_game 同步维护。
        """
        data = _get(base_url + "/api/games")
        assert data["ok"] is True
        for g in data["games"]:
            assert isinstance(g.get("family"), str) and g["family"], (
                f"{g['game_id']} 的 family 缺失/为空，前端会把非 grid 快照误路由到 grid 棋盘"
            )

    def test_no_cors_wildcard_headers(self, base_url: str):
        """审计 B6：同源服务不得发 CORS 通配头——否则本机浏览器里任意网页
        都能跨域读 /api/*（对局史/画像）并触发写操作。默认同源策略即可；
        对外暴露属于后续鉴权议题（docs/design/security-notes.md）。"""
        req = urllib.request.Request(base_url + "/api/games", headers={"Origin": "https://evil.example"})
        with _NO_PROXY_OPENER.open(req) as resp:
            assert resp.headers.get("Access-Control-Allow-Origin") is None
            assert resp.headers.get("Access-Control-Allow-Methods") is None

    def test_dist_missing_serves_html_guide(self, base_url: str):
        """审计 B25：dist 未构建时给浏览器一页可读的三步自救引导（503），
        而非裸 JSON 报错——新手第一屏最重要的体验。"""
        req = urllib.request.Request(base_url + "/")
        try:
            with _NO_PROXY_OPENER.open(req) as resp:
                raise AssertionError(f"dist 缺失应 503，got {resp.status}")
        except urllib.error.HTTPError as err:
            assert err.code == 503
            body = err.read().decode("utf-8")
            assert "npm run build" in body and "<html" in body.lower()
            assert err.headers.get("Content-Type", "").startswith("text/html")


class TestMatch:
    def test_moon_chess_flow(self, base_url: str):
        start = _post(
            base_url + "/api/match/start",
            {
                "game_id": "moon_chess",
                "player_pid": "p_black",
                "difficulty": "easy",
            },
        )
        session = start["session"]
        assert session["over"] is False
        assert session["player_pid"] == "p_black"
        move = _post(
            base_url + "/api/match/move",
            {
                "game_id": session["game_id"],
                "action": {"cell_index": 0},
            },
        )
        assert move["ok"] is True
        assert move["session"]["board"][0] == "p_black"
        state = _post(base_url + "/api/match/state", {"game_id": session["game_id"]})
        assert state["session"]["game_id"] == session["game_id"]

    def test_gomoku_flow(self, base_url: str):
        start = _post(
            base_url + "/api/match/start",
            {
                "game_id": "stochastic_gomoku",
                "player_pid": "p_black",
                "difficulty": "easy",
            },
        )
        session = start["session"]
        move = _post(
            base_url + "/api/match/move",
            {
                "game_id": session["game_id"],
                "action": {"cell_index": 0},
            },
        )
        assert move["session"]["board"][0] == "p_black"
        assert move["session"]["last_vanish"] is None or move["session"]["last_vanish"] in range(81)

    def test_mahjong_variant_snapshot_carries_family(self, base_url: str):
        """曾缺 `_BUILTIN_FAMILY` 映射的麻将变体，快照必须带 family=mahjong。

        回归锚（与前端 InlineBoard 崩溃对齐）：sichuan/changsha/taiwan 此前
        family 为 None，前端默认按 grid 渲染 → GenericGridBoard 在
        board.length 上崩掉对话页。快照现在自描述携带 family，即使游戏目录
        尚未加载也能正确分发。
        """
        for game_id in ("mahjong_sichuan", "mahjong_changsha", "mahjong_taiwan", "mahjong_international"):
            start = _post(
                base_url + "/api/match/start",
                {
                    "game_id": game_id,
                    "player_pid": "p0",
                    "difficulty": "easy",
                    "player_count": 4,  # 麻将默认 4 人（2 人仅为引擎层显式可选）
                },
            )
            assert start["ok"] is True, game_id
            session = start["session"]
            assert session["family"] == "mahjong", game_id
            assert "board" not in session, f"{game_id} 是非 grid 快照，不应含 board"
            # 平台注册表键是随机 session id，快照 game_id 字段即该 id（对齐
            # test_moon_chess_flow 的用法），不能用真实游戏 id 查 state。
            state = _post(base_url + "/api/match/state", {"game_id": session["game_id"]})
            assert state["session"]["family"] == "mahjong", game_id

    def test_texas_fold_ends_and_records(self, base_url: str):
        start = _post(
            base_url + "/api/match/start",
            {
                "game_id": "texas_holdem",
                "player_pid": "p_sb",
                "difficulty": "easy",
            },
        )
        session = start["session"]
        assert len(session["my_hole"]) == 2
        assert session["legal"], "SB preflop must have legal actions"
        move = _post(
            base_url + "/api/match/move",
            {
                "game_id": session["game_id"],
                "action": {"choice": "fold"},
            },
        )
        assert move["session"]["over"] is True
        assert move["session"]["payoff"] is not None
        history = _get(base_url + "/api/history")
        assert any(m["match_id"] == session["game_id"] for m in history["matches"])
        detail = _get(base_url + f"/api/history/{session['game_id']}")
        assert detail["match"]["moves"], "the opening AI actions are in the log"

    def test_unknown_game_400(self, base_url: str):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                base_url + "/api/match/start",
                {
                    "game_id": "nope",
                    "player_pid": "p_black",
                    "difficulty": "easy",
                },
            )
        assert exc.value.code == 400

    def test_unknown_session_400(self, base_url: str):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(base_url + "/api/match/state", {"game_id": "deadbeef"})
        assert exc.value.code == 400

    def test_malformed_json_400(self, base_url: str):
        req = urllib.request.Request(
            base_url + "/api/match/start",
            data=b"{bad json",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            _NO_PROXY_OPENER.open(req)
        assert exc.value.code == 400


class TestBenchmark:
    def test_benchmark_flow(self, base_url: str):
        data = _post(
            base_url + "/api/benchmark/start",
            {
                "game_id": "moon_chess",
                "solver_a": "mcts",
                "solver_b": "random",
                "iterations": 1,
                "budget": 100,
            },
        )
        job_id = data["job_id"]
        status = _get(base_url + f"/api/benchmark/status?job_id={job_id}")
        assert status["job"]["job_id"] == job_id
        listing = _get(base_url + "/api/benchmark")
        assert any(j["job_id"] == job_id for j in listing["jobs"])

    def test_invalid_job_400(self, base_url: str):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                base_url + "/api/benchmark/start",
                {
                    "game_id": "texas_holdem",
                    "solver_a": "cfr",
                    "solver_b": "mcts",
                    "iterations": 2,
                },
            )
        assert exc.value.code == 400


class TestRulesTranslation:
    def test_translate_rules_api(self, base_url: str):
        data = _post(
            base_url + "/api/rules/translate",
            {
                "rule_text": "connect4 是一个 7x7 棋盘，四连成线获胜",
                "run_engine_validation": False,
            },
        )

        assert data["ok"] is True
        assert data["validation"]["valid"] is True
        assert data["rules_json"]["meta"]["family"] == "board_alignment"
        assert data["rules_json"]["constants"]["board_size"] == 7
        assert data["rules_json"]["constants"]["win_length"] == 4


class TestStatic:
    def test_unbuilt_frontend_503(self, base_url: str):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(base_url + "/")
        assert exc.value.code == 503
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(base_url + "/assets/app.js")
        assert exc.value.code == 503

    def test_unknown_api_404(self, base_url: str):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(base_url + "/api/nope")
        assert exc.value.code == 404


@pytest.fixture
def companion_url(tmp_path: pytest.TempPathFactory) -> Generator[str, None, None]:
    """Platform handler with the companion wiring enabled (D 节接线).

    Builds its own PlayManager (profiles/adaptive/agent_factory) and
    passes a ProfileStore to make_handler so the new /api routes are
    exercised end-to-end over a real HTTP server.
    """
    from layer4_interface.agent import PERSONAS, DialogueEngine
    from layer4_interface.difficulty.adaptive import AdaptiveController
    from layer4_interface.profile.store import ProfileStore

    history = MatchHistory(tmp_path / "matches")
    profiles = ProfileStore(tmp_path / "data")
    manager = PlayManager(
        provider=default_provider,
        history=history,
        seed=42,
        profiles=profiles,
        adaptive=AdaptiveController(),
        agent_factory=lambda key: DialogueEngine(PERSONAS[key]),
    )
    benchmark = BenchmarkRunner(provider=default_provider, seed=42)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(manager, history, benchmark, dist_dir=tmp_path / "no-dist", profile_store=profiles),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


class TestCompanionIntegration:
    """D 节接线回归：伴侣钩子 / 档案 / 复盘 新路由."""

    def test_match_start_forwards_persona_and_chat(self, companion_url: str):
        start = _post(
            companion_url + "/api/match/start",
            {
                "game_id": "moon_chess",
                "player_pid": "p_black",
                "difficulty": "easy",
                "player_count": 2,
                "persona": "teacher",
                "hint_level": "direction",
                "pacing": "fast",
                "adaptive": False,
            },
        )
        assert start["ok"] is True
        session = start["session"]
        assert session["chat"], "开局应产生 greet 聊天增量"
        assert session["chat"][0]["scenario"] == "greet"
        assert session["chat"][0]["text"], "兜底台词非空"
        assert session["chat"][0]["mood"] in {"happy", "thinking", "sorry", "neutral"}
        assert session["evaluation"] is not None, "应附带机械局面评估"

    def test_agent_say_and_hint(self, companion_url: str):
        start = _post(
            companion_url + "/api/match/start",
            {"game_id": "moon_chess", "player_pid": "p_black", "difficulty": "easy"},
        )
        game_id = start["session"]["game_id"]
        say = _post(companion_url + "/api/agent/say", {"game_id": game_id, "scenario": "help"})
        assert say["ok"] is True and say["message"] is not None
        assert say["message"]["text"]
        assert say["message"]["mood"] in {"happy", "thinking", "sorry", "neutral"}
        hint = _post(companion_url + "/api/match/hint", {"game_id": game_id, "level": "specific"})
        assert hint["ok"] is True
        assert hint["hint"]["level"] == "specific"
        assert hint["hint"].get("hint"), "具体建议应有文本"

    def test_profile_roundtrip_put_clear(self, companion_url: str):
        _post(companion_url + "/api/profile", {"profile": {"nickname": "阿远", "default_persona": "teacher"}})
        got = _get(companion_url + "/api/profile")
        assert got["profile"]["nickname"] == "阿远"
        assert got["profile"]["default_persona"] == "teacher"
        req = urllib.request.Request(
            companion_url + "/api/profile",
            data=json.dumps({"profile": {"theme": "dark"}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with _NO_PROXY_OPENER.open(req) as resp:
            put = json.loads(resp.read().decode("utf-8"))
        assert put["ok"] is True
        assert put["profile"]["theme"] == "dark"
        cleared = _post(companion_url + "/api/profile/clear", {})
        assert cleared["ok"] is True
        assert cleared["profile"]["nickname"] == ""

    def test_illegal_move_queues_chat(self, companion_url: str):
        start = _post(
            companion_url + "/api/match/start",
            {"game_id": "texas_holdem", "player_pid": "p_sb", "difficulty": "easy"},
        )
        game_id = start["session"]["game_id"]
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(companion_url + "/api/match/move", {"game_id": game_id, "action": {"choice": "bogus"}})
        assert exc.value.code == 400
        state = _post(companion_url + "/api/match/state", {"game_id": game_id})
        assert any(m["scenario"] == "illegal" for m in state["session"]["chat"]), "违规后应有 illegal 聊天增量"

    def test_review_endpoint_after_finish(self, companion_url: str):
        start = _post(
            companion_url + "/api/match/start",
            {"game_id": "texas_holdem", "player_pid": "p_sb", "difficulty": "easy"},
        )
        game_id = start["session"]["game_id"]
        move = _post(companion_url + "/api/match/move", {"game_id": game_id, "action": {"choice": "fold"}})
        assert move["session"]["over"] is True
        report = _get(companion_url + f"/api/review/{game_id}")
        assert report["ok"] is True
        assert report["report"]["summary"]
        assert report["report"]["key_nodes"]
        joined = (
            report["report"]["improvement"]
            + report["report"]["summary"]
            + "".join(k["why"] for k in report["report"]["key_nodes"])
        )
        assert "_bb_hole" not in joined and "底牌" not in joined, "复盘文本不得泄露对手底牌"

    def test_match_active_lists_running_session(self, companion_url: str):
        start = _post(
            companion_url + "/api/match/start",
            {"game_id": "moon_chess", "player_pid": "p_black", "difficulty": "easy", "persona": "gentle"},
        )
        game_id = start["session"]["game_id"]
        active = _get(companion_url + "/api/match/active")
        assert active["ok"] is True
        entry = next((s for s in active["sessions"] if s["game_id"] == game_id), None)
        assert entry is not None, "开局后应在活跃列表可见"
        assert entry["game"] == "moon_chess"
        assert entry["display_name"]
        assert entry["player_pid"] == "p_black"
        assert entry["difficulty"] == "easy"
        assert entry["persona"] == "gentle"
        assert entry["step"] == 0
        # 恢复契约：前端用 game_id 走 /match/state 继续
        restored = _post(companion_url + "/api/match/state", {"game_id": game_id})
        assert restored["session"]["game_id"] == game_id

    def test_match_active_drops_finished_session(self, companion_url: str):
        start = _post(
            companion_url + "/api/match/start",
            {"game_id": "texas_holdem", "player_pid": "p_sb", "difficulty": "easy"},
        )
        game_id = start["session"]["game_id"]
        _post(companion_url + "/api/match/move", {"game_id": game_id, "action": {"choice": "fold"}})
        active = _get(companion_url + "/api/match/active")
        assert all(s["game_id"] != game_id for s in active["sessions"]), "终局后不得再出现在活跃列表"


class TestChatEndpoint:
    """POST /api/chat 契约：一句 → {intent, text, mood, params}，history 可选透传。

    本类钉死确定性回退路径（monkeypatch 掉共享 LLM 单例）：开发机若恰有
    Ollama 在线，真实 LLM 对「你好」这类寒暄的分类不稳定（实测会返回
    help 而非 chat）——HTTP 冒烟测试只验证端点接线，LLM 行为由
    test_chat.py 以注入 mock 覆盖。
    """

    @pytest.fixture(autouse=True)
    def _pin_fallback_path(self, monkeypatch: pytest.MonkeyPatch):
        import layer4_interface.frontend.platform.server as server_mod

        monkeypatch.setattr(server_mod, "_get_chat_llm", lambda: None)

    def test_chat_with_history(self, base_url: str):
        data = _post(
            base_url + "/api/chat",
            {
                "text": "那月亮棋呢",
                "history": [
                    {"role": "user", "content": "我想玩德州扑克"},
                    {"role": "assistant", "content": "好，来一局德州扑克！"},
                ],
            },
        )
        assert data["ok"] is True
        assert data["intent"] in {
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
        assert isinstance(data["text"], str) and data["text"]
        assert data["mood"] in {"happy", "thinking", "sorry", "neutral"}

    def test_chat_bare_text(self, base_url: str):
        data = _post(base_url + "/api/chat", {"text": "你好"})
        assert data["ok"] is True
        assert data["intent"] == "chat"

    def test_chat_sse_accept_event_stream(self, base_url: str):
        """Accept: text/event-stream → SSE 帧序列（intent 收口 + done 结尾）。"""
        content_type, body = _post_stream(base_url + "/api/chat", {"text": "我想玩月亮棋"}, accept=True)
        assert "text/event-stream" in content_type
        frames = _parse_sse(body)
        assert frames[-1] == ("done", "{}")
        intent = next((data for event, data in frames if event == "intent"), None)
        assert intent is not None
        parsed = json.loads(intent)
        assert parsed["intent"] == "play"
        assert parsed["params"]["game_id"] == "moon_chess"
        assert parsed["text"]

    def test_chat_sse_stream_query_flag(self, base_url: str):
        """?stream=1（无 Accept 头）同样协商为 SSE。"""
        content_type, body = _post_stream(base_url + "/api/chat?stream=1", {"text": "你好"})
        assert "text/event-stream" in content_type
        frames = _parse_sse(body)
        assert any(event == "intent" for event, _ in frames)
        assert frames[-1] == ("done", "{}")

    def test_chat_json_envelope_without_stream(self, base_url: str):
        """无流式协商 → 原有 JSON 信封（向后兼容红线）。"""
        content_type, body = _post_stream(base_url + "/api/chat", {"text": "你好"})
        assert "application/json" in content_type
        parsed = json.loads(body.decode("utf-8"))
        assert parsed["ok"] is True
        assert parsed["intent"] == "chat"


@pytest.fixture
def llm_config_url(tmp_path: pytest.TempPathFactory) -> Generator[str, None, None]:
    """带平台 LLM 配置存储的服务器（独立实例，与 base_url fixture 互不影响）。"""
    history = MatchHistory(tmp_path / "matches")
    manager = PlayManager(provider=default_provider, history=history, seed=42)
    benchmark = BenchmarkRunner(provider=default_provider, seed=42)
    settings = LLMSettingsStore(tmp_path / "llm_config.json")
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            manager,
            history,
            benchmark,
            dist_dir=tmp_path / "no-dist",
            llm_settings=settings,
        ),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


class TestLlmConfigApi:
    """GET/PUT /api/llm/config 与 POST /api/llm/test 契约。"""

    @pytest.fixture(autouse=True)
    def _own_env(self, monkeypatch: pytest.MonkeyPatch):
        """PUT 会经 sync_env 写进程环境变量；先由 monkeypatch 接管这三个键
        （清空 → 生效值回落到内置默认；测试结束还原原值，避免 env 泄漏）。"""
        for key in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
            monkeypatch.delenv(key, raising=False)

    def test_get_defaults(self, llm_config_url: str):
        data = _get(llm_config_url + "/api/llm/config")
        assert data["ok"] is True
        cfg = data["config"]
        assert cfg["base_url"] == ""
        assert cfg["model"] == ""
        assert cfg["has_api_key"] is False
        assert cfg["effective_base_url"] == "http://127.0.0.1:11434"
        assert cfg["effective_model"] == "qwen3:8b"
        assert cfg["source"] == "default"

    def test_put_saves_and_get_reflects(self, llm_config_url: str):
        data = _put(
            llm_config_url + "/api/llm/config",
            {"base_url": "http://127.0.0.1:59901", "model": "m-test"},
        )
        assert data["ok"] is True
        cfg = data["config"]
        assert cfg["base_url"] == "http://127.0.0.1:59901"
        assert cfg["model"] == "m-test"
        assert cfg["source"] == "platform"
        got = _get(llm_config_url + "/api/llm/config")
        assert got["config"]["effective_base_url"] == "http://127.0.0.1:59901"
        assert got["config"]["effective_model"] == "m-test"

    def test_put_rejects_bad_scheme(self, llm_config_url: str):
        with pytest.raises(urllib.error.HTTPError) as err:
            _put(llm_config_url + "/api/llm/config", {"base_url": "not-a-url"})
        assert err.value.code == 400

    def test_api_key_omit_keeps_empty_clears(self, llm_config_url: str):
        _put(llm_config_url + "/api/llm/config", {"api_key": "sk-test"})
        assert _get(llm_config_url + "/api/llm/config")["config"]["has_api_key"] is True
        # 省略字段 → 保持不变
        _put(llm_config_url + "/api/llm/config", {"model": "m2"})
        assert _get(llm_config_url + "/api/llm/config")["config"]["has_api_key"] is True
        # 空串 → 清除
        _put(llm_config_url + "/api/llm/config", {"api_key": ""})
        assert _get(llm_config_url + "/api/llm/config")["config"]["has_api_key"] is False

    def test_llm_test_unreachable_and_bad_scheme(self, llm_config_url: str):
        data = _post(llm_config_url + "/api/llm/test", {"base_url": "http://127.0.0.1:59990"})
        assert data["ok"] is True
        assert data["reachable"] is False
        assert data["error"]
        with pytest.raises(urllib.error.HTTPError) as err:
            _post(llm_config_url + "/api/llm/test", {"base_url": "localhost:1"})
        assert err.value.code == 400
