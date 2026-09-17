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


class TestHistoryQuery:
    """`/api/history` 的筛选 / 分页 / 元数据契约（平台「对局记录」页的数据源）。

    回归背景：对局记录页从「一屏只读表格」升级为专业用户的元数据视图后，
    后端必须支撑 offset 分页与 result/q/since 服务端筛选，并在 meta 里补出
    seed / family / custom / player_count / variant；同时 `?limit=abc` 之类
    脏参数要宽容解析（旧实现 `int()` 直转 → 500）。这里锁定这些契约。
    """

    @pytest.fixture
    def seeded_url(self, tmp_path: pytest.TempPathFactory) -> Generator[str, None, None]:
        """带 4 条人造对局记录的服务器（历史页筛选/分页用例专用）。"""
        history = MatchHistory(tmp_path / "matches")
        manager = PlayManager(provider=default_provider, history=history, seed=42)
        benchmark = BenchmarkRunner(provider=default_provider, seed=42)
        records = [
            {
                "match_id": "hist_oldest",
                "game_id": "moon_chess",
                "player_pid": "p_black",
                "ai_pid": "p_white",
                "difficulty": "easy",
                "winner": "p_white",
                "won": False,
                "started_at": "2026-01-01T10:00:00+08:00",
                "finished_at": "2026-01-01T10:01:00+08:00",
                "over": True,
                "moves": [
                    {"step": 0, "snapshot": {"pids": ["p_black", "p_white"]}},
                ],
            },
            {
                # 旧记录：无 won / seed / family / teaching / ai_strength / variant；
                # 且胜者是**阵营名**（社交族），只有读末手快照的 final_roles 才能
                # 判出玩家视角胜负——正是 `_handle_history_list` 回填 won 的场景。
                "match_id": "hist_legacy",
                "game_id": "stochastic_gomoku",
                "player_pid": "p0",
                "ai_pid": "p1",
                "difficulty": "normal",
                "winner": "civilian",
                "started_at": "2026-02-01T10:00:00+08:00",
                "over": True,
                "moves": [
                    {
                        "step": 0,
                        "snapshot": {
                            "pids": ["p0", "p1"],
                            "base_variant": "classic",
                            "winner": "civilian",
                            "final_roles": [{"pid": "p0", "role": "civilian"}, {"pid": "p1", "role": "undercover"}],
                        },
                    },
                ],
            },
            {
                # 最旧的记录：连 moves 都没有（更早期的落盘格式）→ 新元数据全缺失，
                # 记录页必须按「缺就不显示」处理，服务端不得报错。
                "match_id": "hist_empty",
                "game_id": "moon_chess",
                "player_pid": "p_black",
                "ai_pid": "p_white",
                "difficulty": "easy",
                "winner": "p_black",
                "started_at": "2025-12-01T10:00:00+08:00",
                "over": True,
            },
            {
                "match_id": "hist_draw",
                "game_id": "werewolf",
                "player_pid": "p0",
                "ai_pid": "p1",
                "difficulty": "normal",
                "winner": None,
                "won": None,
                "started_at": "2026-03-01T10:00:00+08:00",
                "finished_at": "2026-03-01T10:30:00+08:00",
                "over": True,
                "moves": [],
            },
            {
                "match_id": "hist_custom",
                "game_id": "my_custom_game",
                "player_pid": "p0",
                "ai_pid": "p1",
                "difficulty": "hard",
                "winner": "p0",
                "won": True,
                "started_at": "2026-04-01T10:00:00+08:00",
                "finished_at": "2026-04-01T10:05:00+08:00",
                "over": True,
                "seed": 43,
                "family": "grid",
                "custom": True,
                "persona": "gentle",
                "hinted": True,
                "teaching": False,
                "ai_strength": 300,
                "adaptive": False,
                "moves": [
                    {"step": 0, "snapshot": {"pids": ["p0", "p1"], "variant": "four_in_row"}},
                ],
            },
        ]
        for record in records:
            history.record(record)
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(manager, history, benchmark, dist_dir=tmp_path / "no-dist"),
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
        httpd.shutdown()
        httpd.server_close()

    def test_list_envelope_and_newest_first(self, seeded_url: str):
        data = _get(seeded_url + "/api/history?limit=1")
        assert data["ok"] is True
        assert data["total"] == 5
        assert data["has_more"] is True
        assert [m["match_id"] for m in data["matches"]] == ["hist_custom"]
        # 元数据补全：顶层 seed/family/custom 进 meta，末手快照给出人数与变体。
        custom = data["matches"][0]
        assert custom["seed"] == 43
        assert custom["family"] == "grid"
        assert custom["custom"] is True
        assert custom["player_count"] == 2
        assert custom["variant"] == "four_in_row"
        assert custom["seat_names"] == {}

    def test_meta_backfills_seed_names_and_player_won(self, seeded_url: str):
        data = _get(seeded_url + "/api/history?limit=500")
        by_id = {m["match_id"]: m for m in data["matches"]}
        # 注册表游戏的座位称呼由服务端按 game_id 现算注入。
        assert by_id["hist_oldest"]["seat_names"]["p_black"] == "黑棋"
        # 旧记录缺 won 且胜者是阵营名（civilian）→ 只有末手快照的 final_roles
        # 能判出玩家视角胜负；回填后必须为 True。
        assert by_id["hist_legacy"]["won"] is True
        # 旧记录缺新字段 → 一律 None，前端按「缺就不显示」处理。
        assert by_id["hist_legacy"]["seed"] is None
        assert by_id["hist_legacy"]["family"] is None
        # 没有 variant 键时回退读快照的 base_variant。
        assert by_id["hist_legacy"]["variant"] == "classic"
        # 末手快照给了座位表（pids / hand_counts 任一）→ 人数可推。
        assert by_id["hist_oldest"]["player_count"] == 2
        assert by_id["hist_legacy"]["player_count"] == 2
        # 连 moves 都没有的更旧记录：不得报错，新字段全缺失、won 仍能按 pid 回填。
        empty = by_id["hist_empty"]
        assert empty["player_count"] is None
        assert empty["variant"] is None
        assert empty["family"] is None
        assert empty["seed"] is None
        assert empty["won"] is True  # 胜者即玩家本人 → 胜

    def test_offset_pagination_never_repeats(self, seeded_url: str):
        first = _get(seeded_url + "/api/history?limit=2&offset=0")
        second = _get(seeded_url + "/api/history?limit=2&offset=2")
        first_ids = {m["match_id"] for m in first["matches"]}
        second_ids = {m["match_id"] for m in second["matches"]}
        assert len(first_ids) == 2 and len(second_ids) == 2
        assert not (first_ids & second_ids)
        assert first["total"] == second["total"] == 5
        assert second["has_more"] is True
        # 越过末尾 → 空页，但仍然如实报告总数（前端据此停「加载更多」）。
        tail = _get(seeded_url + "/api/history?limit=2&offset=99")
        assert tail["matches"] == []
        assert tail["total"] == 5
        assert tail["has_more"] is False

    def test_dirty_pagination_params_are_tolerated(self, seeded_url: str):
        # 旧实现 `int(query["limit"])` 直转 → 这两个请求都会 500。
        fallback = _get(seeded_url + "/api/history?limit=abc&offset=xyz")
        assert fallback["ok"] is True
        assert fallback["total"] == 5
        assert len(fallback["matches"]) == 5
        clamped = _get(seeded_url + "/api/history?limit=99999&offset=-5")
        assert clamped["ok"] is True
        assert len(clamped["matches"]) == 5

    def test_result_filter_uses_player_perspective(self, seeded_url: str):
        wins = _get(seeded_url + "/api/history?result=win")
        assert wins["total"] == 3
        assert {m["match_id"] for m in wins["matches"]} == {"hist_legacy", "hist_custom", "hist_empty"}
        loses = _get(seeded_url + "/api/history?result=lose")
        assert {m["match_id"] for m in loses["matches"]} == {"hist_oldest"}
        draws = _get(seeded_url + "/api/history?result=draw")
        assert {m["match_id"] for m in draws["matches"]} == {"hist_draw"}
        # 三态互斥且并集等于全集（前端胜率统计依赖这一点）。
        assert wins["total"] + loses["total"] + draws["total"] == 5
        # 非法结果值 → 忽略该筛选（不 400、不返回空）。
        bogus = _get(seeded_url + "/api/history?result=maybe")
        assert bogus["total"] == 5

    def test_keyword_and_since_filters(self, seeded_url: str):
        by_seat = _get(seeded_url + "/api/history?q=" + urllib.parse.quote("红中"))
        assert by_seat["total"] == 0  # 未注册/无座位称呼的游戏不会伪造命中
        by_game = _get(seeded_url + "/api/history?q=werewolf")
        assert {m["match_id"] for m in by_game["matches"]} == {"hist_draw"}
        future = _get(seeded_url + "/api/history?since=2099-01-01")
        assert future["total"] == 0
        assert future["matches"] == []
        # 非法日期 → 忽略筛选而不是报错。
        bad = _get(seeded_url + "/api/history?since=not-a-date")
        assert bad["total"] == 5
        recent = _get(seeded_url + "/api/history?since=2026-03-01")
        assert {m["match_id"] for m in recent["matches"]} == {"hist_draw", "hist_custom"}

    def test_game_id_filter_still_works(self, seeded_url: str):
        data = _get(seeded_url + "/api/history?game_id=moon_chess")
        assert data["total"] == 2
        assert {m["match_id"] for m in data["matches"]} == {"hist_oldest", "hist_empty"}


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
        # 对话里改偏好走的就是「只提交变更字段」的部分写（chat 的
        # params.applied → 前端 PUT /api/profile）：必须与既有字段合并，
        # 绝不能整体覆盖掉昵称/性格。
        assert put["profile"]["nickname"] == "阿远"
        assert put["profile"]["default_persona"] == "teacher"
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

    def test_chat_style_change_carries_applied_patch(self, base_url: str):
        """HTTP 端到端：「换个风格」给选项、「换成高冷竞技」带 applied 补丁。

        这条链路曾是 UX 事故：`/api/chat` 回 `intent=settings` + 空 params →
        前端直接切到 #/settings 且吞掉回复。现在取值走 `applied`（前端写档案
        并回执、不跳页），没取值走 `clarify` + chips。
        """
        applied = _post(base_url + "/api/chat", {"text": "换成高冷竞技"})
        assert applied["intent"] == "settings"
        assert applied["params"]["applied"] == {"default_persona": "cold"}
        assert "open_page" not in applied["params"]
        ask = _post(base_url + "/api/chat", {"text": "你能换个风格吗？"})
        assert ask["intent"] == "clarify"
        assert "换成温柔陪伴" in ask["params"]["chips"]
        assert "open_page" not in ask["params"]
        opened = _post(base_url + "/api/chat", {"text": "打开设置"})
        assert opened["intent"] == "settings"
        assert opened["params"]["open_page"] is True
        assert "applied" not in opened["params"]

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


class TestMatchStream:
    """/api/match/start 与 /api/match/move 的 SSE 流式契约（?stream=1 / Accept 协商）。

    流式是显式 opt-in：不带流式协商的请求仍走原 JSON 信封（向后兼容）；
    带协商则发 ``progress{session}``（每步可见变化一帧，全游戏通用）→
    ``snapshot{session}`` → ``done{}``——AI 回合（发言/落子/出牌）逐条
    上屏，不再等服务端把整轮 AI 循环跑完才看到结果。
    """

    def test_start_stream_ai_opens_progress_then_snapshot(self, base_url: str):
        """人类坐 p_white → AI（p_black）开局先行：开局期间就推进度帧。"""
        content_type, body = _post_stream(
            base_url + "/api/match/start?stream=1",
            {"game_id": "moon_chess", "player_pid": "p_white", "difficulty": "easy"},
            accept=True,
        )
        assert "text/event-stream" in content_type
        frames = _parse_sse(body)
        events = [e for e, _ in frames]
        assert "progress" in events, "AI 先行开局必须推送逐帧进度"
        assert events[-2] == "snapshot"
        assert events[-1] == "done"
        final = json.loads(dict(frames)["snapshot"])["session"]
        assert final["game_id"]
        assert final["turn"] == "p_white"  # AI 开完局轮到人类
        # 进度帧的棋盘逐帧可见 AI 落子（不是等整段开局跑完才一次性看到）
        progress = [json.loads(d)["session"] for e, d in frames if e == "progress"]
        assert progress[0]["board"].count("") < 9 or final["over"]
        # 进度快照与终帧同属一个对局
        assert all(p["game_id"] == final["game_id"] for p in progress)

    def test_move_stream_progress_then_snapshot(self, base_url: str):
        """流式走子：人类落子后的首帧进度即包含自己的落子（动态上屏核心）。"""
        start = _post(
            base_url + "/api/match/start",
            {"game_id": "moon_chess", "player_pid": "p_white", "difficulty": "easy"},
        )
        snap = start["session"]
        session_id = snap["game_id"]
        board = snap["board"]
        free = [i for i, v in enumerate(board) if not v]
        assert free, "开局后应有空位"
        action = {"cell_index": free[0]}
        content_type, body = _post_stream(
            base_url + "/api/match/move?stream=1",
            {"game_id": session_id, "action": action},
            accept=True,
        )
        assert "text/event-stream" in content_type
        frames = _parse_sse(body)
        events = [e for e, _ in frames]
        assert "progress" in events
        assert events[-2] == "snapshot"
        assert events[-1] == "done"
        final = json.loads(dict(frames)["snapshot"])["session"]
        assert final["game_id"] == session_id
        first_progress = json.loads(next(d for e, d in frames if e == "progress"))["session"]
        assert first_progress["board"][free[0]], "人类落子必须在首帧进度里可见"

    def test_move_json_envelope_without_stream(self, base_url: str):
        """无流式协商 → 走子仍返回原 JSON 信封（向后兼容红线）。"""
        start = _post(
            base_url + "/api/match/start",
            {"game_id": "moon_chess", "player_pid": "p_white", "difficulty": "easy"},
        )
        session_id = start["session"]["game_id"]
        board = start["session"]["board"]
        free = [i for i, v in enumerate(board) if not v]
        content_type, body = _post_stream(
            base_url + "/api/match/move",
            {"game_id": session_id, "action": {"cell_index": free[0]}},
        )
        assert "application/json" in content_type
        parsed = json.loads(body.decode("utf-8"))
        assert parsed["ok"] is True
        assert parsed["session"]["game_id"] == session_id


class TestSoftSseEmitter:
    """``_soft_emitter``：客户端断开后推帧必须变成空操作，绝不能中断对局。

    回归背景（2026-09 卧底「轮不到自己就卡死」）：``on_progress`` 直接在
    ``GameSession.run_ai`` 的 AI 循环体内写 SSE。玩家刷新/关页后 socket 已断，
    ``wfile.write`` 抛 ``BrokenPipeError``/``ConnectionResetError``（均为
    ``OSError``）——异常穿过 ``run_ai`` 会**打断 AI 循环**：人类行动已落地、
    AI 只走了半步，会话停在「当前行动者是某个 AI 座位」上，玩家回来再也走不动
    （``_parse_human_action`` 抛「还没轮到你」）。杀掉这条异常路径后，对局照常
    跑完，状态始终自洽。
    """

    def test_emit_swallows_broken_pipe_and_stops_writing(self):
        from layer4_interface.frontend.platform.server import _soft_emitter

        writes: list[str] = []

        class _DeadSocket:
            def write(self, frame: bytes) -> None:
                writes.append(frame.decode("utf-8"))
                raise BrokenPipeError(32, "Broken pipe")

            def flush(self) -> None:  # pragma: no cover — write 先炸，不会走到
                raise AssertionError("flush 不应被调用")

        class _Handler:
            pass

        handler = _Handler()
        handler.wfile = _DeadSocket()  # type: ignore[attr-defined]
        handler.send_response = lambda *a, **k: None  # type: ignore[attr-defined]
        handler.send_header = lambda *a, **k: None  # type: ignore[attr-defined]

        emit = _soft_emitter(handler)  # type: ignore[arg-type]
        emit("progress", {"session": {"game_id": "g"}})  # 不抛
        emit("progress", {"session": {"game_id": "g"}})  # 断管后应短路
        emit("done", {})
        assert len(writes) == 1, "客户端断开后必须停止推帧，而不是继续写死 socket"

    def test_emit_swallows_connection_reset(self):
        from layer4_interface.frontend.platform.server import _soft_emitter

        class _Handler:
            def write(self, frame: bytes) -> None:
                raise ConnectionResetError(10054, "Connection reset by peer")

            def flush(self) -> None:
                return None

            send_response = staticmethod(lambda *a, **k: None)
            send_header = staticmethod(lambda *a, **k: None)

        handler = _Handler()
        handler.wfile = handler  # type: ignore[attr-defined]
        emit = _soft_emitter(handler)  # type: ignore[arg-type]
        emit("progress", {})  # 不抛

    def test_emit_writes_every_event_for_a_live_client(self):
        from layer4_interface.frontend.platform.server import _soft_emitter

        frames: list[str] = []

        class _LiveSocket:
            def write(self, frame: bytes) -> None:
                frames.append(frame.decode("utf-8"))

            def flush(self) -> None:
                return None

        class _Handler:
            send_response = staticmethod(lambda *a, **k: None)
            send_header = staticmethod(lambda *a, **k: None)

        handler = _Handler()
        handler.wfile = _LiveSocket()  # type: ignore[attr-defined]
        emit = _soft_emitter(handler)  # type: ignore[arg-type]
        emit("progress", {"session": {"game_id": "g"}})
        emit("snapshot", {"session": {"game_id": "g"}})
        emit("done", {})
        assert len(frames) == 3
        assert frames[0].startswith("event: progress\n")
        assert frames[2] == "event: done\ndata: {}\n\n"


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
