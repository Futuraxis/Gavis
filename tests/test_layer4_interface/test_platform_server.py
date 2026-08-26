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


def _get(url: str) -> dict:
    with _NO_PROXY_OPENER.open(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


class TestGames:
    def test_list_games(self, base_url: str):
        data = _get(base_url + "/api/games")
        assert data["ok"] is True
        by_id = {g["game_id"]: g for g in data["games"]}
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
        }
        assert by_id["moon_chess"]["board_size"] == 3
        assert by_id["stochastic_gomoku"]["board_size"] == 9
        assert by_id["texas_holdem"]["kind"] == "poker"
        assert "cfr" not in by_id["texas_holdem"]["solver_options"]
        assert by_id["moon_chess"]["solver_options"] == ["mcts", "cfr", "hybrid", "random"]


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
