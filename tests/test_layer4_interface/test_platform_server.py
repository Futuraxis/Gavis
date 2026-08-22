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

_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@pytest.fixture
def base_url(tmp_path: pytest.TempPathFactory) -> Generator[str, None, None]:
    history = MatchHistory(tmp_path / "matches")
    manager = PlayManager(history=history, seed=42)
    benchmark = BenchmarkRunner(seed=42)
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
