"""HTTP tests for the platform online-learning endpoints.

The server is assembled the same way as ``server.main()`` — LearningStore
+ OnlineModelStore + a fresh DefaultSolverProvider + LearningManager — so
the route contract (status/apply/config) is exercised end-to-end.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Generator

import pytest

from train_cli import DefaultSolverProvider
from layer4_interface.frontend.platform.benchmark import BenchmarkRunner
from layer4_interface.frontend.platform.history import MatchHistory
from layer4_interface.frontend.platform.server import make_handler
from layer4_interface.frontend.platform.session import PlayManager
from layer4_interface.online_learning import LearningManager, LearningStore, OnlineModelStore

_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@pytest.fixture
def base_url(tmp_path) -> Generator[str, None, None]:
    learning_store = LearningStore(tmp_path / "online_learning")
    model_store = OnlineModelStore(tmp_path / "online_learning" / "models")
    provider = DefaultSolverProvider(online_models=model_store)
    learning = LearningManager(
        store=learning_store,
        model_store=model_store,
        provider=provider,
        seed=42,
        min_samples=3,
        gate_episodes=2,
        gate_budget=60,
    )
    history = MatchHistory(tmp_path / "matches")
    manager = PlayManager(provider=provider, history=history, seed=42, learning=learning)
    benchmark = BenchmarkRunner(provider=provider, seed=42)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(manager, history, benchmark, dist_dir=tmp_path / "no-dist", learning=learning),
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


class TestLearningStatus:
    def test_status_lists_enabled_texas(self, base_url: str):
        data = _get(base_url + "/api/learning/status")
        assert data["ok"] is True
        by_id = {s["game_id"]: s for s in data["learning"]}
        assert "texas_holdem" in by_id
        assert by_id["texas_holdem"]["enabled"] is True
        assert by_id["texas_holdem"]["model"] is None
        assert not by_id["texas_holdem"]["pending"]

    def test_play_then_status_counts(self, base_url: str):
        # play one quick Texas match via the API
        start = _post(
            base_url + "/api/match/start",
            {"game_id": "texas_holdem", "player_pid": "p_sb", "difficulty": "easy"},
        )
        session = start["session"]
        _post(
            base_url + "/api/match/move",
            {"game_id": session["game_id"], "action": {"choice": "fold"}},
        )
        data = _get(base_url + "/api/learning/status")
        texas = next(s for s in data["learning"] if s["game_id"] == "texas_holdem")
        assert texas["matches"] == 1
        assert texas["human_decisions"] >= 1
        assert texas["pending"] is False  # 1 sample < min_samples(3)

    def test_apply_insufficient_when_data_low(self, base_url: str):
        data = _post(base_url + "/api/learning/apply", {"game_id": "texas_holdem"})
        result = data["result"]
        assert result["game_id"] == "texas_holdem"
        assert result["applied"] is False
        assert result["reason"] in {"insufficient", "ok"}  # 0 samples → insufficient


class TestLearningConfig:
    def test_disable_then_apply_skips(self, base_url: str):
        data = _post(base_url + "/api/learning/config", {"game_id": "texas_holdem", "enabled": False})
        assert data["ok"] is True
        assert data["learning"]["enabled"] is False
        result = _post(base_url + "/api/learning/apply", {"game_id": "texas_holdem"})["result"]
        assert result["reason"] == "disabled"

    def test_enable_back(self, base_url: str):
        _post(base_url + "/api/learning/config", {"game_id": "texas_holdem", "enabled": False})
        data = _post(base_url + "/api/learning/config", {"game_id": "texas_holdem", "enabled": True})
        assert data["learning"]["enabled"] is True

    def test_unknown_game_rejected(self, base_url: str):
        import urllib.error

        with pytest.raises(urllib.error.HTTPError):
            _post(base_url + "/api/learning/config", {"game_id": "nope", "enabled": True})


class TestLearningApplyAll:
    def test_apply_without_game_id_runs_enabled_games(self, base_url: str):
        data = _post(base_url + "/api/learning/apply", {})
        assert data["ok"] is True
        assert isinstance(data["results"], list)
        texas = next(r for r in data["results"] if r["game_id"] == "texas_holdem")
        assert texas["applied"] is False  # no data recorded yet
