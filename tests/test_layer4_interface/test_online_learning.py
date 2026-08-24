"""Tests for the Layer 4 online-learning capture pipeline.

Covers: JSON-safe conversion, JSONL store round-trips/trimming, per-match
trajectory recording (human + AI decisions), the recording solver handle,
PlayManager hook-in (stub learning manager), signal conversion, and the
layering guarantee (no layer3_solvers import inside the package).

Note on temp dirs: the harness sandbox denies pytest's own tmp machinery
(``tmp_path``), so these tests create unique dirs under ``data/`` (already
gitignored) and remove them afterwards.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

import json

from train_cli import default_provider
from layer2_engine.core.engine import GameEngine
from layer4_interface.frontend.platform.history import MatchHistory
from layer4_interface.frontend.platform.session import PlayManager
from layer4_interface.online_learning import (
    LearningStore,
    OnlineLearner,
    RecordingHandle,
    TrajectoryRecorder,
    jsonable,
    signal_from_match,
)

REPO = Path(__file__).resolve().parents[2]
RULES_DIR = REPO / "rules"


def _moon(seed: int = 42) -> GameEngine:
    with open(RULES_DIR / "moon_chess.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed)


def _texas(seed: int = 3) -> GameEngine:
    with open(RULES_DIR / "texas_holdem.json", "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed)


@pytest.fixture
def store_dir() -> Path:
    """A fresh, writable temp dir under the gitignored ``data/`` tree."""
    d = REPO / "data" / f"ol_test_{uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _module_imports(path: Path) -> list[str]:
    """Real import targets of a Python file (AST-based, ignores comments)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


# ── jsonable ──────────────────────────────────────────────────────────


class TestJsonable:
    def test_plain_structures_pass_through(self):
        value = {"env": {"round": 1, "winner": None}, "_arrays": {"board": [None, "p_black", 1, True]}}
        assert jsonable(value) == value

    def test_opaque_values_fall_back_to_str(self):
        class Opaque:
            def __str__(self) -> str:  # pragma: no cover - trivial
                return "opaque"

        result = jsonable({"x": Opaque(), "y": {"z": Opaque()}})
        assert result == {"x": "opaque", "y": {"z": "opaque"}}

    def test_tuple_becomes_list(self):
        assert jsonable((1, 2)) == [1, 2]


# ── LearningStore ─────────────────────────────────────────────────────


class TestLearningStore:
    def test_append_and_read_roundtrip(self, store_dir: Path):
        store = LearningStore(store_dir)
        store.append_match(
            "moon_chess", [{"match_id": "m1", "step": 1, "actor": "human"}], {"match_id": "m1", "terminal": True}
        )
        records = store.read_records("moon_chess")
        assert len(records) == 2
        assert records[1]["terminal"] is True

    def test_read_matches_groups_decisions_by_match(self, store_dir: Path):
        store = LearningStore(store_dir)
        store.append_match(
            "texas_holdem",
            [
                {"match_id": "a", "step": 1, "actor": "human", "player": "p_sb", "info_key": "k1"},
                {"match_id": "a", "step": 2, "actor": "ai", "player": "p_bb", "info_key": "k2"},
            ],
            {"match_id": "a", "terminal": True, "winner": "p_sb", "utilities": {"p_sb": 1.0, "p_bb": -1.0}},
        )
        store.append_match(
            "texas_holdem",
            [{"match_id": "b", "step": 1, "actor": "human", "player": "p_bb", "info_key": "k3"}],
            {"match_id": "b", "terminal": True, "winner": "p_bb"},
        )
        matches = store.read_matches("texas_holdem")
        assert [m["match_id"] for m in matches] == ["a", "b"]
        assert len(matches[0]["decisions"]) == 2
        assert matches[0]["decisions"][0]["info_key"] == "k1"
        assert matches[1]["decisions"][0]["player"] == "p_bb"

    def test_counts(self, store_dir: Path):
        store = LearningStore(store_dir)
        store.append_match(
            "moon_chess",
            [{"match_id": "a", "step": 1, "actor": "human"}, {"match_id": "a", "step": 2, "actor": "ai"}],
            {"match_id": "a", "terminal": True, "winner": None},
        )
        counts = store.counts("moon_chess")
        assert counts == {"matches": 1, "decisions": 2, "human_decisions": 1, "ai_decisions": 1}

    def test_trim_keeps_newest_matches(self, store_dir: Path):
        store = LearningStore(store_dir)
        for i in range(4):
            store.append_match(
                "moon_chess",
                [{"match_id": f"m{i}", "step": 1, "actor": "human"}],
                {"match_id": f"m{i}", "terminal": True},
            )
        dropped = store.trim("moon_chess", keep_matches=2)
        assert dropped == 2
        matches = store.read_matches("moon_chess")
        assert [m["match_id"] for m in matches] == ["m2", "m3"]

    def test_corrupt_line_is_skipped(self, store_dir: Path):
        store = LearningStore(store_dir)
        store.append_match("moon_chess", [{"match_id": "a", "actor": "human"}], {"match_id": "a", "terminal": True})
        path = store.root / "moon_chess" / "trajectories.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write("{corrupt json\n")
        assert len(store.read_records("moon_chess")) == 2

    def test_invalid_game_id_rejected(self, store_dir: Path):
        store = LearningStore(store_dir)
        with pytest.raises(Exception):
            store.append_match("../evil", [], {"terminal": True})
        with pytest.raises(Exception):
            store.read_records("a/b")

    def test_clear_removes_game_data(self, store_dir: Path):
        store = LearningStore(store_dir)
        store.append_match("moon_chess", [{"match_id": "a", "actor": "human"}], {"match_id": "a", "terminal": True})
        store.clear("moon_chess")
        assert store.read_records("moon_chess") == []
        assert "moon_chess" not in store.game_ids()


# ── TrajectoryRecorder / RecordingHandle ──────────────────────────────


def _fake_session(engine, player_pid="p_black", ai_pid="p_white"):
    return SimpleNamespace(
        game_id="sess1",
        spec=SimpleNamespace(seat_options=("p_black", "p_white")),
        player_pid=player_pid,
        ai_pid=ai_pid,
        difficulty="easy",
        engine=engine,
        state=engine.create_initial_state(),
        winner=None,
        started_at="2026-01-01T00:00:00+08:00",
    )


class TestTrajectoryRecorder:
    def test_finish_persists_decisions_and_terminal(self, store_dir: Path):
        store = LearningStore(store_dir)
        engine = _moon()
        session = _fake_session(engine)
        recorder = TrajectoryRecorder(store, "moon_chess", session.game_id, session.started_at)
        action = engine.get_legal_actions(session.state)[0]
        recorder.record_human(session, action)
        recorder.finish(session)
        matches = store.read_matches("moon_chess")
        assert len(matches) == 1
        assert len(matches[0]["decisions"]) == 1
        assert matches[0]["decisions"][0]["actor"] == "human"
        assert set(matches[0]["terminal"]["utilities"]) == {"p_black", "p_white"}

    def test_abandoned_match_persists_nothing(self, store_dir: Path):
        store = LearningStore(store_dir)
        engine = _moon()
        session = _fake_session(engine)
        recorder = TrajectoryRecorder(store, "moon_chess", session.game_id, session.started_at)
        recorder.record_human(session, engine.get_legal_actions(session.state)[0])
        # never finished -> nothing written
        assert store.read_records("moon_chess") == []

    def test_record_ai_twice_same_state_ok(self, store_dir: Path):
        store = LearningStore(store_dir)
        engine = _moon()
        session = _fake_session(engine)
        recorder = TrajectoryRecorder(store, "moon_chess", session.game_id, session.started_at)
        state = session.state
        recorder.record_ai(session, state, engine.get_legal_actions(state)[0])
        recorder.finish(session)
        assert store.counts("moon_chess")["ai_decisions"] == 1


class TestRecordingHandle:
    def test_select_action_records_and_delegates(self, store_dir: Path):
        store = LearningStore(store_dir)
        engine = _moon()
        session = _fake_session(engine)

        class Inner:
            name = "inner"

            def select_action(self, state):
                return engine.get_legal_actions(state)[0]

            def solve(self, state, **kw):
                return None

            def train(self, episodes, **kw):
                return {"episodes": episodes}

        recorder = TrajectoryRecorder(store, "moon_chess", session.game_id, session.started_at)
        handle = RecordingHandle(Inner(), recorder, session)
        state = session.state
        action = handle.select_action(state)
        assert action is not None
        assert handle.name == "inner"
        assert handle.train(3) == {"episodes": 3}
        recorder.finish(session)
        decisions = store.read_matches("moon_chess")[0]["decisions"]
        assert decisions[0]["actor"] == "ai"
        # the recorder logs the ACTUAL decider at that state (p_black opens)
        assert decisions[0]["player"] == "p_black"


# ── PlayManager hook-in (stub learning manager) ───────────────────────


class StubLearning:
    def __init__(self, store: LearningStore) -> None:
        self.store = store
        self.requests: list[str] = []

    def wrap_handle(self, session, solver):
        self.requests.append(f"wrap:{session.game_id}")
        rec = TrajectoryRecorder(self.store, session.spec.game_id, session.game_id, session.started_at)
        session.recorder = rec
        return RecordingHandle(solver, rec, session)

    def on_finished(self, session) -> None:
        self.requests.append(f"finish:{session.game_id}")
        session.recorder.finish(session)


@pytest.fixture
def learning_manager(store_dir: Path):
    store = LearningStore(store_dir / "online_learning")
    return StubLearning(store), store


def _first_legal_cell(session) -> int:
    for action in session.engine.get_legal_actions(session.state):
        cell = action.params.get("cell", {})
        idx = int(cell.get("_index", -1)) if isinstance(cell, dict) else -1
        if idx >= 0:
            return idx
    return -1


def _full_moon_game(manager: PlayManager) -> None:
    session = manager.start("moon_chess", "p_black", "easy")
    for _ in range(60):
        if session.over:
            break
        manager.move(session.game_id, {"cell_index": _first_legal_cell(session)})
    assert session.over


class TestPlayManagerHooks:
    def test_start_wraps_and_finish_records(self, store_dir: Path, learning_manager):
        learning, store = learning_manager
        manager = PlayManager(
            provider=default_provider,
            history=MatchHistory(store_dir / "matches"),
            seed=42,
            learning=learning,
        )
        _full_moon_game(manager)
        assert learning.requests[0].startswith("wrap:")
        assert learning.requests[-1].startswith("finish:")
        matches = store.read_matches("moon_chess")
        assert len(matches) == 1
        actors = {d["actor"] for d in matches[0]["decisions"]}
        assert actors == {"human", "ai"}
        assert set(matches[0]["terminal"]["utilities"]) == {"p_black", "p_white"}

    def test_no_learning_means_no_recording(self, store_dir: Path):
        manager = PlayManager(
            provider=default_provider,
            history=MatchHistory(store_dir / "matches"),
            seed=42,
        )
        session = manager.start("moon_chess", "p_black", "easy")
        manager.move(session.game_id, {"cell_index": 0})
        assert session.recorder is None

    def test_texas_holdem_captures_info_keys_and_ai_loop(self, store_dir: Path, learning_manager):
        learning, store = learning_manager
        manager = PlayManager(
            provider=default_provider,
            history=MatchHistory(store_dir / "matches"),
            seed=42,
            learning=learning,
        )
        session = manager.start("texas_holdem", "p_sb", "easy")
        # human folds immediately -> single decision, AI loop may act after
        manager.move(session.game_id, {"choice": "fold", "amount": 0})
        assert session.over
        matches = store.read_matches("texas_holdem")
        assert len(matches) == 1
        human = [d for d in matches[0]["decisions"] if d["actor"] == "human"]
        assert human and human[0]["info_key"] is not None
        assert human[0]["legal"], "legal set must be captured"


# ── Signal conversion ─────────────────────────────────────────────────


class TestSignals:
    def test_signal_from_match_win(self):
        match = {
            "terminal": {
                "match_id": "m1",
                "human_pid": "p_black",
                "winner": "p_black",
                "utilities": {"p_black": 1, "p_white": -1},
            },
            "decisions": [
                {
                    "step": 1,
                    "actor": "human",
                    "state": {"env": {"round": 0}},
                    "action": {"canonical_key": "a"},
                    "legal": ["a", "b"],
                    "info_key": "k",
                },
                {
                    "step": 2,
                    "actor": "ai",
                    "state": {},
                    "action": {"canonical_key": "b"},
                    "legal": ["b"],
                    "info_key": "k2",
                },
            ],
        }
        signal = signal_from_match("moon_chess", "mcts", match)
        assert signal.final_outcome == 1.0
        assert signal.controlled_player == "p_black"
        assert len(signal.state_sequence) == 2
        assert signal.solver_suggestions == [None, {"canonical_key": "b"}]
        assert signal.metadata["match_id"] == "m1"

    def test_signal_loss_and_draw(self):
        assert (
            signal_from_match(
                "moon_chess",
                "mcts",
                {"terminal": {"human_pid": "p_black", "winner": "p_white"}, "decisions": []},
            ).final_outcome
            == -1.0
        )
        assert (
            signal_from_match(
                "moon_chess",
                "mcts",
                {"terminal": {"human_pid": "p_black", "winner": None}, "decisions": []},
            ).final_outcome
            == 0.0
        )

    def test_online_learner_collect_match(self, store_dir: Path):
        store = LearningStore(store_dir)
        store.append_match(
            "moon_chess",
            [{"match_id": "m1", "step": 1, "actor": "human", "player": "p_black"}],
            {"match_id": "m1", "terminal": True, "human_pid": "p_black", "winner": "p_black"},
        )
        learner = OnlineLearner()
        learner.collect_match("moon_chess", "mcts", store.read_matches("moon_chess")[0])
        assert learner.size == 1
        assert learner.signals()[0].final_outcome == 1.0


# ── Published model store ─────────────────────────────────────────────


class TestOnlineModelStore:
    def test_publish_versions_and_revert(self, store_dir: Path):
        from layer4_interface.online_learning import OnlineModelStore

        store = OnlineModelStore(store_dir / "models")
        first = store.publish("texas_holdem", {"k1": {"a": 1}}, samples=5, coverage=1)
        assert first.version == 1
        store.publish("texas_holdem", {"k1": {"a": 2}}, samples=8, coverage=1)
        current = store.current("texas_holdem")
        assert current is not None and current.version == 2
        assert current.table == {"k1": {"a": 2}}
        reverted = store.revert("texas_holdem")
        assert reverted is not None and reverted.version == 1
        assert store.current("texas_holdem").table == {"k1": {"a": 1}}

    def test_reload_from_disk(self, store_dir: Path):
        from layer4_interface.online_learning import OnlineModelStore

        root = store_dir / "models"
        OnlineModelStore(root).publish("texas_holdem", {"k1": {"a": 1}}, samples=5, coverage=1)
        fresh = OnlineModelStore(root)
        assert fresh.current_table("texas_holdem") == {"k1": {"a": 1}}

    def test_status_contains_preview_without_exposing_storage(self, store_dir: Path):
        from layer4_interface.online_learning import OnlineModelStore

        store = OnlineModelStore(store_dir / "models")
        model = store.publish("texas_holdem", {"k1": {"a": 5}}, samples=5, coverage=1, gate={"episodes": 2})
        status = store.status("texas_holdem")
        assert status["version"] == 1
        assert status["samples"] == 5
        assert status["coverage"] == 1
        assert status["preview"] == [("k1", {"a": 5})]
        assert status["published_at"] == model.published_at

    def test_invalid_game_id_rejected(self, store_dir: Path):
        from layer4_interface.online_learning import OnlineModelStore

        store = OnlineModelStore(store_dir / "models")
        with pytest.raises(Exception):
            store.publish("../evil", {}, samples=0, coverage=0)

    def test_clear_removes(self, store_dir: Path):
        from layer4_interface.online_learning import OnlineModelStore

        store = OnlineModelStore(store_dir / "models")
        store.publish("texas_holdem", {"k1": {"a": 1}}, samples=1, coverage=1)
        store.clear("texas_holdem")
        assert store.current("texas_holdem") is None
        assert "texas_holdem" not in store.game_ids()


# ── LearningManager pipeline ──────────────────────────────────────────


def _seed_texas_match(store: LearningStore, match_id: str, human_decisions: list[dict]) -> None:
    decisions = []
    for i, d in enumerate(human_decisions, start=1):
        decisions.append(
            {
                "match_id": match_id,
                "step": i,
                "actor": "human",
                "player": "p_sb",
                **d,
            }
        )
    decisions.append(
        {"match_id": match_id, "step": len(decisions) + 1, "actor": "ai", "player": "p_bb", "info_key": "k_ai"}
    )
    store.append_match("texas_holdem", decisions, {"match_id": match_id, "terminal": True, "winner": "p_sb"})


def _make_manager(store_dir: Path, **kwargs):
    from layer4_interface.online_learning import LearningManager, OnlineModelStore

    store = LearningStore(store_dir / "online_learning")
    model_store = OnlineModelStore(store_dir / "online_learning" / "models")
    return LearningManager(store=store, model_store=model_store, provider=default_provider, **kwargs), store


class TestLearningManager:
    def test_build_empirical_table_counts_human_only(self, store_dir: Path):
        manager, store = _make_manager(store_dir)
        _seed_texas_match(store, "m1", [{"info_key": "k1", "action": {"canonical_key": "a"}}])
        _seed_texas_match(
            store,
            "m2",
            [
                {"info_key": "k1", "action": {"canonical_key": "a"}},
                {"info_key": "k2", "action": {"canonical_key": "b"}},
            ],
        )
        table, samples, coverage = manager.build_empirical_table("texas_holdem")
        assert samples == 3  # AI decision (k_ai) excluded
        assert coverage == 2
        assert table["k1"] == {"a": 2}
        assert table["k2"] == {"b": 1}

    def test_apply_insufficient_without_data(self, store_dir: Path):
        manager, _ = _make_manager(store_dir)
        result = manager.apply("texas_holdem")
        assert result.applied is False
        assert result.reason == "insufficient"

    def test_apply_insufficient_below_min_samples(self, store_dir: Path):
        manager, store = _make_manager(store_dir, min_samples=30)
        _seed_texas_match(store, "m1", [{"info_key": "k1", "action": {"canonical_key": "a"}}])
        result = manager.apply("texas_holdem")
        assert result.reason == "insufficient"
        assert result.samples == 1

    def test_apply_disabled_for_other_game(self, store_dir: Path):
        manager, _ = _make_manager(store_dir)
        assert manager.apply("moon_chess").reason == "disabled"

    def test_apply_ok_publishes_on_gate_pass(self, store_dir: Path):
        manager, store = _make_manager(store_dir, min_samples=1, gate_episodes=4)
        _seed_texas_match(store, "m1", [{"info_key": "k1", "action": {"canonical_key": "a"}}])
        manager._play_one = lambda *a, **k: ("c", 3)  # noqa: SLF001 — deterministic gate pass
        result = manager.apply("texas_holdem")
        assert result.applied is True
        assert result.reason == "ok"
        assert result.version == 1
        assert manager._model_store.current_table("texas_holdem") == {"k1": {"a": 1}}  # noqa: SLF001

    def test_apply_rejected_keeps_previous(self, store_dir: Path):

        manager, store = _make_manager(store_dir, min_samples=1, gate_episodes=4)
        manager._model_store.publish(  # noqa: SLF001
            "texas_holdem", {"k_old": {"x": 9}}, samples=10, coverage=1, gate={"episodes": 2}
        )
        _seed_texas_match(store, "m1", [{"info_key": "k1", "action": {"canonical_key": "a"}}])
        manager._play_one = lambda *a, **k: ("b", 3)  # noqa: SLF001 — deterministic gate fail
        result = manager.apply("texas_holdem")
        assert result.reason == "rejected"
        assert result.applied is False
        assert manager._model_store.current_table("texas_holdem") == {"k_old": {"x": 9}}  # noqa: SLF001

    def test_apply_unchanged_when_table_identical(self, store_dir: Path):
        manager, store = _make_manager(store_dir, min_samples=1, gate_episodes=2)
        _seed_texas_match(store, "m1", [{"info_key": "k1", "action": {"canonical_key": "a"}}])
        manager._play_one = lambda *a, **k: ("c", 3)  # noqa: SLF001
        first = manager.apply("texas_holdem")
        assert first.reason == "ok"
        second = manager.apply("texas_holdem")
        assert second.reason == "unchanged"
        assert second.version == 1

    def test_apply_error_is_caught(self, store_dir: Path):
        manager, _ = _make_manager(store_dir)

        def boom(game_id):
            raise RuntimeError("boom")

        manager.build_empirical_table = boom
        result = manager.apply("texas_holdem")
        assert result.reason == "error"
        assert "boom" in (result.error or "")

    def test_status_reflects_counts_and_model(self, store_dir: Path):
        manager, store = _make_manager(store_dir, min_samples=1, gate_episodes=2)
        _seed_texas_match(store, "m1", [{"info_key": "k1", "action": {"canonical_key": "a"}}])
        manager._play_one = lambda *a, **k: ("c", 2)  # noqa: SLF001
        status = manager.status("texas_holdem")
        assert status["enabled"] is True
        assert status["matches"] == 1
        assert status["human_decisions"] == 1
        assert status["model"] is None
        assert status["pending"] is True  # samples(1) >= min(1), no model yet
        manager.apply("texas_holdem")
        status = manager.status("texas_holdem")
        assert status["model"]["version"] == 1
        assert status["pending"] is False

    def test_start_auto_stops_cleanly(self, store_dir: Path):
        manager, _ = _make_manager(store_dir)
        manager.start_auto(interval_seconds=0.05)
        assert manager._auto_thread is not None and manager._auto_thread.is_alive()  # noqa: SLF001
        manager.stop_auto()
        assert manager._auto_thread is None


# ── App-layer provider injection ──────────────────────────────────────


class TestProviderInjection:
    def _engine(self):
        return _texas(3)

    def test_published_table_served_to_new_sessions(self, store_dir: Path):
        from train_cli import DefaultSolverProvider
        from layer4_interface.online_learning import OnlineModelStore

        model_store = OnlineModelStore(store_dir / "models")
        model_store.publish("texas_holdem", {"k1": {"act:fold:0": 3}}, samples=3, coverage=1)
        provider = DefaultSolverProvider(online_models=model_store)
        solver = provider.create_solver("texas_holdem", "hybrid", self._engine(), seed=1, budget=100)
        assert solver.config.opponent_model == "empirical"
        assert solver._opponent._table == {"k1": {"act:fold:0": 3}}  # noqa: SLF001

    def test_unpublished_falls_back_to_uniform(self, store_dir: Path):
        from train_cli import DefaultSolverProvider

        provider = DefaultSolverProvider()
        solver = provider.create_solver("texas_holdem", "hybrid", self._engine(), seed=1, budget=100)
        assert solver.config.opponent_model == "uniform"

    def test_explicit_kwarg_wins_over_published(self, store_dir: Path):
        from train_cli import DefaultSolverProvider
        from layer4_interface.online_learning import OnlineModelStore

        model_store = OnlineModelStore(store_dir / "models")
        model_store.publish("texas_holdem", {"published": {"a": 1}}, samples=1, coverage=1)
        provider = DefaultSolverProvider(online_models=model_store)
        solver = provider.create_solver(
            "texas_holdem", "hybrid", self._engine(), seed=1, budget=100, empirical_table={"gate": {"a": 2}}
        )
        assert solver.config.opponent_model == "empirical"
        assert solver._opponent._table == {"gate": {"a": 2}}  # noqa: SLF001


# ── End-to-end: play → learn → publish → next session uses model ─────


class TestEndToEnd:
    def test_play_then_apply_then_next_session_uses_model(self, store_dir: Path):
        from train_cli import DefaultSolverProvider
        from layer4_interface.online_learning import LearningManager, OnlineModelStore

        store = LearningStore(store_dir / "online_learning")
        model_store = OnlineModelStore(store_dir / "online_learning" / "models")
        provider = DefaultSolverProvider(online_models=model_store)
        learning = LearningManager(
            store=store, model_store=model_store, provider=provider, seed=42, min_samples=1, gate_episodes=2
        )
        manager = PlayManager(
            provider=provider, history=MatchHistory(store_dir / "matches"), seed=42, learning=learning
        )
        # one real quick match: human folds immediately
        session = manager.start("texas_holdem", "p_sb", "easy")
        manager.move(session.game_id, {"choice": "fold", "amount": 0})
        assert session.over
        assert store.counts("texas_holdem")["matches"] == 1

        # learning pipeline: candidate is the human's single decision
        learning._play_one = lambda *a, **k: ("c", 2)  # noqa: SLF001 — gate pass
        result = learning.apply("texas_holdem")
        assert result.reason == "ok"
        assert result.version == 1

        # the same provider now serves the learned model to new sessions
        solver = provider.create_solver("texas_holdem", "hybrid", _texas(1), seed=1, budget=100)
        assert solver.config.opponent_model == "empirical"
        assert solver._opponent.coverage() > 0  # noqa: SLF001


# ── Layering guarantee ────────────────────────────────────────────────


class TestLayering:
    def test_no_layer3_import_in_online_learning(self):
        root = REPO / "layer4_interface" / "online_learning"
        offenders = []
        for path in sorted(root.rglob("*.py")):
            modules = _module_imports(path)
            bad = [m for m in modules if m == "layer3_solvers" or m.startswith("layer3_solvers.") or m == "layer3"]
            if bad:
                offenders.append(f"{path.name}: {bad}")
        assert offenders == [], f"L4 online_learning must not import layer3_solvers: {offenders}"
