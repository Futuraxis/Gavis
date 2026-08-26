"""Custom-game registry tests (Wave A2).

Covers the A2 deliverables end-to-end:

- family auto-discovery (``families/__init__.py``) and grid detection
- ``CustomGameStore`` / ``CustomGameRegistry`` persistence + orchestration
- registry → ``GameSpec`` → ``PlayManager`` session (start/move/AI/terminal)
  with ``board_size`` in the snapshot
- the three ``/api/custom/games`` routes over a real ``ThreadingHTTPServer``
- ``create_solver(..., allow_unknown=True)`` runtime fallback in
  ``train-cli/games.py``
- layer contract self-check (no ``layer3_solvers`` import inside
  ``layer4_interface``)

``translate_variant_rules`` is a parallel (A1) delivery and is lazily
imported from ``create()``; once it lands, the variant-mode tests here
exercise the deterministic template-parameter path against the registry
and the HTTP route.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Generator

import pytest

from layer1_translator import translate_rules_json
from layer2_engine.core.engine import GameEngine
from layer4_interface.frontend.engine_helpers import RULES_DIR, load_rules
from layer4_interface.frontend.platform.benchmark import BenchmarkRunner
from layer4_interface.frontend.platform.custom_games import (
    CustomGameError,
    CustomGameRegistry,
    CustomGameStore,
)
from layer4_interface.frontend.platform.families import FAMILY_IDS, detect_family
from layer4_interface.frontend.platform.games import GameSpec, PlayError
from layer4_interface.frontend.platform.history import MatchHistory
from layer4_interface.frontend.platform.server import make_handler
from layer4_interface.frontend.platform.session import PlayManager
from train_cli import create_solver, default_provider

CONNECT4_TEXT = "connect4：7x7 棋盘，四连即胜"
WEREWOLF_TEXT = "狼人杀：9 人局，3 狼 6 村民"

_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _translate(text: str) -> dict:
    return translate_rules_json(text, run_engine_validation=True).rules_json


def _first_legal_cell(session) -> int:
    for action in session.engine.get_legal_actions(session.state):
        cell = action.params.get("cell", {})
        idx = int(cell.get("_index", -1)) if isinstance(cell, dict) else -1
        if idx >= 0:
            return idx
    return -1


# ── Family discovery / detection ──────────────────────────────────────


class TestFamilies:
    def test_grid_family_auto_discovered(self):
        assert "grid" in FAMILY_IDS
        assert FAMILY_IDS == tuple(sorted(FAMILY_IDS))

    def test_connect4_detects_grid(self):
        rules = _translate(CONNECT4_TEXT)
        family = detect_family(rules)
        assert family is not None
        assert family.FAMILY_ID == "grid"

    def test_gomoku_template_detects_grid(self):
        rules = load_rules("stochastic_gomoku")
        assert detect_family(rules).FAMILY_ID == "grid"

    def test_texas_template_is_poker(self):
        rules = load_rules("texas_holdem")
        assert detect_family(rules).FAMILY_ID == "poker"

    def test_werewolf_detects_social(self):
        rules = _translate(WEREWOLF_TEXT)
        assert detect_family(rules).FAMILY_ID == "social"


# ── Store persistence ─────────────────────────────────────────────────


class TestCustomGameStore:
    def test_roundtrip_list_delete(self, tmp_path):
        store = CustomGameStore(tmp_path / "custom_games")
        entry = {
            "game_id": "my_game",
            "display_name": "我的游戏",
            "description": "roundtrip",
            "kind": "board",
            "family": "grid",
            "board_size": 7,
            "seat_options": ["p_black", "p_white"],
            "seat_label": "颜色",
            "player_counts": [2],
            "difficulties": ["easy", "normal", "hard"],
            "custom": True,
            "rules": {"constants": {"board_size": 7}},
            "created_at": "2026-01-01T00:00:00+08:00",
        }
        assert store.save(entry) == "my_game"
        assert store.load("my_game")["game_id"] == "my_game"
        assert [e["game_id"] for e in store.list()] == ["my_game"]
        assert store.delete("my_game") is True
        assert store.delete("my_game") is False
        with pytest.raises(CustomGameError):
            store.load("my_game")

    @pytest.mark.parametrize(
        "bad_id",
        ["../evil", "UPPER", "", "has space", "a/b", "a.b", "a" * 49],
    )
    def test_invalid_game_id_rejected(self, tmp_path, bad_id: str):
        store = CustomGameStore(tmp_path / "custom_games")
        with pytest.raises(CustomGameError):
            store.save({"game_id": bad_id})


# ── Registry orchestration ────────────────────────────────────────────


class TestCustomGameRegistry:
    @pytest.fixture
    def registry(self, tmp_path) -> CustomGameRegistry:
        return CustomGameRegistry(CustomGameStore(tmp_path / "custom_games"))

    def test_create_spec_from_scratch(self, registry):
        entry = registry.create(mode="from_scratch", rule_text=CONNECT4_TEXT, game_name="connect4")
        assert entry["game_id"] == "connect4"
        assert entry["family"] == "grid"
        assert entry["kind"] == "board"
        assert entry["board_size"] == 7
        assert entry["custom"] is True
        assert entry["validation"]["valid"] is True
        assert any("board_alignment" in w for w in entry["validation"]["warnings"])
        assert entry["diff_summary"] is None

        spec = registry.spec_for("connect4")
        assert isinstance(spec, GameSpec)
        assert spec.board_size == 7
        assert spec.kind == "board"
        assert spec.seat_options == ("p_black", "p_white")
        assert spec.difficulty_budgets["easy"] == 200
        assert registry.family_of("connect4") == "grid"
        assert registry.has("connect4")
        assert [g["game_id"] for g in registry.list_games()] == ["connect4"]

    def test_duplicate_game_id_gets_suffix(self, registry):
        first = registry.create(mode="from_scratch", rule_text=CONNECT4_TEXT, game_name="connect4")
        second = registry.create(mode="from_scratch", rule_text=CONNECT4_TEXT, game_name="connect4")
        assert first["game_id"] == "connect4"
        assert second["game_id"] == "connect4-2"

    def test_missing_rule_text_rejected(self, registry):
        with pytest.raises(CustomGameError, match="缺少规则文本"):
            registry.create(mode="from_scratch", rule_text="")

    def test_unknown_mode_rejected(self, registry):
        with pytest.raises(CustomGameError, match="未知模式"):
            registry.create(mode="bogus", rule_text="x")

    def test_variant_mode_creates_game(self, registry):
        # A1's translate_variant_rules is lazily imported on demand; the
        # deterministic template-parameter path yields a playable grid.
        entry = registry.create(
            mode="variant",
            base_game_id="stochastic_gomoku",
            change_text="棋盘改为 5x5，四连即胜",
            game_name="connect5",
        )
        assert entry["validation"]["valid"] is True
        assert entry["family"] == "grid"
        assert entry["board_size"] == 5
        assert entry["diff_summary"] is not None
        assert "constants" in entry["diff_summary"]
        spec = registry.spec_for(entry["game_id"])
        assert spec.board_size == 5

    def test_variant_unknown_base_rejected(self, registry):
        with pytest.raises(CustomGameError, match="未知基础游戏"):
            registry.create(mode="variant", base_game_id="nope", change_text="棋盘改为 5x5")

    def test_unsupported_family_rejected(self, registry, monkeypatch):
        # create() 的族拒绝分支是防御性的：当前 L1 模板面（grid/poker/mahjong/
        # social 形状）下，凡通过校验的产物必然命中一族。用 monkeypatch 使
        # detect_family 返回 None，验证该分支的消息契约与 validation 载荷。
        import layer4_interface.frontend.platform.custom_games as custom_games_mod

        monkeypatch.setattr(custom_games_mod, "detect_family", lambda rules: None)
        with pytest.raises(CustomGameError, match="该规则暂不支持平台对弈") as exc:
            registry.create(mode="from_scratch", rule_text=CONNECT4_TEXT)
        assert exc.value.validation is not None
        assert exc.value.validation.valid is False
        assert exc.value.validation.errors == ["该规则暂不支持平台对弈"]

    def test_spec_for_rejects_no_family_entry(self, registry, tmp_path):
        # 直接注入"通过校验但无族"的规则（werewolf 去掉 speak 动作后 social
        # 不识别、其余族也不识别）→ spec_for 必须明确拒绝。
        with open(RULES_DIR / "werewolf.json", encoding="utf-8") as f:
            rules = json.load(f)
        rules = {**rules, "actions": [a for a in rules["actions"] if a.get("id") != "speak"]}
        store = registry._store
        store.save(
            {
                "game_id": "no_family_game",
                "display_name": "无族测试",
                "description": "direct injection",
                "kind": "other",
                "family": None,
                "board_size": None,
                "seat_options": [],
                "seat_label": "",
                "player_counts": [],
                "difficulties": [],
                "custom": True,
                "confidence": 0.0,
                "validation": {"valid": True, "errors": [], "warnings": []},
                "diff_summary": None,
                "rules": rules,
                "created_at": "2026-01-01T00:00:00+08:00",
            }
        )
        with pytest.raises(CustomGameError, match="无法识别规则族"):
            registry.spec_for("no_family_game")


# ── Session E2E（注册 → 开局 → 落子 → AI 回手 → 终局）───────────────


class TestCustomGameSession:
    @pytest.fixture
    def manager(self, tmp_path) -> PlayManager:
        registry = CustomGameRegistry(CustomGameStore(tmp_path / "custom_games"))
        registry.create(mode="from_scratch", rule_text=CONNECT4_TEXT, game_name="connect4")
        return PlayManager(
            provider=default_provider,
            history=MatchHistory(tmp_path / "matches"),
            seed=42,
            custom=registry,
        )

    def test_start_snapshot_has_board_size(self, manager):
        session = manager.start("connect4", "p_black", "easy")
        assert session.over is False
        snap = session.snapshot()
        assert snap["board_size"] == 7
        assert snap["win_length"] == 5
        assert len(snap["board"]) == 49
        assert snap["turn"] == "p_black"
        assert session.custom is True
        assert session.family == "grid"

    def test_move_and_ai_reply(self, manager):
        session = manager.start("connect4", "p_black", "easy")
        manager.move(session.game_id, {"cell_index": 0})
        snap = session.snapshot()
        assert snap["board"][0] == "p_black"
        assert session.over or snap["turn"] == "p_black"

    def test_play_to_terminal_records_family(self, manager, tmp_path):
        session = manager.start("connect4", "p_black", "easy")
        guard = 0
        while not session.over and guard < 200:
            manager.move(session.game_id, {"cell_index": _first_legal_cell(session)})
            guard += 1
        assert session.over
        record = manager._history.get(session.game_id)  # type: ignore[union-attr]
        assert record["family"] == "grid"
        assert record["custom"] is True

    def test_unknown_custom_game_raises(self, manager):
        with pytest.raises(PlayError, match="未知游戏"):
            manager.start("no_such_custom", "p_black", "easy")


# ── HTTP routes ───────────────────────────────────────────────────────


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


@pytest.fixture
def base_url(tmp_path) -> Generator[str, None, None]:
    registry = CustomGameRegistry(CustomGameStore(tmp_path / "custom_games"))
    history = MatchHistory(tmp_path / "matches")
    manager = PlayManager(provider=default_provider, history=history, seed=42, custom=registry)
    benchmark = BenchmarkRunner(provider=default_provider, seed=42)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(manager, history, benchmark, dist_dir=tmp_path / "no-dist", custom=registry),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


class TestCustomGameHttp:
    def test_create_list_merged_delete_flow(self, base_url: str):
        created = _post(
            base_url + "/api/custom/games",
            {"mode": "from_scratch", "rule_text": CONNECT4_TEXT, "game_name": "connect4"},
        )
        assert created["ok"] is True
        assert created["game_id"] == "connect4"
        assert created["family"] == "grid"
        assert created["confidence"] > 0
        assert created["validation"]["valid"] is True
        assert created["game"]["board_size"] == 7

        listed = _get(base_url + "/api/custom/games")
        assert any(g["game_id"] == "connect4" for g in listed["games"])

        merged = _get(base_url + "/api/games")
        entry = next(g for g in merged["games"] if g["game_id"] == "connect4")
        assert entry["custom"] is True
        assert entry["family"] == "grid"
        assert entry["kind"] == "board"
        assert entry["board_size"] == 7
        builtin = next(g for g in merged["games"] if g["game_id"] == "moon_chess")
        assert builtin["custom"] is False
        assert builtin["family"] == "grid"

        deleted = _delete(base_url + "/api/custom/games/connect4")
        assert deleted["ok"] is True
        with pytest.raises(urllib.error.HTTPError) as exc:
            _delete(base_url + "/api/custom/games/connect4")
        assert exc.value.code == 404

    def test_match_e2e_over_http(self, base_url: str):
        created = _post(
            base_url + "/api/custom/games",
            {"mode": "from_scratch", "rule_text": CONNECT4_TEXT, "game_name": "connect4"},
        )
        start = _post(
            base_url + "/api/match/start",
            {"game_id": created["game_id"], "player_pid": "p_black", "difficulty": "easy"},
        )
        session = start["session"]
        assert session["board_size"] == 7
        assert session["turn"] == "p_black"
        move = _post(
            base_url + "/api/match/move",
            {"game_id": session["game_id"], "action": {"cell_index": 0}},
        )
        assert move["ok"] is True
        assert move["session"]["board"][0] == "p_black"

    def test_invalid_rules_400_with_validation(self, base_url: str):
        # L1 不识别的规则文本（非模板形状）→ 可达到的真实拒绝路径：
        # 400 + 校验载荷（中文原因），与狼人杀等受支持文本 200 形成对照。
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(base_url + "/api/custom/games", {"mode": "from_scratch", "rule_text": "石头剪刀布，三局两胜"})
        assert exc.value.code == 400
        # urllib HTTPError exposes the body via .read() on the response
        response = _read_http_error(exc.value)
        assert response["ok"] is False
        assert response["validation"]["valid"] is False
        assert response["validation"]["errors"], "应返回非空中文校验原因"

    def test_variant_mode_creates_over_http(self, base_url: str):
        created = _post(
            base_url + "/api/custom/games",
            {
                "mode": "variant",
                "base_game_id": "stochastic_gomoku",
                "change_text": "棋盘改为 5x5，四连即胜",
                "game_name": "connect5",
            },
        )
        assert created["ok"] is True
        assert created["family"] == "grid"
        assert created["validation"]["valid"] is True
        assert created["diff_summary"] is not None
        assert created["game"]["board_size"] == 5


def _read_http_error(exc: urllib.error.HTTPError) -> dict:
    body = exc.read().decode("utf-8")
    return json.loads(body)


# ── create_solver allow_unknown ───────────────────────────────────────


class TestCreateSolverAllowUnknown:
    def _engine(self) -> GameEngine:
        return GameEngine(_translate(CONNECT4_TEXT), seed=42)

    def test_allow_unknown_creates_and_selects(self):
        engine = self._engine()
        solver = create_solver("connect4", "mcts", engine, 42, 200, allow_unknown=True)
        assert solver.name.startswith("MCTS")
        action = solver.select_action(engine.create_initial_state())
        assert action is not None

    def test_provider_forwards_allow_unknown(self):
        engine = self._engine()
        solver = default_provider.create_solver("connect4", "mcts", engine, 42, 200, allow_unknown=True)
        assert solver is not None and solver.name.startswith("MCTS")

    def test_without_allow_unknown_raises(self):
        engine = self._engine()
        with pytest.raises(ValueError, match="未知游戏"):
            create_solver("connect4", "mcts", engine, 42, 200)

    def test_unknown_solver_name_still_raises(self):
        engine = self._engine()
        with pytest.raises(ValueError, match="未知求解器"):
            create_solver("connect4", "nope", engine, 42, 200, allow_unknown=True)


# ── 层契约自检 ────────────────────────────────────────────────────────


class TestLayerContract:
    def test_no_layer3_import_in_layer4(self):
        """layer4_interface 内不得 import layer3_solvers（求解器经 provider 注入）。"""
        root = Path(__file__).resolve().parents[2] / "layer4_interface"
        pattern = re.compile(r"^\s*(?:from\s+layer3_solvers|import\s+layer3_solvers)")
        hits: list[str] = []
        for path in sorted(root.rglob("*.py")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if pattern.match(line):
                    hits.append(f"{path.relative_to(root)}: {line.strip()}")
        assert not hits, "layer4_interface 内出现 layer3_solvers 导入:\n" + "\n".join(hits)
