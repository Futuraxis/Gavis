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

    def test_create_with_llm_failure_surfaces_real_error(self, registry):
        """审查（LLM 兜底系统性排查）：``use_llm=True`` 时 LLM API/传输失败
        不再静默模板兜底 —— ``create()`` 抛 ``CustomGameError`` 且
        ``validation.errors`` 携带真实失败原因，也不产生半成品游戏。"""

        class DeadClient:
            def complete(self, messages, max_tokens=8192):  # noqa: ANN001
                raise ConnectionError("LLM 服务不可达")

        with pytest.raises(CustomGameError, match="规则校验未通过") as exc:
            registry.create(
                mode="from_scratch",
                rule_text=CONNECT4_TEXT,
                use_llm=True,
                llm_client=DeadClient(),
            )
        assert exc.value.validation is not None
        assert exc.value.validation.valid is False
        assert any("LLM 服务不可达" in e for e in exc.value.validation.errors)
        assert registry.list_games() == []  # 失败不落盘

    def test_create_variant_llm_failure_surfaces_real_error(self, registry):
        class DeadClient:
            def complete(self, messages, max_tokens=8192):  # noqa: ANN001
                raise ConnectionError("LLM API 连接失败")

        with pytest.raises(CustomGameError, match="规则校验未通过") as exc:
            registry.create(
                mode="variant",
                base_game_id="stochastic_gomoku",
                change_text="棋盘改为 5x5",
                use_llm=True,
                llm_client=DeadClient(),
            )
        assert exc.value.validation is not None
        assert exc.value.validation.valid is False
        assert any("LLM API 连接失败" in e for e in exc.value.validation.errors)

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
        """layer4_interface 内不得 import layer3_solvers（求解器经 provider 注入）。

        ``botzone/`` 是唯一刻意的例外：向 Botzone 平台提交棋子的薄适配器，
        直接驱动 Layer 3 求解器（docs/user/botzone.md 明确记载「接口再调用
        Layer 4 适配和 Layer 3 solver」）；平台前端（platform/）一律经
        ``SolverProvider`` 注入，不直连 L3。
        """
        root = Path(__file__).resolve().parents[2] / "layer4_interface"
        botzone_dir = root / "botzone"  # 文档记载的例外适配器（薄客户端→L3）
        pattern = re.compile(r"^\s*(?:from\s+layer3_solvers|import\s+layer3_solvers)")
        hits: list[str] = []
        for path in sorted(root.rglob("*.py")):
            if path.is_relative_to(botzone_dir):
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if pattern.match(line):
                    hits.append(f"{path.relative_to(root)}: {line.strip()}")
        assert not hits, "layer4_interface 内出现 layer3_solvers 导入:\n" + "\n".join(hits)


# ── 创建了却玩不了：可玩性探针 + 参数名容错 ────────────────────────


def _grid_rules_with_param(param_name: str, param_spec: dict) -> dict:
    """stochastic_gomoku 参考规则，把落子参数改名为 ``param_name``。

    平台网格族按 ``cell`` 参数定位格位；LLM 生成的规则可能叫 ``square`` /
    ``pos`` —— 这正是「创建成功但点哪都非法」的真实成因。
    """
    rules = load_rules("stochastic_gomoku")
    rules["actions"][0]["params"] = {param_name: param_spec}
    return rules


class TestActionCellIndexRobustness:
    """落子参数名容错：模型自创参数名也要能换算成格位."""

    def test_foreign_param_name_resolves(self) -> None:
        from layer2_engine.core.state_graph import ActionInstance
        from layer4_interface.frontend.platform.families.helpers import action_cell_index

        action = ActionInstance(
            template_id="place",
            type="move",
            actor_id="p_black",
            params={"square": {"_index": 10, "id": "cell_1_1", "occupant": None}},
            canonical_key="place:1,1",
        )
        assert action_cell_index(action, 9) == 10

    def test_cell_id_fallback_without_index(self) -> None:
        from layer2_engine.core.state_graph import ActionInstance
        from layer4_interface.frontend.platform.families.helpers import action_cell_index

        action = ActionInstance(
            template_id="place",
            type="move",
            actor_id="p_black",
            params={"pos": {"id": "cell_2_3"}},
            canonical_key="place:2,3",
        )
        assert action_cell_index(action, 9) == 2 * 9 + 3

    def test_unresolvable_params_yield_minus_one(self) -> None:
        from layer2_engine.core.state_graph import ActionInstance
        from layer4_interface.frontend.platform.families.helpers import action_cell_index

        action = ActionInstance(
            template_id="place", type="move", actor_id="p_black", params={"amount": 30}, canonical_key="raise:30"
        )
        assert action_cell_index(action, 9) == -1


class TestPlayabilityProbe:
    """注册前探针：能开局、有人类可执行的落子，否则拒绝注册（不产废游戏）."""

    def test_probe_accepts_renamed_cell_param(self) -> None:
        from layer4_interface.frontend.platform.families import detect_family, probe_playable

        rules = _grid_rules_with_param("square", {"view": "cell", "domain": {"ref": "empty_cells"}})
        family = detect_family(rules)
        assert family is not None
        assert probe_playable(family, rules) == []

    def test_probe_rejects_unusable_placement_param(self) -> None:
        from layer4_interface.frontend.platform.families import detect_family, probe_playable

        rules = _grid_rules_with_param("square", {"type": "int"})
        family = detect_family(rules)
        assert family is not None
        assert probe_playable(family, rules), "无法换算格位的规则必须被判不可玩"

    def test_probe_rejects_missing_board_size(self) -> None:
        from layer4_interface.frontend.platform.families import detect_family, probe_playable

        rules = load_rules("stochastic_gomoku")
        rules["constants"].pop("board_size")
        family = detect_family(rules)
        assert family is not None
        assert any("board_size" in problem for problem in probe_playable(family, rules))

    def test_unknown_family_probe_is_noop(self) -> None:
        from layer4_interface.frontend.platform.families import probe_playable

        class NoProbe:
            FAMILY_ID = "stub"

        assert probe_playable(NoProbe(), {}) == []

    def test_create_rejects_unplayable_llm_rules(self, tmp_path) -> None:
        """LLM 产出的规则若平台驱动不了，创建阶段就报错而不是落盘废游戏。"""
        registry = CustomGameRegistry(CustomGameStore(tmp_path / "custom_games"))
        rules = _grid_rules_with_param("square", {"type": "int"})

        class FakeClient:
            def complete(self, messages, max_tokens=None):  # noqa: ANN001
                return json.dumps(rules, ensure_ascii=False)

        with pytest.raises(CustomGameError, match="无法对弈") as exc:
            registry.create(
                mode="from_scratch",
                rule_text="9x9 棋盘，五子连珠获胜",
                game_name="坏棋盘",
                use_llm=True,
                llm_client=FakeClient(),
            )
        assert exc.value.validation is not None
        assert exc.value.validation.errors
        assert registry.list_games() == []  # 废游戏绝不落盘

    def test_created_game_with_renamed_param_is_playable(self, tmp_path) -> None:
        """参数名不是 cell 也能开局 + 人类落子（修复前：点哪都非法）。"""
        registry = CustomGameRegistry(CustomGameStore(tmp_path / "custom_games"))
        rules = _grid_rules_with_param("square", {"view": "cell", "domain": {"ref": "empty_cells"}})
        rules["effectors"]["do_place"]["ops"] = [
            op
            for op in rules["effectors"]["do_place"]["ops"]
            if op.get("op") != "setIndex" or op.get("array") != "board"
        ]

        class FakeClient:
            def complete(self, messages, max_tokens=None):  # noqa: ANN001
                return json.dumps(rules, ensure_ascii=False)

        entry = registry.create(
            mode="from_scratch",
            rule_text="9x9 棋盘，五子连珠获胜",
            game_name="改名棋盘",
            use_llm=True,
            llm_client=FakeClient(),
        )
        spec = registry.spec_for(entry["game_id"])
        session = PlayManager(provider=default_provider, history=None, seed=42, custom=registry).start(
            entry["game_id"], spec.seat_options[0], "easy"
        )
        legal = session.engine.get_legal_actions(session.state)
        first = next(a for a in legal if a.params)
        from layer4_interface.frontend.platform.families.helpers import action_cell_index

        index = action_cell_index(first, entry["board_size"])
        assert index >= 0
        # 人类落子经 spec.parse_human_action 走通（不再抛「非法落子」）
        action = spec.parse_human_action(session, {"cell_index": index})
        assert action is not None


class TestCreateStageProgress:
    """创建过程必须报阶段（前端据此显示「在做什么 + 等了多久」）."""

    def test_stages_reported_in_order(self, tmp_path) -> None:
        registry = CustomGameRegistry(CustomGameStore(tmp_path / "custom_games"))
        seen: list[tuple[str, str]] = []
        registry.create(
            mode="from_scratch",
            rule_text=CONNECT4_TEXT,
            game_name="staged",
            on_stage=lambda stage, detail: seen.append((stage, detail)),
        )
        assert [stage for stage, _ in seen] == ["translate", "validate", "register"]
        assert all(detail for _, detail in seen)


# ── 创建体验：SSE 阶段进度 + LLM 失败不再空手而归 ──────────────────


def _post_sse(url: str, payload: dict) -> list[tuple[str, dict]]:
    """POST ``?stream=1`` 并解析 SSE 事件（event, data）序列。"""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    events: list[tuple[str, dict]] = []
    with _NO_PROXY_OPENER.open(req) as resp:
        event = ""
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                events.append((event, json.loads(line[len("data:") :].strip())))
    return events


class TestCreateStreamHttp:
    """创建页走 SSE：阶段进度可见，失败带原因（不再「等半天什么都没有」）."""

    def test_stream_reports_stages_then_result(self, base_url: str) -> None:
        events = _post_sse(
            base_url + "/api/custom/games?stream=1",
            {"mode": "from_scratch", "rule_text": CONNECT4_TEXT, "game_name": "streamed"},
        )
        names = [name for name, _ in events]
        assert "stage" in names and names[-1] == "done"
        stages = [data["stage"] for name, data in events if name == "stage"]
        assert stages == ["translate", "validate", "register"]
        result = next(data for name, data in events if name == "result")
        assert result["ok"] is True
        assert result["game_id"] == "streamed"
        assert result["validation"]["valid"] is True

    def test_stream_failure_reports_validation_error(self, base_url: str) -> None:
        events = _post_sse(
            base_url + "/api/custom/games?stream=1",
            {"mode": "from_scratch", "rule_text": "石头剪刀布，三局两胜", "game_name": "rps"},
        )
        error = next(data for name, data in events if name == "error")
        assert error["ok"] is False
        assert error["validation"]["errors"]
        assert [name for name, _ in events][-1] == "done"


class TestCreateWithLlmFallback:
    """勾了 LLM 也绝不空手而归：端点不可达 / 翻译失败 → 确定性模板 + 醒目告警."""

    def _registry(self, tmp_path) -> CustomGameRegistry:
        return CustomGameRegistry(CustomGameStore(tmp_path / "custom_games"))

    def test_unreachable_endpoint_falls_back_with_warning(self, tmp_path, monkeypatch) -> None:
        from layer4_interface.frontend.platform import server as server_mod

        monkeypatch.setattr(server_mod, "_llm_preflight_error", lambda: "LLM 端点不可达（stub）")
        entry = server_mod._create_custom_game(
            self._registry(tmp_path),
            {"mode": "from_scratch", "rule_text": CONNECT4_TEXT, "game_name": "fallback"},
            use_llm=True,
        )
        assert entry["llm_fallback"]["used"] is True
        assert "LLM 端点不可达（stub）" in entry["llm_fallback"]["reason"]
        assert entry["validation"]["warnings"][0].startswith("⚠️ LLM 翻译未生效")
        assert entry["validation"]["valid"] is True

    def test_llm_failure_falls_back_to_template(self, tmp_path, monkeypatch) -> None:
        from layer4_interface.frontend.platform import server as server_mod

        monkeypatch.setattr(server_mod, "_llm_preflight_error", lambda: "")

        class DeadClient:
            def complete(self, messages, max_tokens=None):  # noqa: ANN001
                raise ConnectionError("boom")

        registry = self._registry(tmp_path)
        entry = server_mod._create_custom_game(
            registry,
            {
                "mode": "from_scratch",
                "rule_text": CONNECT4_TEXT,
                "game_name": "fallback2",
                "llm_client": DeadClient(),
            },
            use_llm=True,
        )
        assert entry["llm_fallback"]["used"] is True
        assert "boom" in entry["llm_fallback"]["reason"]

    def test_both_paths_failing_reports_both_reasons(self, tmp_path, monkeypatch) -> None:
        from layer4_interface.frontend.platform import server as server_mod

        monkeypatch.setattr(server_mod, "_llm_preflight_error", lambda: "LLM 端点不可达（stub）")
        with pytest.raises(CustomGameError, match="LLM 翻译失败") as exc:
            server_mod._create_custom_game(
                self._registry(tmp_path),
                {"mode": "from_scratch", "rule_text": "石头剪刀布，三局两胜", "game_name": "rps"},
                use_llm=True,
            )
        assert "确定性模板也生成不了" in str(exc.value)

    def test_use_llm_false_stays_deterministic(self, tmp_path) -> None:
        from layer4_interface.frontend.platform import server as server_mod

        entry = server_mod._create_custom_game(
            self._registry(tmp_path),
            {"mode": "from_scratch", "rule_text": CONNECT4_TEXT, "game_name": "plain"},
            use_llm=False,
        )
        assert entry.get("llm_fallback") is None

# ── 自定义游戏开局的三个红线（对话里「界面调不出来」的真凶）──────────


class TestCustomGameStartRedLines:
    """自定义游戏在平台开局路径上的红线：默认自适应不能炸、界面要立刻出现、AI 不能想太久.

    真凶（实测用户自定义 16×16 三子棋）：
    1. 对话开局的默认配置带 ``adaptive=true``（``battleConfig.ts`` 表单初值），
       而 ``AdaptiveController`` 的预算表只登记内置 ``GAMES``，对自定义 id 抛
       ``ValueError("未知游戏: …")`` → ``/match/start`` 直接失败 → 前端只显示
       「对局正在创建…」，棋盘界面永远出不来（用户原话「对话里调不出来界面」）。
    2. 流式开局的**第一帧**原本要等 AI 先手算完才推 —— 大棋盘上就是几十秒的
       空白对话页，棋盘区根本不渲染。
    3. 迭代预算管不住墙钟：16×16 板 normal=800 次要 ~55s，easy=200 次也要 ~14s。
    """

    @pytest.fixture
    def manager(self, tmp_path) -> PlayManager:
        from layer4_interface.difficulty.adaptive import AdaptiveController

        registry = CustomGameRegistry(CustomGameStore(tmp_path / "custom_games"))
        registry.create(mode="from_scratch", rule_text=CONNECT4_TEXT, game_name="connect4")
        return PlayManager(
            provider=default_provider,
            history=MatchHistory(tmp_path / "matches"),
            seed=42,
            custom=registry,
            adaptive=AdaptiveController(),
        )

    def test_adaptive_start_does_not_raise_for_custom_game(self, manager) -> None:
        """``adaptive=true``（对话开局默认）对自定义游戏必须能开局."""
        session = manager.start("connect4", "p_black", "easy", adaptive_enabled=True)
        assert session.spec.game_id == "connect4"
        assert session.adaptive_active is True
        # 自适应对自定义游戏无历史数据 → 回落到该 spec 自己的 normal 档
        assert session.ai_strength == 800

    def test_explicit_adaptive_tier_also_safe(self, manager) -> None:
        session = manager.start("connect4", "p_black", "adaptive")
        assert session.ai_strength == 800

    def test_adaptive_controller_still_strict_for_unknown_game(self) -> None:
        """控制器本身仍对未知游戏报错（红线在会话层兜底，不在控制器里静默）."""
        from layer4_interface.difficulty.adaptive import AdaptiveController

        with pytest.raises(ValueError, match="未知游戏"):
            AdaptiveController().pick_budget("some-custom-id", "adaptive", [])

    def test_grid_family_caps_search_time(self) -> None:
        """自定义网格游戏的求解器必须带时间上限（大棋盘不再一步几十秒）."""
        from layer4_interface.frontend.platform.families import grid

        rules = load_rules("stochastic_gomoku")
        spec = grid.build_spec("connect4", rules)
        engine = spec.create_engine(42)
        for difficulty, expected in (("easy", 1.5), ("normal", 3.0), ("hard", 6.0)):
            solver = spec.create_solver(default_provider, engine, 42, 200, difficulty=difficulty)
            assert solver.config.time_limit == expected, difficulty

    def test_start_stream_pushes_board_before_ai_opens(self, manager) -> None:
        """AI 先手时，第一帧必须是**空局面**（界面立刻出现），AI 落子随后补帧."""
        frames: list[dict] = []
        # seat_options[1] = p_white → ai_opens True（AI 先手）
        session = manager.start("connect4", "p_white", "easy", on_progress=frames.append)
        assert frames, "开局必须立刻推一帧，否则对话页没有棋盘可渲染"
        first = frames[0]
        assert not any(v for v in first["board"]), "第一帧应是空局面（AI 还没落子）"
        assert len(frames) >= 2, "AI 落子后应再推一帧"
        assert any(v for v in frames[-1]["board"]), "最后一帧应看到 AI 的落子"
        assert session.snapshot()["family"] == "grid"
