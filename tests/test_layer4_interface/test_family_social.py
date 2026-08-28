"""social family runtime tests (Wave-B B3).

Covers the B3 deliverables:

- ``detect`` positive set (rules/werewolf.json, rules/undercover.json) and
  negative set (moon_chess / stochastic_gomoku / texas_holdem / mahjong);
- ``build_spec`` shape (seats, seat label, player counts, budgets, kind);
- registry-injected sessions — the rules JSON goes straight into
  ``CustomGameStore`` (mirroring test_custom_games.py's injection style)
  — then ``PlayManager.start`` → human speak move → per-seat AI replies
  (ollama unavailable → random fallback, ``ai_mode=random``) → vote moves
  to a terminal state;
- the hidden-information red line: every snapshot is built only from the
  projected observation — other players' roles/words never appear;
- ``LLMClient.available()`` probing: True → per-seat ``ollama``
  solvers (captured via a recording ``SolverProvider``), False →
  ``random``;
- illegal / out-of-turn human actions raise ``PlayError``.

pytest seed is fixed by the harness; sessions here also pin ``seed=42``
so role deals are deterministic.
"""

from __future__ import annotations

import pytest

from layer2_engine.core.engine import GameEngine
from layer4_interface.frontend.engine_helpers import load_rules
from layer4_interface.frontend.platform.custom_games import CustomGameRegistry, CustomGameStore
from layer4_interface.frontend.platform.families import detect_family
from layer4_interface.frontend.platform.families.social import LLMClient, build_spec, detect
from layer4_interface.frontend.platform.games import GameSpec, PlayError
from layer4_interface.frontend.platform.history import MatchHistory
from layer4_interface.frontend.platform.session import PlayManager
from train_cli import default_provider

#: 后端 SocialSnapshot 契约键（session.snapshot() 还会追加 chat/evaluation）。
SOCIAL_CONTRACT_KEYS = frozenset(
    {
        "family",
        "game_id",
        "player_pid",
        "difficulty",
        "over",
        "winner",
        "turn",
        "phase",
        "my_role",
        "alive",
        "discourse",
        "last_action",
        "winners",
        "legal",
        "ai_mode",
    }
)


def _seat_ids(rules: dict) -> list[str]:
    """Player ids from the rules (str or ``{"id": ...}`` entries)."""
    return [str(e["id"]) if isinstance(e, dict) else str(e) for e in rules.get("players", [])]


def _entry(game_id: str, rules: dict) -> dict:
    """A persisted registry entry (listing shape; ``spec_for`` rebuilds from rules)."""
    return {
        "game_id": game_id,
        "display_name": game_id,
        "description": "social family test game",
        "kind": "board",
        "family": "social",
        "board_size": None,
        "seat_options": _seat_ids(rules),
        "seat_label": "座位",
        "player_counts": [8],
        "difficulties": ["easy", "normal", "hard"],
        "solver_options": ["ollama", "random"],
        "custom": True,
        "confidence": 1.0,
        "validation": {"valid": True, "errors": [], "warnings": []},
        "rules": rules,
        "created_at": "2026-01-01T00:00:00+08:00",
    }


def _registry(tmp_path, game_id: str) -> CustomGameRegistry:
    store = CustomGameStore(tmp_path / "custom")
    registry = CustomGameRegistry(store)
    store.save(_entry(game_id, load_rules(game_id)))
    return registry


def _collect_strings(value: object) -> set[str]:
    """All string values of a JSON-ish structure (leak check helper)."""
    out: set[str] = set()
    if isinstance(value, str):
        out.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            out |= _collect_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            out |= _collect_strings(item)
    return out


class _ProbeSession:
    """Minimal GameSession-shaped holder for spec-closure unit checks."""

    def __init__(self, engine: GameEngine, player_pid: str = "p0") -> None:
        self.engine = engine
        self.state = engine.create_initial_state()
        self.player_pid = player_pid

    @property
    def over(self) -> bool:
        return self.engine.is_terminal(self.state)

    @property
    def current_player(self) -> str | None:
        return self.engine.get_current_player(self.state)


class _StubHandle:
    """SolverHandle stub returning the first legal action (no AI search)."""

    def __init__(self, engine: GameEngine) -> None:
        self.engine = engine

    @property
    def name(self) -> str:
        return "stub"

    def select_action(self, state: dict):
        legal = self.engine.get_legal_actions(state)
        return legal[0] if legal else None

    def solve(self, state: dict, **kwargs: object):
        return self.select_action(state)

    def train(self, episodes: int, **kwargs: object) -> None:
        return None


class _RecordingProvider:
    """SolverProvider recording every per-seat ``create_solver`` call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create_solver(self, game_id: str, name: str, engine: GameEngine, seed: int, budget: int, **kwargs: object):
        self.calls.append(
            {
                "game_id": game_id,
                "name": name,
                "player_id": kwargs.get("player_id"),
                "allow_unknown": kwargs.get("allow_unknown"),
            }
        )
        return _StubHandle(engine)


# ── 检测正/负集 ────────────────────────────────────────────────────────


class TestDetect:
    @pytest.mark.parametrize("game_id", ["werewolf", "undercover"])
    def test_social_rules_detected(self, game_id: str):
        rules = load_rules(game_id)
        assert detect(rules) is True
        family = detect_family(rules)
        assert family is not None
        assert family.FAMILY_ID == "social"

    @pytest.mark.parametrize("game_id", ["moon_chess", "stochastic_gomoku", "texas_holdem", "mahjong"])
    def test_non_social_rules_not_detected(self, game_id: str):
        assert detect(load_rules(game_id)) is False


# ── build_spec 形态 ────────────────────────────────────────────────────


class TestBuildSpec:
    def test_undercover_spec_shape(self):
        spec = build_spec("undercover", load_rules("undercover"))
        assert isinstance(spec, GameSpec)
        assert spec.kind == "board"
        assert spec.board_size is None
        assert spec.seat_label == "座位"
        assert spec.seat_options[0] == "p0"
        assert len(spec.seat_options) == 12  # rules 声明 12 座，引擎按 player_count 裁剪
        assert spec.player_counts == (8,)  # helpers 读 variants.player_count
        assert spec.difficulty_budgets == {"easy": 500, "normal": 1500, "hard": 3000}

    def test_werewolf_spec_shape(self):
        spec = build_spec("werewolf", load_rules("werewolf"))
        assert len(spec.seat_options) == 9
        assert spec.seat_options[8] == "p8"
        # werewolf variants 未声明 player_count → helpers 回退 (2,)，族内
        # _player_counts 按麻将同款先例补全为完整座位数（配比由 role_pool
        # 唯一决定，平台只提供 9 人一档）。
        assert spec.player_counts == (9,)

    def test_spec_resolve_parse_apply(self):
        spec = build_spec("undercover", load_rules("undercover"))
        engine = spec.create_engine(42, player_count=8)
        session = _ProbeSession(engine, "p0")
        spec.resolve_start(session)
        assert session.state["env"]["phase"] == "describe"
        assert session.current_player == "p0"
        legal = engine.get_legal_actions(session.state)
        assert any(a.template_id == "speak" and a.params.get("text") == "" for a in legal)
        action = spec.parse_human_action(session, {"type": "speak", "text": "我来了"})
        assert action.template_id == "speak"
        assert action.params["text"] == "我来了"
        spec.apply_human(session, action)
        speech = session.state["_arrays"]["speechLog"]
        assert any(entry.get("speaker") == "p0" and entry.get("text") == "我来了" for entry in speech)


# ── 会话 E2E（注册 → 开局 → 发言 → AI 回手 → 投票 → 终局）──────────────


@pytest.fixture
def manager(tmp_path, monkeypatch):
    # Ollama 探测按 CI 环境强制不可用 → 每 AI 座位回退 random 求解器。
    monkeypatch.setattr(LLMClient, "available", staticmethod(lambda: False))
    registry = _registry(tmp_path, "undercover")
    return PlayManager(
        provider=default_provider,
        history=MatchHistory(tmp_path / "matches"),
        seed=42,
        custom=registry,
    )


class TestUndercoverSession:
    def _check_snapshot(self, snap: dict, roles: list, words: list, mine: str, index: int) -> None:
        """Contract + hidden-information red-line assertions for one snapshot."""
        assert snap["family"] == "social"
        assert set(snap) - {"chat", "evaluation"} == SOCIAL_CONTRACT_KEYS
        assert snap["my_role"] == roles[index]
        assert snap["ai_mode"] in {"ollama", "random"}
        strings = _collect_strings(snap)
        # 公开标签（自己的角色 / 终局胜方标签）可合法出现——其余他人角色值不得进入快照。
        public = {snap["my_role"], snap["winner"]}
        for i, role in enumerate(roles):
            if i == index or role == snap["my_role"]:
                continue  # 同角色字符串以 my_role 合法出现
            assert role not in strings or role in public, f"他人角色泄露: {role}"
        for i, word in enumerate(words):
            if i == index:
                continue
            assert word not in strings, f"他人词卡泄露: {word}"

    def test_play_to_terminal_and_no_role_leak(self, manager):
        session = manager.start("undercover", "p0", "easy", player_count=8)
        assert session.custom is True
        assert session.family == "social"
        roles = list(session.state["_arrays"]["roles"])
        words = list(session.state["_arrays"]["words"])
        assert len(roles) == len(words) == 8

        guard = 0
        while not session.over and guard < 60:
            if session.current_player != session.player_pid:
                break  # 不应发生：每步后 AI 完成回手，轮到人类或已终局
            snap = session.snapshot()
            self._check_snapshot(snap, roles, words, "p0", 0)
            legal = snap["legal"]
            assert legal, f"人类回合无合法动作: phase={snap['phase']}"
            first = legal[0]
            if first["type"] == "speak":
                manager.move(session.game_id, {"type": "speak", "text": f"测试发言{guard}"})
            else:
                assert first["target"] is not None
                manager.move(session.game_id, {"type": first["type"], "target": first["target"]})
            guard += 1

        assert session.over, f"未能终局（{guard} 步）"
        final = session.snapshot()
        self._check_snapshot(final, roles, words, "p0", 0)
        assert final["over"] is True
        assert final["winner"] is not None
        assert final["winners"] == []

    def test_human_speech_enters_discourse(self, manager):
        session = manager.start("undercover", "p0", "easy", player_count=8)
        result = manager.move(session.game_id, {"type": "speak", "text": "我很喜欢这种水果"})
        assert result["ai_mode"] == "random"
        assert any(
            entry.get("speaker") == "p0" and entry.get("text") == "我很喜欢这种水果" for entry in result["discourse"]
        ), "人类发言未进入公开发言记录"

    def test_illegal_action_raises(self, manager):
        session = manager.start("undercover", "p0", "easy", player_count=8)
        with pytest.raises(PlayError, match="非法动作"):
            manager.move(session.game_id, {"type": "vote", "target": "p1"})  # describe 阶段无投票

    def test_not_your_turn_raises_when_ai_opens(self, manager):
        session = manager.start("undercover", "p3", "easy", player_count=8)
        # AI 先发言 p0..p2 后停手 → 人类 p3 回合（非首座座位开局由 AI 先行）。
        assert session.current_player == "p3"
        session.state["env"]["turn"] = "p1"  # 人为拨回 AI 座位
        with pytest.raises(PlayError, match="还没轮到你"):
            manager.move(session.game_id, {"type": "speak", "text": "越位发言"})


# ── Ollama 探测：可用 → 每座位 ollama；不可用 → 每座位 random ──────────


class TestOllamaProbe:
    def test_ollama_available_uses_per_seat_ollama(self, tmp_path, monkeypatch):
        monkeypatch.setattr(LLMClient, "available", staticmethod(lambda: True))
        provider = _RecordingProvider()
        manager = PlayManager(provider=provider, seed=42, custom=_registry(tmp_path, "undercover"))
        session = manager.start("undercover", "p0", "easy", player_count=8)
        assert session.snapshot()["ai_mode"] == "ollama"
        result = manager.move(session.game_id, {"type": "speak", "text": "大家好"})
        assert provider.calls, "AI 席位未创建求解器"
        for call in provider.calls:
            assert call["name"] == "ollama"
            assert call["allow_unknown"] is True
            assert call["player_id"] in {f"p{i}" for i in range(1, 8)}
        assert result["ai_mode"] == "ollama"

    def test_ollama_unavailable_falls_back_random(self, tmp_path, monkeypatch):
        monkeypatch.setattr(LLMClient, "available", staticmethod(lambda: False))
        provider = _RecordingProvider()
        manager = PlayManager(provider=provider, seed=42, custom=_registry(tmp_path, "undercover"))
        session = manager.start("undercover", "p0", "easy", player_count=8)
        result = manager.move(session.game_id, {"type": "speak", "text": "大家好"})
        assert result["ai_mode"] == "random"
        assert provider.calls
        assert all(call["name"] == "random" for call in provider.calls)


# ── 狼人杀冒烟（夜晚由 AI 先行 / 快照红线条目）────────────────────────


class TestWerewolfSmoke:
    def test_start_and_snapshot_red_line(self, tmp_path, monkeypatch):
        monkeypatch.setattr(LLMClient, "available", staticmethod(lambda: False))
        registry = _registry(tmp_path, "werewolf")
        manager = PlayManager(
            provider=default_provider,
            history=MatchHistory(tmp_path / "matches"),
            seed=42,
            custom=registry,
        )
        session = manager.start("werewolf", "p0", "easy", player_count=9)
        assert session.family == "social"
        roles = list(session.state["_arrays"]["roles"])
        snap = session.snapshot()
        assert set(snap) - {"chat", "evaluation"} == SOCIAL_CONTRACT_KEYS
        assert snap["my_role"] == roles[0]
        if not snap["over"]:
            assert len(snap["alive"]) >= 1  # 存活列表为公开投影
        strings = _collect_strings(snap)
        public = {snap["my_role"], snap["winner"]}
        for i, role in enumerate(roles):
            if i == 0 or role == snap["my_role"]:
                continue
            assert role not in strings or role in public, f"他人角色泄露: {role}"

        # 本测试 seed 下人类（p0 村民）首夜出局 → 全部座位都是 AI，开局
        # ai_opens 直接驱动到终局；若人类已轮到（人类是狼/神职）则走一步。
        if session.current_player == session.player_pid and not session.over:
            first = snap["legal"][0]
            if first["type"] == "speak":
                manager.move(session.game_id, {"type": "speak", "text": "大家好"})
            else:
                manager.move(session.game_id, {"type": first["type"], "target": first["target"]})
        after = session.snapshot()
        assert after["family"] == "social"
        assert after["ai_mode"] == "random"
        if after["over"]:
            assert after["winner"] is not None  # 终局胜方来自公开 env.winner
