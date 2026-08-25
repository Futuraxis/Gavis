"""Werewolf play session — human vs local-LLM players (one web table).

Each session owns one ``GameEngine`` (rules/werewolf.json, v5.2) plus one solver
handle per AI seat (local ollama, one instance per player so each AI
sees only its own partial observation).  Chance nodes (dealing,
night/vote settlement) are resolved automatically; AI turns are driven
until it is the human's turn again (or the game ends).

The human acts through ``human_move`` with the same action vocabulary
the AIs use (``speak:{intent}+text`` for speeches, ``{kill|check|...}:{target}``
for night/vote actions) — the frontend renders the legal options.

Layer contract: solvers are injected through a ``SolverProvider``
(``train-cli/games.py``), so this module holds no
``layer3_solvers`` import.  The small target/text normalization helpers
(``_target_of`` / ``_sanitize_speech``) are duplicated here deliberately
— they are action-vocabulary knowledge, and Layer 4 must not reach into
``OllamaSolver`` privates (review C1).

Usage:  ``python -m layer4_interface.frontend.play_werewolf.server``
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

from layer2_engine.core.engine import GameEngine

from ...solver_provider import SolverHandle, SolverProvider

RULES_PATH = Path(__file__).resolve().parent.parent.parent.parent / "rules" / "werewolf.json"


def _load_engine(seed: int) -> GameEngine:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return GameEngine(json.load(f), seed=seed)


def _expect_composition(engine: GameEngine, players: int, wolves: int) -> None:
    """v5.2: 配比由 JSON 声明 — 请求必须与声明一致，只做校验不注入。

    当前 werewolf.json 声明 9 人 / 3 狼 / 预言家 / 女巫 / 猎人。
    """
    pool = sorted((getattr(engine, "_constants", None) or {}).get("role_pool", []))
    expected = sorted(
        ["wolf"] * wolves
        + ["villager"] * (players - wolves - 3)
        + ["seer", "witch", "hunter"]
    )
    if pool != expected:
        raise PlayError(f"配比不受支持：规则声明 {pool} ≠ 请求 {players}人/{wolves}狼")


def _my_role_of(obs: dict) -> str | None:
    """v5.2 视图形状：my_role 是单行视图 [{role, _index}]。"""
    rows = obs.get("my_role") or []
    return str(rows[0].get("role")) if rows else None

PHASE_LABEL = {
    "night_wolf": "夜晚·狼人行动",
    "night_guard": "夜晚·守卫守人",
    "night_witch": "夜晚·女巫救/毒",
    "night_seer": "夜晚·预言家验人",
    "night_end": "夜晚结算",
    "night_hunter": "夜晚·猎人开枪",
    "day_speech": "白天·发言",
    "day_vote": "白天·投票",
    "vote_resolve": "白天·放逐结算",
    "vote_hunter": "白天·猎人开枪",
    "game_over": "游戏结束",
}

ROLE_LABEL = {"wolf": "狼人", "villager": "村民", "seer": "预言家", "witch": "女巫", "hunter": "猎人", "guard": "守卫"}

# 发言清洗：剔除控制字符（含 \x00-\x1f、\x7f）——与 layer3_solvers/llm/
# ollama_solver.py 保持一致（审计 3.6 prompt 注入修复）。
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def _target_of(params: dict) -> object:
    """Normalize an action param's target value.

    Player-entity params are dicts (``{"id": ...}``); night_witch
    ``heal``'s target is the raw ``nightKill`` string, which crashed the
    old ``.get("id")`` calls (审查 P1-15) — same logic as
    ``OllamaSolver._target_of``, kept local so Layer 4 needs no L3 import.
    """
    t = params.get("target")
    return t.get("id") if isinstance(t, dict) else t


def _sanitize_speech(speech, max_len: int = 200) -> str:
    """发言清洗：长度上限 + 剔除控制字符（审计 3.6 prompt 注入）。"""
    text = _CONTROL_CHARS_RE.sub("", str(speech or ""))
    return text[:max_len]


def _with_speech(action, text: str):
    """Attach sanitized free text to a speak action (mirrors OllamaSolver)."""
    return replace(action, params={**action.params, "text": _sanitize_speech(text)})


class PlayError(Exception):
    """Bad play request (unknown game, illegal move, game over, ...)."""


@dataclass
class GameSession:
    """One werewolf table: human seat + LLM AI seats."""

    game_id: str
    human_pid: str
    engine: GameEngine
    ai_solvers: dict[str, SolverHandle]
    model: str
    state: dict = field(init=False)
    ai_steps: int = 0  # 累计 AI 决策步数（本轮）
    ai_think_s: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.state = self.engine.create_initial_state()

    # ── State queries ──────────────────────────────────────────────

    @property
    def over(self) -> bool:
        return self.engine.is_terminal(self.state)

    @property
    def current_player(self) -> str | None:
        return self.engine.get_current_player(self.state)

    @property
    def my_turn(self) -> bool:
        return not self.over and self.current_player == self.human_pid

    @property
    def my_role(self) -> str | None:
        obs = self.engine.get_observation(self.state, self.human_pid)
        return _my_role_of(obs)

    def _resolve_chance(self) -> None:
        """Auto-apply chance nodes (dealing / settlement)."""
        while self.engine.get_node_type(self.state) == "chance":
            outs = self.engine.get_chance_outcomes(self.state)
            if not outs:
                break
            self.state = self.engine.apply_chance(self.state, outs[0])

    # ── Play actions ───────────────────────────────────────────────

    def human_move(self, template_id: str, params: dict) -> str:
        """Apply the human's action, then run all pending AI turns."""
        if self.over:
            raise PlayError("本局已结束")
        if not self.my_turn:
            raise PlayError("还没轮到你")
        action = self._find_action(template_id, params)
        if action is None:
            raise PlayError(f"非法动作: {template_id} {params}")
        self.state = self.engine.apply_action(self.state, action)
        self._resolve_chance()
        self._ai_turns()
        return action.canonical_key

    def _find_action(self, template_id: str, params: dict):
        for action in self.engine.get_legal_actions(self.state):
            if action.template_id != template_id:
                continue
            p = action.params
            if template_id == "speak":
                intent = p.get("intent", {}).get("id")
                if intent == params.get("intent"):
                    return _with_speech(action, params.get("text", ""))
                continue
            # target 归一化：heal 的 target 是 nightKill 字符串，其余是实体 dict
            if _target_of(p) == params.get("target"):
                return action
        return None

    def _ai_turns(self) -> None:
        """Drive AI players until the human's turn (or game over)."""
        import time

        while not self.over:
            self._resolve_chance()
            if self.over or self.current_player == self.human_pid:
                break
            pid = self.current_player
            solver = self.ai_solvers.get(pid)
            if solver is None:
                break
            legal = self.engine.get_legal_actions(self.state)
            if not legal:
                break
            t0 = time.time()
            action = solver.select_action(self.state)
            self.ai_think_s += time.time() - t0
            self.ai_steps += 1
            if action is None:
                import random

                action = random.choice(legal)
            self.state = self.engine.apply_action(self.state, action)
            self._resolve_chance()

    # ── Serialization ──────────────────────────────────────────────

    def snapshot(self) -> dict:
        env = self.state["env"]
        obs = self.engine.get_observation(self.state, self.human_pid)
        constants = getattr(self.engine, "_constants", None) or {}
        pids = constants.get("player_ids", [])
        alive_arr = self.state["_arrays"].get("alive", [])
        roles = self.state["_arrays"].get("roles", [])

        players = []
        for i, pid in enumerate(pids):
            alive = i < len(alive_arr) and alive_arr[i] == 1
            role = roles[i] if i < len(roles) else None
            players.append(
                {
                    "id": pid,
                    "alive": bool(alive),
                    "role": role if pid == self.human_pid else None,  # 只暴露自己的身份
                    "dead_role": ROLE_LABEL.get(role) if not alive else None,  # 死后公布身份
                }
            )

        legal = []
        if self.my_turn:
            for action in self.engine.get_legal_actions(self.state):
                p = action.params
                if action.template_id == "speak":
                    legal.append({"id": action.template_id, "intent": p.get("intent", {}).get("id")})
                else:
                    legal.append({"id": action.template_id, "target": _target_of(p)})

        phase = env.get("phase")
        return {
            "game_id": self.game_id,
            "over": self.over,
            "winner": env.get("winner"),
            "my_pid": self.human_pid,
            "my_role": ROLE_LABEL.get(_my_role_of(obs)),
            "my_turn": self.my_turn,
            "phase": phase,
            "phase_label": PHASE_LABEL.get(phase, phase),
            "round": env.get("round"),
            "players": players,
            "speech_log": list(self.state["_arrays"].get("speechLog", [])),
            "vote_log": list(self.state["_arrays"].get("voteLog", [])),
            "deaths": list(self.state["_arrays"].get("deathsArr", [])),
            "seer_result": (obs.get("env") or {}).get("seerResult"),
            "guard_last_target": (obs.get("env") or {}).get("guardLastTarget"),
            "witch_save_used": bool((obs.get("env") or {}).get("witchSaveUsed")),
            "witch_poison_used": bool((obs.get("env") or {}).get("witchPoisonUsed")),
            "legal": legal,
            "ai_steps": self.ai_steps,
        }


class PlayManager:
    """Registry of active werewolf tables (in-memory, single-process).

    Thread-safe: registry mutations are guarded by one lock; each
    session owns a per-session lock so concurrent ``move`` calls cannot
    interleave on the same game.  Finished sessions are reclaimed via
    :meth:`remove` (or FIFO eviction).
    """

    def __init__(self, provider: SolverProvider, seed: int = 42, max_sessions: int = 128) -> None:
        self._provider = provider
        self._sessions: dict[str, GameSession] = {}
        self._seed = seed
        self._max_sessions = max_sessions
        self._lock = threading.Lock()

    def start(
        self, players: int = 9, wolves: int = 3, model: str = "qwen3:8b", human_pid: str | None = None
    ) -> GameSession:
        # 参数边界（审查 P1-17）：players=0 时 uuid % players 会 ZeroDivisionError。
        if players < 3 or players > 12:
            raise PlayError(f"players 须在 3..12 之间，得到 {players}")
        if wolves < 1 or wolves >= players:
            raise PlayError(f"wolves 须在 1..{players - 1} 之间，得到 {wolves}")
        model = str(model or "").strip()
        if not model:
            raise PlayError("model 不能为空")
        if human_pid is None:
            human_pid = f"p{uuid.uuid4().int % players}"
        try:
            engine = _load_engine(self._seed)
            _expect_composition(engine, players, wolves)
        except PlayError:
            raise
        except ValueError as exc:
            # 当前生成的 rules/werewolf.json 只含 9 人/3 狼配比
            raise PlayError(f"狼人杀配比不受支持: {exc}") from exc
        constants = getattr(engine, "_constants", None) or {}
        pids = constants.get("player_ids", [])
        if human_pid not in pids:
            raise PlayError(f"unknown seat: {human_pid}")
        ai_solvers = {
            pid: self._provider.create_solver("werewolf", "ollama", engine, self._seed, 0, model=model, player_id=pid)
            for pid in pids
            if pid != human_pid
        }
        session = GameSession(
            game_id=uuid.uuid4().hex[:8],
            human_pid=human_pid,
            engine=engine,
            ai_solvers=ai_solvers,
            model=model,
        )
        session._resolve_chance()
        session._ai_turns()
        self._register(session)
        return session

    def get(self, game_id: str) -> GameSession:
        with self._lock:
            session = self._sessions.get(game_id)
        if session is None:
            raise PlayError(f"unknown game: {game_id}")
        return session

    def remove(self, game_id: str) -> None:
        with self._lock:
            self._sessions.pop(game_id, None)

    # ── Internals ──────────────────────────────────────────────────

    def _register(self, session: GameSession) -> None:
        """Register under the lock, evicting the oldest unfinished session (FIFO)."""
        with self._lock:
            while len(self._sessions) >= self._max_sessions:
                _, oldest = next(iter(self._sessions.items()))
                self._sessions.pop(oldest.game_id, None)
            self._sessions[session.game_id] = session
