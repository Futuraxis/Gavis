"""social family — free-text social deduction games (werewolf, undercover, …).

Detection signal: at least one action template declares a
``{"type": "text"}`` parameter (the v5.2 pre-canned speech capability)
and plays through non-``playing`` phases (night / vote / describe /
eliminate rounds).  The built spec wires the platform session for one
human seat vs an arbitrary number of AI seats:

- every AI seat gets its own solver instance
  (``provider.create_solver(..., player_id=<seat>)``) so each agent
  reasons only from its own partial projection — the archived
  play_werewolf app built one solver per seat the same way
  (``archive/legacy_play_apps/play_werewolf/session.py``);
- the solver kind is probed once at session start: ``ollama`` when the
  local Ollama server answers ``OllamaClient.available()``, else
  ``random`` — the snapshot's ``ai_mode`` records which one is live;
  ``allow_unknown=True`` because the custom game id is unregistered;
- snapshots are built **only** from ``engine.project_observation`` (the
  visibility-projected partial view), ``engine.get_legal_actions`` and
  public env fields — the hidden-information red line: another player's
  role/word/hand is never read from ``_arrays`` (werewolf.json /
  undercover.json declare ``my_role`` / ``my_word`` / ``dead_roles``
  visibility; ``seerResult`` stays env-hidden for non-seers);
- speak actions carry a ``text: ""`` placeholder (the text-parameter
  convention — text params never participate in legal enumeration and
  the effector reads them via ``$text``).

Solvers are assembled exclusively through ``SolverProvider`` — no
direct Layer-3 solver import here (layer-4 contract; grep for the
solver package name in this module finds nothing by construction).
"""

from __future__ import annotations

import random
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Callable

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.llm import LLMClient
from layer2_engine.core.state_graph import ActionInstance

from ....solver_provider import SolverHandle, SolverProvider
from ..games import GameSpec, PlayError
from .helpers import (
    declared_player_counts,
    engine_from_rules_dict,
    normalize_players,
    resolve_all_chance,
)

if TYPE_CHECKING:
    from ..session import GameSession

FAMILY_ID = "social"

#: 族默认难度预算（社交游戏以回合/发言为思考单位，比网格对弈高一档）。
DIFFICULTY_BUDGETS = {"easy": 500, "normal": 1500, "hard": 3000}

#: 快照里展示的最近发言条数上限（完整日志留在引擎投影里）。
_DISCOURSE_MAX = 12

#: 动作 template_id → 中文短标签（前端历史/复盘展示；未知回退 canonical_key）。
_ACTION_LABELS = {
    "speak": "发言",
    "vote": "投票",
    "kill": "击杀",
    "check": "查验",
    "shoot": "开枪",
    "shoot_lynched": "开枪",
    "heal": "救援",
    "poison": "下毒",
    "guard": "守护",
    "pass": "过",
}

#: 发言清洗：剔除控制字符（含 ``\\x00-\\x1f``、``\\x7f``）并限长——
#: 与 archived play_werewolf / layer3 ollama solver 的消毒策略一致
#: （audit 3.6 prompt 注入修复）。
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_SPEECH_MAX = 200


def _sanitize_speech(text: object) -> str:
    """Strip control characters and cap the speech length."""
    return _CONTROL_CHARS_RE.sub("", str(text or ""))[:_SPEECH_MAX]


def _player_counts(rules: dict, seats: tuple[str, ...]) -> tuple[int, ...]:
    """Playable player counts: the declared option when present, else the
    full seat count (mirroring the mahjong family's supplement).

    A social game's seating is fixed by its rules (role pool / word
    pairs); ``declared_player_counts`` falls back to the ``(2,)`` default
    when the rules declare no ``player_count`` option, and that default
    must not leak into the platform as the only offered count.
    """
    counts = list(declared_player_counts(rules))
    if counts == [2] and len(seats) > 2:
        return (len(seats),)
    return tuple(counts)


def _target_id_of(params: dict) -> str | None:
    """Normalize a target param to its id.

    Player-entity params are dicts (``{"id": ...}``); some rules expose
    a raw string target (werewolf ``heal``'s ``nightKill``) — same
    normalization as the archived play_werewolf ``_target_of``.
    """
    target = params.get("target")
    return target.get("id") if isinstance(target, dict) else target


def _intent_id_of(params: dict) -> str | None:
    """Intent-entity param id (werewolf ``speak`` carries an intent)."""
    intent = params.get("intent")
    return intent.get("id") if isinstance(intent, dict) else intent


def detect(rules: dict) -> bool:
    """Whether ``rules`` is a social deduction / free-speech game.

    Signal: any action template declares a ``{"type": "text"}`` parameter
    (the v5.2 speech capability) and plays through a phase other than the
    generic ``"playing"`` (werewolf's night/vote phases, undercover's
    describe/vote rounds).
    """
    actions = rules.get("actions", [])
    if not isinstance(actions, list):
        return False
    has_text_param = False
    has_other_phase = False
    for action in actions:
        if not isinstance(action, dict):
            continue
        params = action.get("params", {})
        if isinstance(params, dict) and any(
            isinstance(pdef, dict) and pdef.get("type") == "text" for pdef in params.values()
        ):
            has_text_param = True
        phases = action.get("phases", [])
        if isinstance(phases, list) and any(phase != "playing" for phase in phases):
            has_other_phase = True
    return has_text_param and has_other_phase


class _SocialSolverAssembly:
    """Per-session solver assembly captured at ``create_solver`` time.

    ``PlayManager.start`` passes the ``SolverProvider`` into the spec's
    ``create_solver`` closure exactly once; the social family needs a
    fresh per-seat solver later (``run_ai``), so this assembly pins the
    provider + engine + budget and the probed solver mode, and hands out
    per-seat handles on demand.
    """

    def __init__(
        self,
        provider: SolverProvider,
        game_id: str,
        engine: GameEngine,
        seed: int,
        budget: int,
        mode: str,
        seats: tuple[str, ...],
    ) -> None:
        self.provider = provider
        self.game_id = game_id
        self.engine = engine
        self.seed = seed
        self.budget = budget
        self.mode = mode
        self.seats = seats

    def solver_for(self, seat: str) -> SolverHandle:
        """One solver instance grounded on ``seat`` (only its own view)."""
        return self.provider.create_solver(
            self.game_id,
            self.mode,
            self.engine,
            self.seed,
            self.budget,
            allow_unknown=True,
            player_id=seat,
        )


class _SocialSolverHandle:
    """``SolverHandle`` facade for social sessions.

    The platform hands every session exactly one handle; social's
    ``run_ai`` instead drives one solver per AI seat, so this facade
    satisfies the protocol for the platform plumbing and carries the
    per-seat factory.  ``select_action`` falls back to a default seat
    (``seats[1]`` — the generic "AI seat") for any caller that uses the
    plain handle.
    """

    def __init__(self, assembly: _SocialSolverAssembly) -> None:
        self.social_assembly = assembly

    @property
    def name(self) -> str:
        return f"social/{self.social_assembly.mode}"

    def _default_seat(self) -> str:
        seats = self.social_assembly.seats
        return seats[1] if len(seats) > 1 else seats[0]

    def select_action(self, state: dict) -> ActionInstance | None:
        return self.social_assembly.solver_for(self._default_seat()).select_action(state)

    def solve(self, state: dict, **kwargs: object) -> ActionInstance | None:
        return self.social_assembly.solver_for(self._default_seat()).solve(state, **kwargs)

    def train(self, episodes: int, **kwargs: object) -> None:
        # Social solvers (ollama / random) are stateless at play time.
        return None


def _assembly_of(session: GameSession) -> _SocialSolverAssembly | None:
    """Read the assembly off the session's solver handle.

    When online learning is enabled ``PlayManager`` wraps the handle in a
    ``RecordingHandle`` (one ``_inner`` level) — unwrap that single layer;
    otherwise the handle is our own facade.
    """
    handle = session.solver
    inner = getattr(handle, "_inner", None)
    if inner is not None:
        handle = inner
    return getattr(handle, "social_assembly", None)


def _view_rows(obs: dict, hint: str) -> list[dict]:
    """Rows of the projected view whose name contains ``hint`` (``[]`` when absent)."""
    for name, rows in obs.items():
        if hint in name and isinstance(rows, list):
            return rows

    return []


def _build_snapshot(session: GameSession) -> dict:
    """Public snapshot — only from projection + legal actions + public env."""
    env = session.state.get("env", {})
    obs = session.engine.project_observation(session.state, session.player_pid)
    obs_env = obs.get("env") or {}
    seats = session.spec.seat_options
    over = session.over

    # my_role: 投影 ``my_role`` 视图只保留观看者自己的行（visibility drop 规则）。
    my_role: str | None = None
    rows = _view_rows(obs, "my_role")
    if rows and isinstance(rows[0], dict):
        my_role = rows[0].get("role")
    if my_role is not None:
        my_role = str(my_role)

    # my_word: 投影 ``my_word`` 视图（谁是卧底的词表；狼人杀无此视图 → None）。
    # 没有词，拿到"平民"也无从描述——卧底玩法的最低可玩信息。
    my_word: str | None = None
    word_rows = _view_rows(obs, "my_word")
    if word_rows and isinstance(word_rows[0], dict):
        word = word_rows[0].get("word")
        if word is not None:
            my_word = str(word)

    # alive: 投影 ``alive`` 视图（数组值公开）→ 存活玩家 id（回到座位序）。
    alive: list[str] = []
    for row in _view_rows(obs, "alive"):
        if not isinstance(row, dict):
            continue
        if row.get("alive", row.get("value")) == 1:
            index = row.get("_index")
            if isinstance(index, int) and 0 <= index < len(seats):
                alive.append(str(seats[index]))

    # discourse: 投影 ``speech_log`` 视图的最近发言（speech 记录公开）。
    discourse: list[dict] = []
    for row in _view_rows(obs, "speech")[-_DISCOURSE_MAX:]:
        entry = row.get("entry") if isinstance(row, dict) else None
        if isinstance(entry, dict):
            discourse.append(entry)

    # legal: 仅人类行动时给出；speak 折叠为一条 ``text: ""`` 占位，
    # 目标类动作给 ``target`` id（含 ``pass`` 等特殊值）。
    legal: list[dict] = []
    if not over and session.current_player == session.player_pid:
        seen_types: set[str] = set()
        seen_targets: set[tuple[str, str]] = set()
        for action in session.engine.get_legal_actions(session.state):
            if action.template_id == "speak":
                if "speak" not in seen_types:
                    legal.append({"type": "speak", "text": ""})
                    seen_types.add("speak")
                continue
            target = _target_id_of(action.params)
            if target is None:
                continue
            key = (action.template_id, target)
            if key in seen_targets:
                continue
            seen_targets.add(key)
            legal.append({"type": action.template_id, "target": target})

    assembly = _assembly_of(session)
    # ai_mode 优先读 latest 决策标注（_run_ai 每步写入；LLM 实际失败时如实
    # 降级为 "random"），未跑过 AI 的新会话回退到探测时的 assembly.mode。
    ai_mode = session.last_ai_info.get("ai_mode") or (assembly.mode if assembly is not None else "random")

    phase = obs_env.get("phase", env.get("phase"))

    # turn 脱敏（公平性红线）：夜晚/发牌/猎人开枪等私密阶段，非本人回合的
    # 行动者身份不得暴露（夜间当前行动者 = 狼人/预言家/女巫——前端高亮
    # 该座位等于官方外挂）。白天发言/投票的顺序是公开信息，照常透出；
    # 本人回合保留（前端 myTurn 依赖 turn === player_pid）。
    turn = session.current_player
    secret_phase = isinstance(phase, str) and (
        phase.startswith("night") or phase.startswith("deal") or phase == "vote_hunter"
    )
    if turn is not None and turn != session.player_pid and secret_phase:
        turn = None

    return {
        "family": "social",
        "game_id": session.game_id,
        "player_pid": session.player_pid,
        "difficulty": session.difficulty,
        "over": over,
        "winner": session.winner,
        "turn": turn,
        "phase": phase,
        "my_role": my_role,
        "my_word": my_word,
        "alive": alive,
        "discourse": discourse,
        "last_action": obs_env.get("last_action"),
        "winners": list(obs_env.get("winners") or []),
        "legal": legal,
        "ai_mode": ai_mode,
    }


def build_spec(game_id: str, rules: dict) -> GameSpec:
    """Build the platform ``GameSpec`` for a validated social rules dict.

    Args:
        game_id: The custom game id (registry-assigned, whitelisted).
        rules: Validated v5 rules JSON (social family).

    Returns:
        The ``GameSpec`` wiring engine / solver / session closures —
        snapshot keys follow the ``SocialSnapshot`` contract in
        ``platform-frontend/src/types.ts``.
    """
    meta = rules.get("meta", {}) if isinstance(rules.get("meta", {}), dict) else {}
    seats = normalize_players(rules) or ("p0",)

    def _create_engine(seed: int, player_count: int = 2) -> GameEngine:
        return engine_from_rules_dict(rules, seed, player_count=player_count)

    def _create_solver(provider: SolverProvider, engine: GameEngine, seed: int, budget: int) -> SolverHandle:
        mode = "ollama" if LLMClient.available() else "random"
        assembly = _SocialSolverAssembly(provider, game_id, engine, seed, budget, mode, seats)
        return _SocialSolverHandle(assembly)

    def _resolve_start(session: GameSession) -> None:
        """Deal roles / words (chance chain) before play starts."""
        session.state = resolve_all_chance(session.engine, session.state)

    def _ai_opens(session: GameSession) -> bool:
        """AI may open when the first actor is an AI seat.

        ［契约补充］社交开局的第一行动者不一定是首座（狼人杀夜晚由第一个
        存活狼人行动），所以这里做状态驱动的"尝试"而不是静态判断"人类非首座"：
        仅当开局当前行动者确实不是人类时才驱动 AI 循环——对以首座开局的
        社交游戏（undercover describe 从 p0 开始）与"人类非首座"等价，
        同时避免狼人杀夜晚开局卡死在 AI 回合。
        """
        current = session.current_player
        return current is not None and current != session.player_pid

    def _parse_human_action(session: GameSession, payload: dict) -> ActionInstance:
        if session.over:
            raise PlayError("本局已结束")
        if session.current_player != session.player_pid:
            raise PlayError("还没轮到你")
        action_type = payload.get("type")
        matches = [a for a in session.engine.get_legal_actions(session.state) if a.template_id == action_type]
        if not matches:
            raise PlayError(f"非法动作: {action_type} {payload}")
        if action_type == "speak":
            text = _sanitize_speech(payload.get("text"))
            intent = payload.get("intent")
            if intent is not None:
                for action in matches:
                    if _intent_id_of(action.params) == intent:
                        return replace(action, params={**action.params, "text": text})
            return replace(matches[0], params={**matches[0].params, "text": text})
        target = payload.get("target")
        for action in matches:
            if _target_id_of(action.params) == target:
                return action
        raise PlayError(f"非法动作: {action_type} {payload}")

    def _apply_human(session: GameSession, action: ActionInstance) -> None:
        session.state = session.engine.apply_action(session.state, action)
        session.state = resolve_all_chance(session.engine, session.state)

    def _run_ai(session: GameSession, on_ai_action: Callable[[ActionInstance], None] | None = None) -> None:
        """Drive every AI seat (multi-seat / multi-phase) until the human or end.

        Each AI seat gets its own solver (``player_id=<seat>``); decisions
        are recorded through ``session.recorder`` when online learning is
        enabled (mirroring ``RecordingHandle`` — the wrapped platform
        handle is bypassed because we create per-seat handles).
        """
        assembly = _assembly_of(session)
        while not session.over and session.current_player is not None and session.current_player != session.player_pid:
            seat = session.current_player
            if assembly is not None:
                solver = assembly.solver_for(seat)
                session.last_ai_info["ai_mode"] = assembly.mode
            else:
                solver = session.solver
            state = session.state
            action = solver.select_action(state)
            # 探测通过但 LLM 调用实际失败（OllamaSolver.last_call_ok=False →
            # 随机兜底）：如实标注 ai_mode，避免前端继续显示「本地大模型」。
            # 非 Ollama 求解器（random 等）没有该属性，默认 True 不受影响。
            if not getattr(solver, "last_call_ok", True):
                session.last_ai_info["ai_mode"] = "random"
            if action is None:  # solver found nothing — random fallback
                legal = session.engine.get_legal_actions(session.state)
                action = random.choice(legal) if legal else None
            if action is None:
                break
            if session.recorder is not None:
                session.recorder.record_ai(session, state, action)
            session.state = session.engine.apply_action(session.state, action)
            session.state = resolve_all_chance(session.engine, session.state)
            session.last_ai_info["action"] = action.canonical_key
            if on_ai_action is not None:
                on_ai_action(action)

    def _describe_action(action: ActionInstance) -> str:
        label = _ACTION_LABELS.get(action.template_id)
        if label is None:
            return action.canonical_key
        target = _target_id_of(action.params)
        if target == "pass":
            return "过"
        if target:
            return f"{label} {target}"
        return label

    return GameSpec(
        game_id=game_id,
        display_name=str(meta.get("gameId") or game_id),
        description=str(meta.get("description") or "") or "由规则翻译生成的社交推理游戏（social 族）",
        kind="board",
        board_size=None,
        seat_options=seats,
        seat_label="座位",
        difficulty_budgets=DIFFICULTY_BUDGETS,
        player_counts=_player_counts(rules, seats),
        create_engine=_create_engine,
        create_solver=_create_solver,
        resolve_start=_resolve_start,
        ai_opens=_ai_opens,
        parse_human_action=_parse_human_action,
        apply_human=_apply_human,
        run_ai=_run_ai,
        build_snapshot=_build_snapshot,
        describe_action=_describe_action,
    )


__all__ = ["FAMILY_ID", "detect", "build_spec"]
