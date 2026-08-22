"""Werewolf play session — human vs local-LLM players (one web table).

Each session owns one ``WerewolfAdapter`` (v5.1 rules) plus one
``OllamaSolver`` per AI seat (local ollama, one instance per player so
each AI sees only its own partial observation).  Chance nodes (dealing,
night/vote settlement) are resolved automatically; AI turns are driven
until it is the human's turn again (or the game ends).

The human acts through ``human_move`` with the same action vocabulary the
AIs use (``speak:{intent}+text`` for speeches, ``{kill|check|...}:{target}``
for night/vote actions) — the frontend renders the legal options.

Usage:  ``python -m layer4_interface.frontend.play_werewolf.server``
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

from layer2_engine.games.werewolf.werewolf_adapter import WerewolfAdapter
from layer3_solvers import OllamaConfig, OllamaSolver

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


class PlayError(Exception):
    """Bad play request (unknown game, illegal move, game over, ...)."""


@dataclass
class GameSession:
    """One werewolf table: human seat + LLM AI seats."""

    game_id: str
    human_pid: str
    engine: WerewolfAdapter
    ai_solvers: dict[str, OllamaSolver]
    model: str
    state: dict = field(init=False)
    ai_steps: int = 0  # 累计 AI 决策步数（本轮）
    ai_think_s: float = 0.0

    def __post_init__(self) -> None:
        self.state = self.engine.create_initial_state()

    # ── State queries ──────────────────────────────────────────────

    @property
    def over(self) -> bool:
        return self.engine.is_terminal(self.state)

    @property
    def current_player(self) -> Optional[str]:
        return self.engine.get_current_player(self.state)

    @property
    def my_turn(self) -> bool:
        return not self.over and self.current_player == self.human_pid

    @property
    def my_role(self) -> Optional[str]:
        obs = self.engine.get_observation(self.state, self.human_pid)
        return obs.get("my_role")

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
                    return self._with_speech(action, params.get("text", ""))
                continue
            # target 归一化：heal 的 target 是 nightKill 字符串，其余是实体 dict
            if OllamaSolver._target_of(p) == params.get("target"):  # noqa: SLF001
                return action
        return None

    @staticmethod
    def _with_speech(action, text: str):
        from dataclasses import replace

        # 输入侧清洗：人类发言同样剔除控制字符并限长（审查 P2-6）
        return replace(action, params={**action.params, "text": OllamaSolver._sanitize_speech(text)})  # noqa: SLF001

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
                action = random.choice(legal)
            self.state = self.engine.apply_action(self.state, action)
            self._resolve_chance()

    # ── Serialization ──────────────────────────────────────────────

    def snapshot(self) -> dict:
        env = self.state["env"]
        obs = self.engine.get_observation(self.state, self.human_pid)
        pids = self.engine._constants.get("player_ids", [])
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
                    legal.append({"id": action.template_id, "target": OllamaSolver._target_of(p)})  # noqa: SLF001

        phase = env.get("phase")
        return {
            "game_id": self.game_id,
            "over": self.over,
            "winner": env.get("winner"),
            "my_pid": self.human_pid,
            "my_role": ROLE_LABEL.get(obs.get("my_role")),
            "my_turn": self.my_turn,
            "phase": phase,
            "phase_label": PHASE_LABEL.get(phase, phase),
            "round": env.get("round"),
            "players": players,
            "speech_log": list(self.state["_arrays"].get("speechLog", [])),
            "vote_log": list(self.state["_arrays"].get("voteLog", [])),
            "deaths": list(self.state["_arrays"].get("deathsArr", [])),
            "seer_result": obs.get("seer_result"),
            "guard_last_target": obs.get("guard_last_target"),
            "witch_save_used": bool(obs.get("witch_save_used")),
            "witch_poison_used": bool(obs.get("witch_poison_used")),
            "legal": legal,
            "ai_steps": self.ai_steps,
        }


class PlayManager:
    """Registry of active werewolf tables (in-memory, single-process)."""

    def __init__(self, seed: int = 42) -> None:
        self._sessions: dict[str, GameSession] = {}
        self._seed = seed

    def start(
        self, players: int = 9, wolves: int = 3, model: str = "qwen3:8b", human_pid: str | None = None
    ) -> GameSession:
        if human_pid is None:
            human_pid = f"p{uuid.uuid4().int % players}"
        engine = WerewolfAdapter(seed=self._seed, players=players, wolves=wolves)
        pids = engine._constants.get("player_ids", [])
        if human_pid not in pids:
            raise PlayError(f"unknown seat: {human_pid}")
        ai_solvers = {
            pid: OllamaSolver(engine, OllamaConfig(model=model), player_id=pid) for pid in pids if pid != human_pid
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
        self._sessions[session.game_id] = session
        return session

    def get(self, game_id: str) -> GameSession:
        session = self._sessions.get(game_id)
        if session is None:
            raise PlayError(f"unknown game: {game_id}")
        return session

    def remove(self, game_id: str) -> None:
        self._sessions.pop(game_id, None)
