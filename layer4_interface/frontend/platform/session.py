"""Generic human-vs-AI session management for the platform frontend.

``GameSession`` is game-agnostic: all per-game behaviour (action
parsing, chance resolution, AI turn loop, snapshot serialization) is
provided by the ``GameSpec`` registry in ``games.py``.  Finished
sessions are recorded into ``MatchHistory`` by ``PlayManager``.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance

from ...agent import PERSONAS, Coach, DialogueEngine, Opponent, Skills, TeachContext
from ...difficulty.adaptive import AdaptiveController, pacing_scale
from ...online_learning.recorder import LearningHooks, TrajectoryRecorder
from ...profile.store import ProfileStore
from ...result import player_won
from ...solver_provider import SolverHandle, SolverProvider
from .custom_games import CustomGameRegistry
from .games import GAMES, GameSpec, PlayError
from .history import MatchHistory

#: 内置游戏 → 规则族（对局记录/复盘元数据；自定义游戏由注册表提供）。
#: 与 games.py 注册表保持全量覆盖：缺项会让 GameInfo.family / 快照 family 为
#: None，前端按“未知”兜底、后端按 grid 兜底（LLM action 形态、正则落子解析）。
_BUILTIN_FAMILY: dict[str, str] = {
    "moon_chess": "grid",
    "stochastic_gomoku": "grid",
    "texas_holdem": "poker",
    "mahjong_guangdong": "mahjong",
    "mahjong_hongzhong": "mahjong",
    "mahjong_blood": "mahjong",
    "mahjong_sichuan": "mahjong",
    "mahjong_changsha": "mahjong",
    "mahjong_taiwan": "mahjong",
    "mahjong_international": "mahjong",
    # UNO 六变体（P1-4/P1-6 接入）：前端 FAMILY_BOARDS["uno"] 分发。
    "uno": "uno",
    "uno_seven_zero": "uno",
    "uno_jump_in": "uno",
    "uno_stacking": "uno",
    "uno_draw_until": "uno",
    "uno_strict_wild4": "uno",
    # 谁是卧底（social 族）：前端 FAMILY_BOARDS["social"] → SocialChatTable。
    "undercover": "social",
    # 狼人杀（social 族，9 人固定）：前端同样走 SocialChatTable。
    "werewolf": "social",
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class GameSession:
    """One human-vs-AI game, driven by its ``GameSpec``."""

    game_id: str
    spec: GameSpec
    player_pid: str
    difficulty: str
    engine: GameEngine
    solver: SolverHandle
    state: dict = field(init=False)
    last_ai_info: dict = field(default_factory=dict)  # {'move'|'vanish'|'action': ...} per game
    log: list[dict] = field(default_factory=list)  # move entries for history/replay
    started_at: str = field(default_factory=_now_iso)
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: Online-learning capture hook (set by ``PlayManager`` when learning
    #: is enabled); records every human/AI decision at adapter level.
    recorder: TrajectoryRecorder | None = field(default=None, init=False)
    #: 陪伴 Agent（开局按性格装配；``None`` = 关闭表达，纯对局）。见 D 节接线。
    persona: str | None = field(default=None)
    hint_level: str = field(default="off")
    hinted: bool = field(default=False)
    ai_strength: int | None = field(default=None)
    agent: DialogueEngine | None = field(default=None)
    pending_chat: list[dict] = field(default_factory=list)  # 待投递的聊天增量
    custom: bool = field(default=False)  # 自定义游戏（platform/custom_games.py 注册表条目）
    family: str | None = field(default=None)  # 规则族（grid/poker/mahjong/social）
    #: 教学对局开关：教练 Agent 能看到玩家自己的牌并推理（见 agent/coach.py
    #: 的三条设计红线：教练看玩家投影、双脑分离、raw_solver 防录制污染）。
    teaching: bool = field(default=False)
    #: 自适应难度已生效：AI 强度由玩家近 ``window`` 局胜率自动升降。
    #: ``difficulty`` 仍记录显式档位（展示/历史用），本标记独立承载
    #: "这局强度是自适应算出来的"这一事实，供快照/活跃列表/对局记录回显。
    adaptive_active: bool = field(default=False)
    #: 教练通道用的**原始**（未被 RecordingHandle 包装的）求解器句柄。
    #: 在线学习开启时 ``solver`` 会被包成录制句柄——AI 自己的落子照常
    #: 采集；教练替玩家算的参考动作必须走本句柄，否则会以 "ai" actor
    #: 混进训练轨迹（那是给玩家座位的建议，不是 AI 决策）。
    raw_solver: SolverHandle | None = field(default=None, init=False)
    #: 教学讲评载荷：玩家行动前算好的教学上下文（含参考动作），由
    #: ``PlayManager._chat_after_move`` 消费成 ``teach_move`` 消息。
    pending_teach: TeachContext | None = field(default=None, init=False)
    #: 对手模式单步守门：记录本步已发 ``opp_react`` 的 log 长度，防
    #: ``run_ai`` 多动作回调重复队列（见 ``_record_ai_action``）。
    _opp_react_step: int = field(default=0, init=False)
    #: 本局实际使用的随机种子（PlayManager 按 base+开局序号派生——首局等于
    #: base seed 保持既有测试确定性，第二局起牌墙不同，「再来一局」不再同牌）。
    seed: int = 42

    def __post_init__(self) -> None:
        self.state = self.engine.create_initial_state()
        if self.raw_solver is None:
            self.raw_solver = self.solver

    @property
    def ai_pid(self) -> str:
        return self.spec.seat_options[1] if self.player_pid == self.spec.seat_options[0] else self.spec.seat_options[0]

    @property
    def over(self) -> bool:
        return self.engine.is_terminal(self.state)

    @property
    def winner(self) -> str | None:
        return self.state["env"].get("winner")

    @property
    def winners(self) -> list[str]:
        """所有胡/胜玩家（血战等多胡局承载；普通局与 ``winner`` 一致或为空）。"""
        value = self.state["env"].get("winners")
        return list(value) if isinstance(value, list) else []

    @property
    def current_player(self) -> str | None:
        return self.engine.get_current_player(self.state)

    @property
    def is_opponent_mode(self) -> bool:
        """二人非教练 = 对手模式（陪伴以「座内对手」身份说话）.

        判据：非教学、Agent 在场、座位数为 2（二人局的对手机理成立）。
        多人非教练（麻将 4 人等）在 P1 仍走啦啦队 fallback，P2 才换成
        桌友群聊（``agents: dict`` + 发言调度）。教学对局（``teaching``）
        走教练通道，零改动。
        """
        return not self.teaching and self.agent is not None and len(self.spec.seat_options) == 2

    @property
    def speaker(self) -> str:
        """pending_chat 条目的说话人标签（前端按此渲染头像/名字）.

        P1 单 Agent：对手模式/啦啦队都取 persona 显示名（如「轻松吐槽」）；
        教学模式取「教练」前缀 + persona 显示名。P2 多座位会扩展为
        按座位的 pid + 显示名（如「下家 p2」）。
        """
        if self.agent is None:
            return ""
        name = self.agent.persona.display_name
        return f"教练 · {name}" if self.teaching else name

    # ── Play ─────────────────────────────────────────────────────

    def step(self, payload: dict) -> None:
        """Validate and apply the human's action, then run the AI reply.

        教学对局：应用动作**之前**（状态还是玩家决策时刻的快照）由教练
        从玩家座位算一手参考动作——求解器按状态当前行动者推理，此刻它
        读的正是玩家的手牌；``Coach.review`` 把玩家实际动作并入对照。
        """
        if self.over:
            raise PlayError("本局已结束")
        action = self.spec.parse_human_action(self, payload)
        if self.recorder is not None:
            # ``self.state`` is still the pre-move snapshot here — engine
            # applies return new states, so the reference is safe.
            self.recorder.record_human(self, action)
        reference: TeachContext | None = None
        if self.teaching:
            reference = Coach.build(self.state, self.player_pid, self.engine, self.raw_solver)
        self.spec.apply_human(self, action)
        self.log.append(self._log_entry("human", action))
        self.spec.run_ai(self, self._record_ai_action)
        if reference is not None:
            self.pending_teach = Coach.review(reference, action, self.spec.describe_action)

    def _record_ai_action(self, action: ActionInstance) -> None:
        self.log.append(self._log_entry("ai", action))
        # 对手模式：AI 行动后队列一句对手反应（opp_react）。``run_ai`` 可能在
        # 一个回合内多次回调（多动作轮），用 ``_opp_react_step`` 守门确保单
        # 步只发一句（防多动作刷屏）；``PlayManager._chat_after_move`` 也会
        # 在步末兜底（若本步 AI 行动未被此处捕捉，例如 opening 的 AI 先手）。
        if self.is_opponent_mode and self._opp_react_step != len(self.log):
            self._opp_react_step = len(self.log)
            self._queue_opp("opp_react")

    def _queue_opp(self, scenario: str) -> None:
        """对手模式队列一条消息：用 AI 投影构建 :class:`OpponentContext` 成文.

        fail-soft：OpponentContext 构建或 Agent 成文任一异常都静默跳过
        （对手说话是表达层增值项，绝不阻断对局主流程）。
        """
        if self.agent is None:
            return
        try:
            ctx = Opponent.build(self.state, self.ai_pid, self.player_pid, self.engine, self.log)
            msg = self.agent.reply(ctx, scenario, game_id=self.game_id)
        except Exception:  # noqa: BLE001 — 对手通道 fail-soft
            return
        self.pending_chat.append(
            {
                "scenario": scenario,
                "text": msg.text,
                "mood": msg.mood,
                "step": len(self.log),
                "reasoning": msg.reasoning,
                "speaker": self.speaker,
            }
        )

    def _log_entry(self, actor: str, action: ActionInstance) -> dict:
        return {
            "step": len(self.log),
            "actor": actor,
            "action": self.spec.describe_action(action),
            "snapshot": self.spec.build_snapshot(self),
        }

    def snapshot(self) -> dict:
        """Public view of the session for the API.

        Extends the per-game snapshot with a ``family`` marker (the rule
        family, self-describing for frontend dispatch — the frontend must
        not depend on the games catalog being loaded), a ``chat`` list
        (incremental agent messages pending delivery) and an
        ``evaluation`` dict (mechanical position eval; ``None`` when the
        agent is disabled).
        """
        snap = self.spec.build_snapshot(self)
        snap["family"] = self.family
        snap["teaching"] = self.teaching
        # 自适应状态回显：adaptive 标记本局强度是否由胜率自适应算出，
        # ai_strength 是本局实际搜索预算（自适应时随近 10 局胜率浮动）。
        snap["adaptive"] = self.adaptive_active
        snap["ai_strength"] = self.ai_strength
        snap["chat"] = self.drain_chat()
        snap["evaluation"] = self._evaluate_position()
        return snap

    def drain_chat(self) -> list[dict]:
        """Take (and clear) the chat increments accumulated since the last call.

        Each entry: ``{"scenario", "text", "mood", "step"}``.
        """
        pending, self.pending_chat = self.pending_chat, []
        return pending

    def _evaluate_position(self) -> dict | None:
        """Mechanical position evaluation (``None`` when the agent is off)."""
        if self.agent is None:
            return None
        ctx = Skills.build(self.state, self.player_pid, self.engine)
        return Skills.evaluate_position(ctx, self.engine)


class PlayManager:
    """Registry of active sessions (in-memory, single-process).

    Thread-safe: registry mutations and history writes are guarded by
    one lock; each session additionally owns a per-session lock so two
    concurrent ``move`` calls cannot interleave on the same game.
    """

    def __init__(
        self,
        provider: SolverProvider,
        history: MatchHistory | None = None,
        seed: int = 42,
        max_sessions: int = 128,
        learning: LearningHooks | None = None,
        *,
        profiles: ProfileStore | None = None,
        adaptive: AdaptiveController | None = None,
        agent_factory: Callable[[str], DialogueEngine | None] | None = None,
        custom: CustomGameRegistry | None = None,
    ) -> None:
        self._provider = provider
        self._history = history
        self._seed = seed
        self._max_sessions = max_sessions
        self._learning = learning
        self._profiles = profiles
        self._adaptive = adaptive
        self._agent_factory = agent_factory
        self._custom = custom
        self._sessions: dict[str, GameSession] = {}
        self._lock = threading.Lock()
        # 开局序号：每局 seed = base + 序号（首局 = base，与旧行为一致）。
        self._start_count = 0

    @property
    def provider(self) -> SolverProvider:
        """The SolverProvider used to build sessions (hint/agent routes)."""
        return self._provider

    def start(
        self,
        game_id: str,
        player_pid: str,
        difficulty: str,
        player_count: int | None = None,  # None → spec.player_counts[0]（麻将 4 人）
        *,
        persona: str | None = None,
        hint_level: str = "off",
        pacing: str = "standard",
        adaptive_enabled: bool = False,
        teaching: bool = False,
        variant: str | None = None,
    ) -> GameSession:
        """Create a new session; resolves start chance nodes and lets the AI open.

        Companion wiring (D 节接线): the persona is resolved from the
        explicit argument or the profile default (``gentle`` fallback);
        the AI search budget comes from the explicit tier, or is computed
        adaptively from the player's recent win rate, then scaled by the
        pacing preset.  A ``greet`` chat increment is queued on the
        session for the start response.

        教学对局（``teaching=True``）：未显式指定性格且档案无默认时，
        人格兜底从 ``gentle`` 换成 ``teacher``；开局队列 ``teach_greet``
        （教练开场）+ ``teach_turn``（轮到玩家时的读牌导读）。
        """
        spec = GAMES.get(game_id)
        is_custom = False
        family: str | None = None
        if spec is None and self._custom is not None:
            custom_spec = self._custom.spec_for(game_id)
            if custom_spec is not None:
                spec = custom_spec
                is_custom = True
                family = self._custom.family_of(game_id)
        if spec is None:
            raise PlayError(f"未知游戏: {game_id}")
        if family is None:
            family = _BUILTIN_FAMILY.get(game_id)
        if player_count is None:
            # 未显式指定时按注册表默认人数开局（麻将 = 4 人）。
            player_count = spec.player_counts[0]
        if player_count not in spec.player_counts:
            raise PlayError(f"该游戏不支持 {player_count} 人")
        if player_pid == "random":
            player_pid = spec.seat_options[0] if uuid.uuid4().int % 2 == 0 else spec.seat_options[1]
        if player_pid not in spec.seat_options:
            raise PlayError(f"未知{spec.seat_label}: {player_pid}")
        if difficulty != "adaptive" and difficulty not in spec.difficulty_budgets:
            raise PlayError(f"未知难度: {difficulty}")

        persona_key = persona
        if persona_key is None and self._profiles is not None:
            persona_key = str(self._profiles.load().get("default_persona", "") or "")
        if not persona_key:
            # 教学对局的兜底人格是「认真教学」（显式选择 / 档案默认不受影响）。
            persona_key = "teacher" if teaching else "gentle"
        if persona_key not in PERSONAS:
            raise PlayError(f"未知性格: {persona_key}")

        budget = self._pick_budget(spec, difficulty, pacing, adaptive_enabled)

        session_id = uuid.uuid4().hex[:8]
        # 每局派生新种子（base + 开局序号）：首局等于 base seed（既有测试
        # 的确定性不变），第二局起发牌不同——修复「再来一局永远同牌」。
        seed = self._seed + self._start_count
        self._start_count += 1
        if spec.player_counts != (2,):
            engine = spec.create_engine(seed, player_count=player_count, variant=variant, difficulty=difficulty)
        else:
            engine = spec.create_engine(seed, variant=variant, difficulty=difficulty)
        solver = spec.create_solver(self._provider, engine, seed, budget, difficulty=difficulty, pacing=pacing)
        session = GameSession(
            game_id=session_id,
            spec=spec,
            player_pid=player_pid,
            difficulty=difficulty,
            engine=engine,
            solver=solver,
            persona=persona_key,
            hint_level=hint_level,
            ai_strength=budget,
            agent=self._agent_factory(persona_key) if self._agent_factory is not None else None,
            custom=is_custom,
            family=family,
            teaching=teaching,
            # 与 _pick_budget 的自适应判定保持一致：显式 "adaptive" 档位或
            # payload 的自适应开关二选一都算自适应局。
            adaptive_active=difficulty == "adaptive" or adaptive_enabled,
            seed=seed,
        )
        if self._learning is not None and self._learning.enabled(spec.game_id):
            # Wrap the solver in a recording handle and attach a
            # per-match recorder; every AI decision (incl. multi-action
            # loops) is captured with zero GameSpec changes.
            # 门控（隐私红线）：enabled=False 的新会话**不再采集**——开关
            # 同时控制捕获与 apply，而不是只停发布。
            # 注意：``session.raw_solver`` 仍指未包装的原始句柄——教练
            # 通道的参考动作不被采集（防训练数据污染）。
            session.solver = self._learning.wrap_handle(session, session.solver)
        spec.resolve_start(session)
        if spec.ai_opens(session):
            spec.run_ai(session, session._record_ai_action)
        if teaching:
            self._say(session, "teach_greet")
            if not session.over and session.current_player == session.player_pid:
                self._say_teach_turn(session)
        else:
            self._say(session, "greet")
        with self._lock:
            self._evict_locked()
            self._sessions[session_id] = session
        return session

    def get(self, game_id: str) -> GameSession:
        with self._lock:
            session = self._sessions.get(game_id)
        if session is None:
            raise PlayError(f"未知对局: {game_id}")
        return session

    def move(self, game_id: str, payload: dict) -> dict:
        """Apply a human move, run the AI reply, and return the snapshot.

        Companion hook (D 节接线): after every move the nine scenarios
        are assessed and chat increments are queued on the session; an
        illegal attempt queues an ``illegal`` message before the error is
        re-raised.  When the game ends, the session is recorded into
        history, the profile tally is updated, and the session is removed
        from the registry.
        """
        session = self.get(game_id)
        with session.lock:
            try:
                session.step(payload)
            except PlayError:
                self._say(session, "illegal")
                raise
            self._chat_after_move(session)
            snapshot = session.snapshot()
            if session.over:
                if self._history is not None:
                    self._history.record(self._build_record(session))
                if self._learning is not None:
                    self._learning.on_finished(session)
                self._update_profile(session)
                self.remove(game_id)
        return snapshot

    def remove(self, game_id: str) -> None:
        with self._lock:
            self._sessions.pop(game_id, None)

    def active_sessions(self) -> list[dict]:
        """Lightweight listing of in-flight sessions (oldest first).

        Backs the ``GET /api/match/active`` route so the frontend can
        offer a real \"continue last game\" (resume via ``?game=<id>``
        + ``/api/match/state``) instead of restarting blindly.  Entries
        never expose hidden arrays: only public meta fields are returned.
        """
        with self._lock:
            sessions = list(self._sessions.values())
        return [
            {
                "game_id": s.game_id,
                "game": s.spec.game_id,
                "display_name": s.spec.display_name,
                "player_pid": s.player_pid,
                "difficulty": s.difficulty,
                "persona": s.persona,
                "hint_level": s.hint_level,
                "teaching": s.teaching,
                "adaptive": s.adaptive_active,
                "ai_strength": s.ai_strength,
                "step": len(s.log),
                "started_at": s.started_at,
            }
            for s in sessions
        ]

    # ── Companion agent ──────────────────────────────────────────

    def say(self, game_id: str, scenario: str) -> dict | None:
        """One agent message for an active session (``None`` when agent off).

        教学对局：上下文换成 :class:`TeachContext`（玩家自己的投影），
        教练在自由对话里也能围绕玩家的牌回答（"我听什么？"）。
        对手模式（二人非教练）：上下文换成 :class:`OpponentContext`（AI
        自己的投影），对手以「座内对手」身份应答（adversarial scan 放行
        AI 自己的牌、拦玩家的隐藏牌）。
        """
        session = self.get(game_id)
        if session.agent is None:
            return None
        ctx = self._speak_ctx(session)
        msg = session.agent.reply(ctx, scenario, game_id=session.game_id)
        return {
            "scenario": scenario,
            "text": msg.text,
            "mood": msg.mood,
            "reasoning": msg.reasoning,
            "speaker": session.speaker,
        }

    def hint(self, game_id: str, level: str) -> dict:
        """Mechanical hint for an active session (direction/specific/demo).

        教学对局：``specific`` / ``demo`` 级别的"具体建议/演示"升级为
        教练参考动作（raw_solver 在玩家座位算的真实走法）——这本来就是
        教学局的承诺（AI 看着玩家的牌推理），无需再走占位的确定性中位
        选动作。
        """
        session = self.get(game_id)
        ctx = Skills.build(session.state, session.player_pid, session.engine)
        session.hinted = True
        if session.teaching and level != "direction":
            return self._teach_hint(session, level, ctx)
        return Skills.suggest_hint(ctx, level, self._provider, session.engine)

    def _teach_hint(self, session: GameSession, level: str, ctx: object) -> dict:
        """Teaching-mode hint: the coach's reference action for the player."""
        result: dict = {
            "level": level,
            "direction": Skills.evaluate_position(ctx, session.engine).get("summary", ""),
            "mechanical_text": Skills.evaluate_position(ctx, session.engine).get("mechanical_text", ""),
        }
        reference = Coach.reference_action(session.state, session.player_pid, session.engine, session.raw_solver)
        if reference is not None:
            result["action"] = reference.canonical_key
            label = "教练演示" if level == "demo" else "教练建议"
            result["hint"] = f"{label}：{session.spec.describe_action(reference)}"
        else:
            result["hint"] = Skills.suggest_hint(ctx, "direction", None, session.engine)["direction"]
        return result

    # ── Internals ────────────────────────────────────────────────

    def _say(self, session: GameSession, scenario: str) -> None:
        """Queue one agent message on the session (no-op when agent off).

        上下文按陪伴身份分派（``_speak_ctx``）：教学→玩家投影、对手→AI
        投影、默认啦啦队→玩家投影。``pending_chat`` 条目带 ``speaker``。
        """
        if session.agent is None:
            return
        ctx = self._speak_ctx(session)
        msg = session.agent.reply(ctx, scenario, game_id=session.game_id)
        session.pending_chat.append(
            {
                "scenario": scenario,
                "text": msg.text,
                "mood": msg.mood,
                "step": len(session.log),
                "reasoning": msg.reasoning,
                "speaker": session.speaker,
            }
        )

    def _speak_ctx(self, session: GameSession) -> object:
        """按陪伴身份构建说话上下文（``say`` / ``_say`` / ``_chat_after_move`` 共用）.

        - 教学对局（``teaching``）：玩家自己的投影（教练看玩家的牌）。
        - 对手模式（二人非教练）：AI 自己的投影（对手看自己的牌 + 玩家
          公开动作序列），驱动 adversarial scan。
        - 默认啦啦队（多人非教练，P2 前的 fallback）：玩家投影。
        """
        if session.teaching:
            return Coach.build(session.state, session.player_pid, session.engine, None)
        if session.is_opponent_mode:
            return Opponent.build(session.state, session.ai_pid, session.player_pid, session.engine, session.log)
        return Skills.build(session.state, session.player_pid, session.engine)

    def _chat_after_move(self, session: GameSession) -> None:
        """场景检测与消息队列（D 节接线 + 对手模式双触发）.

        - 教学对局：玩家的 blunder/good_move 泛化点评升级为 ``teach_move``
          讲评（对照教练在玩家座位算的参考动作），讲评后若又轮到玩家则
          追加 ``teach_turn`` 读牌导读。
        - 对手模式（二人非教练）：玩家行动后队列 ``opp_read``（读人，不
          依赖评分命中）；AI 行动后的 ``opp_react`` 已在
          ``_record_ai_action`` 里队列（步末此处不重复）。终局走
          ``ai_win`` / ``ai_lose`` / ``game_over``（对手视角，showdown 后
          ``revealed=True`` 全放行可复盘）。
        - 默认啦啦队（多人非教练，P2 前 fallback）：保留既有
          blunder/good_move 评分触发。
        """
        if session.agent is None:
            return
        # 教学讲评优先于胜负播报：最后一手的对照点评也值得讲（之后照常
        # 播报 ai_win/ai_lose/game_over）。
        if session.teaching and session.pending_teach is not None:
            self._say_ctx(session, session.pending_teach, "teach_move")
            session.pending_teach = None
        if session.over:
            # 胜负播报按玩家视角解析（阵营胜者经 final_roles 解析——社交游戏
            # 的 winner 是阵营名，直接与 player_pid 比较会把卧底胜误判成 AI 胜；
            # 见 layer4_interface/result.py）。平局（win is None）只播 game_over。
            snap = None
            try:
                snap = session.spec.build_snapshot(session)
            except Exception:
                snap = None
            won = player_won(session.winner, session.player_pid, session.winners, snap)
            if won is True:
                self._say(session, "ai_lose")
            elif won is False:
                self._say(session, "ai_win")
            self._say(session, "game_over")
            return
        if session.teaching:
            if session.current_player == session.player_pid:
                self._say_teach_turn(session)
            return
        # 对手模式：玩家行动后读人（opp_read），不依赖评分命中。AI 行动后
        # 的 opp_react 已在 ``_record_ai_action`` 队列（此处只补 opp_read，
        # 保持「双触发」的玩家侧半边）。去重窗口（5 分钟）+ 人设分寸节制频率。
        if session.is_opponent_mode:
            ctx = Opponent.build(session.state, session.ai_pid, session.player_pid, session.engine, session.log)
            self._say_ctx(session, ctx, "opp_read")
            return
        # 默认啦啦队（多人非教练 fallback）：保留评分触发的 blunder/good_move。
        last = session.log[-1] if session.log else None
        if last is None or last.get("actor") != "human":
            return
        ctx = Skills.build(session.state, session.player_pid, session.engine)
        if Skills.detect_blunder(ctx, session.engine) is not None:
            self._say_ctx(session, ctx, "blunder")
        elif Skills.detect_good_move(ctx, session.engine) is not None:
            self._say_ctx(session, ctx, "good_move")

    def _say_teach_turn(self, session: GameSession) -> None:
        """Queue the ``teach_turn`` reading (player's own hand, no spoilers).

        导读不带参考动作（不剧透答案——玩家还没走）；参考动作只在
        ``teach_move`` 讲评时给出。教练关闭（agent=None）时跳过。
        """
        if session.agent is None:
            return
        ctx = Coach.build(session.state, session.player_pid, session.engine, None)
        self._say_ctx(session, ctx, "teach_turn")

    def _say_ctx(self, session: GameSession, ctx: object, scenario: str) -> None:
        """Queue a message from a prebuilt context (avoid a second build).

        ``pending_chat`` 条目带 ``speaker``（与 ``_say`` / ``_queue_opp``
        对齐，前端按 speaker 渲染头像/名字）。
        """
        msg = session.agent.reply(ctx, scenario, game_id=session.game_id)  # type: ignore[arg-type] — ctx is SkillContext
        session.pending_chat.append(
            {
                "scenario": scenario,
                "text": msg.text,
                "mood": msg.mood,
                "step": len(session.log),
                "reasoning": msg.reasoning,
                "speaker": session.speaker,
            }
        )

    def _pick_budget(self, spec: GameSpec, difficulty: str, pacing: str, adaptive_enabled: bool) -> int:
        """Resolve the AI search budget (explicit tier / adaptive + pacing)."""
        budgets = spec.difficulty_budgets
        if difficulty == "adaptive" or adaptive_enabled:
            if self._adaptive is None:
                base = budgets["normal"]
            else:
                base = self._adaptive.pick_budget(spec.game_id, "adaptive", self._recent_matches(spec))
        else:
            base = budgets[difficulty]
        return max(1, int(base * pacing_scale(pacing)))

    def default_persona(self) -> str:
        """Resolve the global default persona key (profile → gentle fallback).

        封装 ``start()`` 里曾内联的解析逻辑，供平台聊天层（``chat``）在
        没有进行中对局时取全局人设——让平台助手与对局陪玩共用同一份
        persona（方向 C 人设统一）。
        """
        if self._profiles is not None:
            key = str(self._profiles.load().get("default_persona", "") or "")
            if key and key in PERSONAS:
                return key
        return "gentle"

    def _recent_matches(self, spec: GameSpec) -> list[dict]:
        """Win-rate window derived from the profile tally (oldest-first).

        The profile schema stores ``{wins, plays}`` tallies per game (C3
        contract); AdaptiveController only needs the window win rate, so
        the tally is expanded into a synthetic oldest-first list.
        """
        if self._profiles is None:
            return []
        profile = self._profiles.load()
        recent = profile.get("recent", {})
        tally = recent.get(spec.game_id, {}) if isinstance(recent, dict) else {}
        plays = int(tally.get("plays", 0))
        wins = int(tally.get("wins", 0))
        return [
            {"winner": "player" if i < wins else "ai", "player_pid": "player", "difficulty": "normal"}
            for i in range(plays)
        ]

    def _update_profile(self, session: GameSession) -> None:
        """Tally the finished match into the profile's per-game record."""
        if self._profiles is None:
            return
        profile = self._profiles.load()
        recent = profile.get("recent", {})
        if not isinstance(recent, dict):
            recent = {}
            profile["recent"] = recent
        tally = recent.setdefault(session.spec.game_id, {"wins": 0, "plays": 0})
        tally["plays"] = int(tally.get("plays", 0)) + 1
        # 胜局统计按玩家视角解析（多胡局看 winners；社交阵营胜者经
        # final_roles 解析——否则卧底获胜不计入胜场）。
        snap = None
        try:
            snap = session.spec.build_snapshot(session)
        except Exception:
            snap = None
        won = player_won(session.winner, session.player_pid, session.winners, snap)
        if won is True:
            tally["wins"] = int(tally.get("wins", 0)) + 1
        self._profiles.save(profile)

    def _build_record(self, session: GameSession) -> dict:
        """Assemble the persisted match record (see history module)."""
        # 阵营胜者解析为玩家视角的 won（None=平局），写入记录供列表/复盘/战绩
        # 复用——否则 social 阵营胜者（winner=undercover）在 history/review 页
        # 会被误标「失败」（实测 e7deb84b）。
        snap = None
        try:
            snap = session.spec.build_snapshot(session)
        except Exception:
            snap = None
        won = player_won(session.winner, session.player_pid, session.winners, snap)
        return {
            "match_id": session.game_id,
            "game_id": session.spec.game_id,
            "player_pid": session.player_pid,
            "ai_pid": session.ai_pid,
            "difficulty": session.difficulty,
            "seed": session.seed,
            "started_at": session.started_at,
            "finished_at": _now_iso(),
            "winner": session.winner,
            "winners": session.winners,
            "won": won,
            "over": True,
            "persona": session.persona,
            "hinted": session.hinted,
            "ai_strength": session.ai_strength,
            "adaptive": session.adaptive_active,
            "family": session.family,
            "custom": session.custom,
            "teaching": session.teaching,
            "moves": session.log,
        }

    def _evict_locked(self) -> None:
        """Bound the registry by dropping the oldest unfinished session (FIFO)."""
        while len(self._sessions) >= self._max_sessions:
            _, oldest = next(iter(self._sessions.items()))
            self._sessions.pop(oldest.game_id, None)
