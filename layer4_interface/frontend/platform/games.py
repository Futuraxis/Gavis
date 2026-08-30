"""Unified game registry for the platform frontend.

Each supported game is described by a frozen ``GameSpec`` dataclass that
wires engine creation, solver creation, human-action parsing and
snapshot serialization.  The three play apps under
``layer4_interface/frontend/play_*`` were the templates for these
specs; the platform owns one generic session implementation
(``session.py``) driven by them.

Engine creation is generic (v5.2): every spec builds a bare
``GameEngine`` from the game's rules JSON — variants / player counts are
declared inside the JSON and selected via constructor args, so no
per-game adapter class is required.  Per-game *display* helpers that are
not pure rule evaluation (pot, hand name, seat constants) live in this
frontend module, which is where game-specific presentation is allowed.

``PlayError`` lives here (not in ``session.py``) because the spec
closures raise it at runtime, and ``session.py`` already imports
``GAMES`` from this module.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

from layer2_engine.core.engine import GameEngine
from layer2_engine.core.state_graph import ActionInstance

from ...solver_provider import SolverHandle, SolverProvider
from ..engine_helpers import engine_from_rules, mahjong_tile_name, resolve_all_chance, texas_hand_name

if TYPE_CHECKING:
    from .session import GameSession

Difficulty = Literal["easy", "normal", "hard"]


class PlayError(Exception):
    """Bad play request (unknown game, illegal move, game over, ...)."""


@dataclass(frozen=True)
class GameSpec:
    """Everything the platform needs to run one game's human-vs-AI sessions."""

    game_id: str
    display_name: str
    description: str
    kind: Literal["board", "poker", "mahjong", "uno"]
    board_size: int | None
    seat_options: tuple[str, ...]
    seat_label: str  # '颜色' for board games, '座位' for poker
    difficulty_budgets: dict[Difficulty, int]
    create_engine: Callable[..., GameEngine]  # (seed, player_count=<spec default>)
    create_solver: Callable[[SolverProvider, GameEngine, int, int], SolverHandle]  # (provider, engine, seed, budget)
    resolve_start: Callable[[GameSession], None]  # start chance nodes (dealing)
    ai_opens: Callable[[GameSession], bool]  # AI moves before the human's first move
    parse_human_action: Callable[[GameSession, dict], ActionInstance]
    apply_human: Callable[[GameSession, ActionInstance], None]  # apply + chance resolution
    run_ai: Callable[[GameSession, Callable[[ActionInstance], None] | None], None]
    build_snapshot: Callable[[GameSession], dict]
    describe_action: Callable[[ActionInstance], str]  # history log caption
    player_counts: tuple[int, ...] = (2,)  # seat count options (mahjong specs override with (4,))


# ── Moon Chess ────────────────────────────────────────────────────


def _moon_create_engine(seed: int) -> GameEngine:
    return engine_from_rules("moon_chess", seed)


def _moon_create_solver(provider: SolverProvider, engine: GameEngine, seed: int, budget: int) -> SolverHandle:
    return provider.create_solver("moon_chess", "mcts", engine, seed, budget)


def _moon_cell_index(action: ActionInstance) -> int:
    cell = action.params.get("cell", {})
    cell_id = cell.get("id", "") if isinstance(cell, dict) else str(cell)
    _, r, c = cell_id.split("_")
    return int(r) * 3 + int(c)


def _moon_parse_human_action(session: GameSession, payload: dict) -> ActionInstance:
    if session.over:
        raise PlayError("本局已结束")
    if session.current_player != session.player_pid:
        raise PlayError("还没轮到你")
    try:
        cell_index = int(payload.get("cell_index", -1))
    except (TypeError, ValueError):
        raise PlayError("需要 cell_index") from None
    for action in session.engine.get_legal_actions(session.state):
        if _moon_cell_index(action) == cell_index:
            return action
    raise PlayError(f"非法落子: {cell_index}")


def _moon_apply_human(session: GameSession, action: ActionInstance) -> None:
    session.state = session.engine.apply_action(session.state, action)


def _moon_run_ai(session: GameSession, on_ai_action: Callable[[ActionInstance], None] | None = None) -> None:
    if session.over or session.current_player != session.ai_pid:
        return
    action = session.solver.select_action(session.state)
    if action is None:
        return
    session.state = session.engine.apply_action(session.state, action)
    session.last_ai_info["move"] = _moon_cell_index(action)
    if on_ai_action is not None:
        on_ai_action(action)


def _moon_snapshot(session: GameSession) -> dict:
    """3×3 board; ``round_age`` maps cell index → piece age (1 = newest)."""
    age_map: dict[str, int] = {}
    for entry in session.state["_arrays"].get("pieceOrder", []):
        cid = entry.get("cell_id", "")
        if cid:
            _, r, c = cid.split("_")
            age_map[int(r) * 3 + int(c)] = len(age_map) + 1
    return {
        "game_id": session.game_id,
        "player_pid": session.player_pid,
        "difficulty": session.difficulty,
        "board": session.state["_arrays"]["board"],
        "round_age": age_map,
        "turn": session.current_player,
        "winner": session.winner,
        "over": session.over,
        "last_ai_move": session.last_ai_info.get("move"),
        "round": session.state["env"].get("round", 0),
    }


def _moon_describe_action(action: ActionInstance) -> str:
    cell = action.params.get("cell", {})
    return cell.get("id", "") if isinstance(cell, dict) else str(cell)


# ── Stochastic Gomoku ─────────────────────────────────────────────


def _gomoku_create_engine(seed: int) -> GameEngine:
    return engine_from_rules("stochastic_gomoku", seed)


def _gomoku_create_solver(provider: SolverProvider, engine: GameEngine, seed: int, budget: int) -> SolverHandle:
    return provider.create_solver("stochastic_gomoku", "mcts", engine, seed, budget)


def _gomoku_cell_index(action: ActionInstance) -> int:
    cell = action.params.get("cell", {})
    return int(cell.get("_index", -1)) if isinstance(cell, dict) else -1


def _gomoku_find_vanished(session: GameSession, state: dict) -> tuple[int | None, str | None]:
    """Locate the cell that vanished; returns (cell, vanished piece color)."""
    board = state["_arrays"]["board"]
    last_actor = state["env"].get("lastActor")
    last_cell = state["env"].get("lastPlacedCell")
    if last_actor is None or last_cell is None:
        return None, None
    try:
        _, r, c = str(last_cell).split("_")
        idx = int(r) * session.spec.board_size + int(c)
    except ValueError:
        return None, None
    size = len(board)
    vanished = idx if 0 <= idx < size and board[idx] is None else None
    return (vanished, last_actor) if vanished is not None else (None, None)


def _gomoku_resolve_chance(session: GameSession, state: dict) -> tuple[dict, int | None, str | None]:
    """Resolve all pending chance nodes; returns (state, vanished_cell, vanished_color)."""
    vanished: int | None = None
    color: str | None = None
    while session.engine.get_node_type(state) == "chance":
        outcome, state = session.engine.sample_chance(state)
        if outcome.key == "vanish":
            vanished, color = _gomoku_find_vanished(session, state)
    return state, vanished, color


def _gomoku_parse_human_action(session: GameSession, payload: dict) -> ActionInstance:
    if session.over:
        raise PlayError("本局已结束")
    if session.current_player != session.player_pid:
        raise PlayError("还没轮到你")
    try:
        cell_index = int(payload.get("cell_index", -1))
    except (TypeError, ValueError):
        raise PlayError("需要 cell_index") from None
    for action in session.engine.get_legal_actions(session.state):
        if _gomoku_cell_index(action) == cell_index:
            return action
    raise PlayError(f"非法落子: {cell_index}")


def _gomoku_apply_human(session: GameSession, action: ActionInstance) -> None:
    session.state = session.engine.apply_action(session.state, action)
    state, vanished, color = _gomoku_resolve_chance(session, session.state)
    session.state = state
    session.last_ai_info["vanish"] = vanished
    session.last_ai_info["vanish_color"] = color


def _gomoku_run_ai(session: GameSession, on_ai_action: Callable[[ActionInstance], None] | None = None) -> None:
    if session.over or session.current_player != session.ai_pid:
        return
    action = session.solver.select_action(session.state)
    if action is None:
        return
    session.state = session.engine.apply_action(session.state, action)
    session.last_ai_info["move"] = _gomoku_cell_index(action)
    state, vanished, color = _gomoku_resolve_chance(session, session.state)
    session.state = state
    session.last_ai_info["vanish"] = vanished
    session.last_ai_info["vanish_color"] = color
    if on_ai_action is not None:
        on_ai_action(action)


def _gomoku_snapshot(session: GameSession) -> dict:
    return {
        "game_id": session.game_id,
        "player_pid": session.player_pid,
        "difficulty": session.difficulty,
        "board": session.state["_arrays"]["board"],
        "turn": session.current_player,
        "winner": session.winner,
        "over": session.over,
        "last_ai_move": session.last_ai_info.get("move"),
        "last_vanish": session.last_ai_info.get("vanish"),
        "last_vanish_color": session.last_ai_info.get("vanish_color"),
        "round": session.state["env"].get("round", 0),
    }


def _gomoku_describe_action(action: ActionInstance) -> str:
    idx = _gomoku_cell_index(action)
    return f"cell_{idx // 9}_{idx % 9}" if idx >= 0 else "?"


# ── Texas Hold'em ─────────────────────────────────────────────────

# Display helpers live here (frontend domain): the *rules* stay the
# single source of truth — ``best5`` is a rules-declared alias evaluated
# through the engine's generic ``eval_expr`` (v5.2 engine service).

# (hand-name display helper lives in ``..engine_helpers``)


def _poker_create_engine(seed: int) -> GameEngine:
    return engine_from_rules("texas_holdem", seed)


def _poker_create_solver(provider: SolverProvider, engine: GameEngine, seed: int, budget: int) -> SolverHandle:
    return provider.create_solver("texas_holdem", "hybrid", engine, seed, budget)


def _poker_resolve_start(session: GameSession) -> None:
    """Deal blinds and hole cards (chance nodes) before play starts."""
    session.state = resolve_all_chance(session.engine, session.state)


def _poker_parse_human_action(session: GameSession, payload: dict) -> ActionInstance:
    if session.over:
        raise PlayError("本局已结束")
    if session.current_player != session.player_pid:
        raise PlayError("还没轮到你")
    choice = payload.get("choice")
    amount = payload.get("amount")
    for action in session.engine.get_legal_actions(session.state):
        if action.params.get("choice") != choice:
            continue
        if amount is None or action.params.get("amount") == amount:
            return action
    raise PlayError(f"非法动作: {choice} {amount}")


def _poker_apply_human(session: GameSession, action: ActionInstance) -> None:
    session.state = session.engine.apply_action(session.state, action)
    session.state = resolve_all_chance(session.engine, session.state)


def _poker_run_ai(session: GameSession, on_ai_action: Callable[[ActionInstance], None] | None = None) -> None:
    """Let the AI act while it is its turn (multi-action rounds)."""
    while not session.over and session.current_player == session.ai_pid:
        action = session.solver.select_action(session.state)
        if action is None:  # search found nothing — random fallback
            legal = session.engine.get_legal_actions(session.state)
            action = random.choice(legal) if legal else None
        if action is None:
            break
        session.state = session.engine.apply_action(session.state, action)
        session.state = resolve_all_chance(session.engine, session.state)
        session.last_ai_info["action"] = action.canonical_key
        if on_ai_action is not None:
            on_ai_action(action)


def _poker_snapshot(session: GameSession) -> dict:
    env = session.state["env"]
    arrs = session.state["_arrays"]
    over = session.over
    revealed = over and env.get("last_action") == "showdown"
    legal = []
    if not over and session.current_player == session.player_pid:
        for action in session.engine.get_legal_actions(session.state):
            legal.append({"choice": action.params["choice"], "amount": action.params["amount"]})
    raise_amts = sorted({a["amount"] for a in legal if a["choice"] == "raise"})

    def _cards(pid: str) -> list:
        return list(arrs.get(f"{pid[2:]}_hole", []))

    def _hand_name(pid: str, *, gate: bool) -> str | None:
        # P1-2 修复：牌型类别必须按显式门计算，不能统一用 ``over``。
        # 人类自己的牌（my_hand_name）以 ``over`` 为门——玩家始终知道自己的底牌，
        # 终局展示牌型无泄露；AI 的牌型（ai_hand_name）必须以 ``revealed``
        # （= over 且 last_action=="showdown"）为门，否则弃牌局（over=True、
        # revealed=False）会从 AI 隐藏底牌+公共牌反推出 AI 牌型类别（口袋对/成牌），
        # 击穿 §A.3 reveal-gate 红线。
        if not gate:
            return None
        cards = [*_cards(pid), *arrs.get("community", [])]
        return texas_hand_name(session.engine, cards)

    return {
        "game_id": session.game_id,
        "player_pid": session.player_pid,
        "ai_pid": session.ai_pid,
        "difficulty": session.difficulty,
        "over": over,
        "winner": session.winner,
        "turn": session.current_player,
        "phase": env.get("phase"),
        "street": env.get("street", 0),
        "street_name": ("翻前", "翻牌", "转牌", "河牌")[env.get("street", 0)],
        "pot": int(env.get("sb_committed", 0) + env.get("bb_committed", 0)),
        "community": list(arrs.get("community", [])),
        "my_hole": _cards(session.player_pid),
        "ai_hole": _cards(session.ai_pid) if revealed else [],
        "revealed": revealed,
        "my_stack": int(env.get(f"{session.player_pid[2:]}_stack", 0)),
        "ai_stack": int(env.get(f"{session.ai_pid[2:]}_stack", 0)),
        "my_committed": int(env.get(f"{session.player_pid[2:]}_committed", 0)),
        "ai_committed": int(env.get(f"{session.ai_pid[2:]}_committed", 0)),
        "my_folded": bool(env.get(f"{session.player_pid[2:]}_folded")),
        "ai_folded": bool(env.get(f"{session.ai_pid[2:]}_folded")),
        "last_actor": env.get("last_actor"),
        "last_action": env.get("last_action"),
        "last_ai_action": session.last_ai_info.get("action"),
        "call_to": int(env.get("last_call_to", 0)),
        "my_hand_name": _hand_name(session.player_pid, gate=over),
        "ai_hand_name": _hand_name(session.ai_pid, gate=revealed),
        "payoff": session.engine.get_utility(session.state, session.player_pid) if over else None,
        "legal": legal,
        "raise_amounts": raise_amts,
    }


def _poker_describe_action(action: ActionInstance) -> str:
    return action.canonical_key


# ── Mahjong (guangdong / hongzhong / blood) ───────────────────────


def _make_mahjong_engine(variant: str) -> Callable[..., GameEngine]:
    def _create(seed: int, player_count: int = 4) -> GameEngine:
        # Variants and player counts are declared in the JSON's
        # ``variants`` section (v5.2); the engine selects them as pure data.
        # 麻将默认 4 人（rules/mahjong.json 的 variants.player_count 亦为 4）。
        return engine_from_rules("mahjong", seed, variant=variant, player_count=player_count)

    return _create


def _make_mahjong_solver(game_id: str) -> Callable[[SolverProvider, GameEngine, int, int], SolverHandle]:
    def _create(provider: SolverProvider, engine: GameEngine, seed: int, budget: int) -> SolverHandle:
        return provider.create_solver(game_id, "mahjong", engine, seed, budget)

    return _create


def _mahjong_resolve_start(session: GameSession) -> None:
    while session.engine.get_node_type(session.state) == "chance":
        _, session.state = session.engine.sample_chance(session.state)


def _mahjong_parse_human_action(session: GameSession, payload: dict) -> ActionInstance:
    if session.over:
        raise PlayError("本局已结束")
    if session.current_player != session.player_pid:
        raise PlayError("还没轮到你")
    action_type = payload.get("type")
    for action in session.engine.get_legal_actions(session.state):
        if action.template_id != action_type:
            continue
        matched = True
        for key in ("tile", "tiles"):
            if key in payload and action.params.get(key) != payload[key]:
                matched = False
        if matched:
            return action
    raise PlayError(f"非法动作: {action_type} {payload}")


def _mahjong_apply_human(session: GameSession, action: ActionInstance) -> None:
    session.state = session.engine.apply_action(session.state, action)
    while session.engine.get_node_type(session.state) == "chance":
        _, session.state = session.engine.sample_chance(session.state)


def mahjong_auto_pass_claim(session: GameSession) -> bool:
    """自动替人类「过」掉无选择的 claim 回合（快照只呈现真实决策点）。

    麻将 claim 阶段按 胡 → 碰/杠 → 吃 三档轮询每个响应者：当轮到人类
    且合法动作**只剩** ``claim_pass``（碰/杠/胡/吃全都不可用）时，这次
    「过」不承载任何决策——直接替玩家执行，对局继续推进。任一碰/杠/
    胡/吃可选时照常停下来交给玩家（claim 操作条）。

    曾由此产生的体验问题：AI 每打一张牌玩家都要连点三次「过」，且
    出牌按钮在 claim 阶段仍然可点 →「非法动作: discard …」。

    Returns:
        是否执行了一次无选择 pass（``False`` 时状态原样不动）。
    """
    if session.over or session.current_player != session.player_pid:
        return False
    if session.state["env"].get("phase") != "claim":
        return False
    legal = session.engine.get_legal_actions(session.state)
    if not legal or any(action.template_id != "claim_pass" for action in legal):
        return False
    pass_action = next(action for action in legal if action.template_id == "claim_pass")
    session.state = session.engine.apply_action(session.state, pass_action)
    while session.engine.get_node_type(session.state) == "chance":
        _, session.state = session.engine.sample_chance(session.state)
    return True


def _mahjong_run_ai(session: GameSession, on_ai_action: Callable[[ActionInstance], None] | None = None) -> None:
    """Drive every non-human seat (4-player: the three AI seats; 2-player
    (explicit): the single AI) through their draw/claim/discard turns.

    人类的无选择 claim 回合由 :func:`mahjong_auto_pass_claim` 自动跳过：
    循环只在「轮到人类做真实决策」（出牌阶段，或碰/杠/胡可选的 claim）
    时才交还控制权。
    """
    while not session.over and session.current_player is not None:
        if session.current_player == session.player_pid:
            if mahjong_auto_pass_claim(session):
                continue
            break  # 真实决策点：交还控制权
        action = session.solver.select_action(session.state)
        if action is None:  # heuristic found nothing — random fallback
            legal = session.engine.get_legal_actions(session.state)
            action = random.choice(legal) if legal else None
        if action is None:
            break
        session.state = session.engine.apply_action(session.state, action)
        while session.engine.get_node_type(session.state) == "chance":
            _, session.state = session.engine.sample_chance(session.state)
        session.last_ai_info["action"] = action.canonical_key
        if on_ai_action is not None:
            on_ai_action(action)


def _mahjong_snapshot(session: GameSession) -> dict:
    env = session.state["env"]
    arrs = session.state["_arrays"]
    seats = session.spec.seat_options
    over = session.over

    def _hand(pid: str) -> list:
        return list(arrs.get(f"hand_{pid}", []))

    def _melds(pid: str) -> list:
        return list(arrs.get(f"melds_{pid}", []))

    def _discards(pid: str) -> list:
        return list(arrs.get(f"discard_{pid}", []))

    legal = []
    if not over and session.current_player == session.player_pid:
        for action in session.engine.get_legal_actions(session.state):
            legal.append({"type": action.template_id, **action.params})

    return {
        "game_id": session.game_id,
        "player_pid": session.player_pid,
        "ai_pid": session.ai_pid,
        "difficulty": session.difficulty,
        "over": over,
        "winner": session.winner,
        # During a claim the effective actor is the queue head, not the
        # During a claim the effective actor is the queue head.
        "turn": ((env.get("claim_queue") or [None])[int(env.get("claim_index", 0))])
        if env.get("phase") == "claim"
        else session.current_player,
        "phase": env.get("phase"),
        "my_hand": _hand(session.player_pid),
        "ai_hand": _hand(session.ai_pid) if over else [],
        "hand_counts": {pid: len(_hand(pid)) for pid in seats},
        "melds": {pid: _melds(pid) for pid in seats},
        "discards": {pid: _discards(pid) for pid in seats},
        "wall_remaining": int(env.get("wall_count", 0)),
        "last_discard": env.get("last_discard"),
        #: 谁打的这张牌（claim 操作条用它提示「对家打出了 X，请响应」）。
        "last_discarder": env.get("last_discarder"),
        #: 刚摸进的牌（do_draw 写 env.last_drawn）：轮到你且 phase=action 时
        #: 就是你这手刚摸的那张，前端用于高亮「新摸的牌」。
        "last_drawn": env.get("last_drawn"),
        "last_action": env.get("last_action"),
        "done": list(env.get("done", [])),
        "winners": list(env.get("winners", [])),
        "payoffs": list(env.get("payoffs", [])),
        "claim": {
            "queue": list(env.get("claim_queue", [])),
            "passed": int(env.get("claim_index", 0)),
            "actor": env.get("actor"),
        }
        if env.get("phase") == "claim"
        else None,
        "legal": legal,
        "last_ai_action": session.last_ai_info.get("action"),
    }


def _mahjong_describe_action(action: ActionInstance) -> str:
    if action.template_id == "discard":
        return f"打 {mahjong_tile_name(action.params['tile'])}"
    if action.template_id == "win_self":
        return "自摸"
    if action.template_id == "claim_win":
        return "荣和"
    if action.template_id == "claim_peng":
        return f"碰 {mahjong_tile_name(action.params['tile'])}"
    if action.template_id == "claim_gang":
        return f"明杠 {mahjong_tile_name(action.params['tile'])}"
    if action.template_id == "claim_chi":
        tiles = action.params.get("tiles", [])
        return "吃 " + "".join(mahjong_tile_name(t) for t in tiles)
    if action.template_id == "gang_concealed":
        return f"暗杠 {mahjong_tile_name(action.params['tile'])}"
    if action.template_id == "gang_added":
        return f"加杠 {mahjong_tile_name(action.params['tile'])}"
    if action.template_id == "claim_pass":
        return "过"
    return action.canonical_key


def _noop_resolve_start(session: GameSession) -> None:
    pass


def _board_ai_opens(session: GameSession) -> bool:
    return session.player_pid == "p_white"


def _poker_ai_opens(session: GameSession) -> bool:
    return True


# ── UNO (classic / seven_zero / jump_in / stacking / draw_until / strict_wild4)


def _make_uno_engine(variant: str) -> Callable[..., GameEngine]:
    """构造 UNO 变种引擎（同一 uno.json + variants 声明选择，默认 4 人）。

    ``allow_codegen=False``：走纯解释器路径。实测（2026-09 平台接入冒烟）
    UNO 的编译快路径在 ``draw_result`` 等阶段即与解释器分叉（compiled=1
    vs interp=2 合法动作），并会在牌堆耗尽后返回 0 合法动作导致对局卡死
    ——正是审查报告 P2-6~10 编译/解释器静默分叉组的现实表现。引擎测试
    （tests/test_layer2_engine/test_uno.py，39 例）全部走解释器路径，
    故平台同样固定解释器（正确性优先；见该测试文件头部说明）。
    """

    def _create(seed: int, player_count: int = 4) -> GameEngine:
        return engine_from_rules("uno", seed, variant=variant, player_count=player_count, allow_codegen=False)

    return _create


def _make_uno_solver(game_id: str) -> Callable[[SolverProvider, GameEngine, int, int], SolverHandle]:
    """UNO 手牌部分可观测 → 与德州同样的不完全信息 Hybrid 求解器
    （train-cli 注册表 runtime_configs 注入 ``imperfect_information=True``）。

    交互节奏调优（PRD §4.3.2：AI 思考 ≤1s/5s/15s）：UNO 引擎对内联
    ``hand_of`` switch 自动回退纯解释器（设计内行为），单步求值昂贵——
    平台运行时默认的 CFR 1000×depth8 + 大预算会让单次决策达 ~3.5s、
    三 AI 座位轮转不可交互。此处显式传入轻量搜索参数（kwargs 覆盖
    runtime 默认）：budget 25/50/100 实测 ≈0.5s/1.3s/2.6s 每次决策。
    """

    def _create(provider: SolverProvider, engine: GameEngine, seed: int, budget: int) -> SolverHandle:
        return provider.create_solver(
            game_id,
            "hybrid",
            engine,
            seed,
            budget,
            cfr_iterations=100,
            cfr_depth_limit=4,
        )

    return _create


def _uno_resolve_start(session: GameSession) -> None:
    """发牌与翻首张（chance 节点）直到进入 play 阶段。"""
    while session.engine.get_node_type(session.state) == "chance":
        _, session.state = session.engine.sample_chance(session.state)


def _uno_parse_human_action(session: GameSession, payload: dict) -> ActionInstance:
    """匹配合法动作：按 template_id + 可选 card/color/target 参数。

    UNO 动作形态：``play``/``play_wild``（带 card，wild 另带 color）、
    ``play7``（带 target，seven_zero）、``draw``/``pick``/``pass``、
    ``play_drawn``、``jump_play``/``jump_pass``、``stack2``/``stack4``/
    ``take_penalty``。所有参数都从 ``ActionInstance.params`` 取，匹配策略
    与麻将一致（payload 缺省的键不参与比对）。
    """
    if session.over:
        raise PlayError("本局已结束")
    if session.current_player != session.player_pid:
        raise PlayError("还没轮到你")
    action_type = payload.get("type")
    for action in session.engine.get_legal_actions(session.state):
        if action.template_id != action_type:
            continue
        matched = True
        for key in ("card", "color", "target"):
            if key in payload and action.params.get(key) != payload[key]:
                matched = False
        if matched:
            return action
    raise PlayError(f"非法动作: {action_type} {payload}")


def _uno_apply_human(session: GameSession, action: ActionInstance) -> None:
    session.state = session.engine.apply_action(session.state, action)
    while session.engine.get_node_type(session.state) == "chance":
        _, session.state = session.engine.sample_chance(session.state)


def _uno_run_ai(session: GameSession, on_ai_action: Callable[[ActionInstance], None] | None = None) -> None:
    """驱动所有非人类座位（4 人局：p1/p2/p3 三个 AI 座位）的摸/出/罚牌轮转。"""
    while not session.over and session.current_player is not None and session.current_player != session.player_pid:
        action = session.solver.select_action(session.state)
        if action is None:  # 搜索无果 — 随机兜底
            legal = session.engine.get_legal_actions(session.state)
            action = random.choice(legal) if legal else None
        if action is None:
            break
        session.state = session.engine.apply_action(session.state, action)
        while session.engine.get_node_type(session.state) == "chance":
            _, session.state = session.engine.sample_chance(session.state)
        session.last_ai_info["action"] = action.canonical_key
        if on_ai_action is not None:
            on_ai_action(action)


def _uno_snapshot(session: GameSession) -> dict:
    """UNO 公开视图：本人手牌、各座张数、台面顶牌、方向、罚牌、合法动作。

    隐藏信息红线：他人手牌只暴露张数（``hand_counts``），牌面永不公开；
    ``ai_hand`` 仅终局展示（与德州 ``ai_hole`` 同门）。``env.handsSnapshot``
    是 7-0 换手 scratch 字段，已由规则 visibility 隐藏 + 轮转后清空（P1-3），
    此处亦不读取。
    """
    env = session.state["env"]
    arrs = session.state["_arrays"]
    seats = session.spec.seat_options
    over = session.over

    def _hand(pid: str) -> list:
        return list(arrs.get(f"hand_{pid}", []))

    legal: list[dict] = []
    if not over and session.current_player == session.player_pid:
        for action in session.engine.get_legal_actions(session.state):
            legal.append({"type": action.template_id, **action.params})

    discard = list(arrs.get("discard", []))
    # 牌堆剩余 = 总牌数 − 各家手牌 − 弃牌堆（108 张标准 UNO）。
    total_cards = len(session.engine._constants.get("card_ids") or [])  # type: ignore[attr-defined]
    in_play = sum(len(_hand(pid)) for pid in seats) + len(discard)
    deck_count = max(0, total_cards - in_play)

    return {
        "game_id": session.game_id,
        "player_pid": session.player_pid,
        "ai_pid": session.ai_pid,
        "difficulty": session.difficulty,
        "over": over,
        "winner": session.winner,
        "turn": session.current_player,
        "phase": env.get("phase"),
        "direction": int(env.get("direction", 1)),
        "top_color": env.get("topColor"),
        "top_symbol": env.get("topSymbol"),
        "my_hand": _hand(session.player_pid),
        "ai_hand": _hand(session.ai_pid) if over else [],
        "hand_counts": {pid: len(_hand(pid)) for pid in seats},
        "discard_top": discard[-1] if discard else None,
        "discard_recent": discard[-5:],
        "deck_count": deck_count,
        "pending_draw": int(env.get("pendingDraw", 0)),
        "penalty_target": env.get("penaltyTarget"),
        "last_action": env.get("lastActor"),
        "last_ai_action": session.last_ai_info.get("action"),
        "legal": legal,
        "payoff": session.engine.get_utility(session.state, session.player_pid) if over else None,
    }


def _uno_describe_action(action: ActionInstance) -> str:
    return action.canonical_key


def _uno_ai_opens(session: GameSession) -> bool:
    """首张特殊牌会改写先手（skip→跳过下家、reverse→反转方向）：
    发牌后只要当前行动者不是人类座位，AI 座位就先行。"""
    return session.current_player != session.player_pid


# 注意：平台注册表共 15 款 = 月亮棋/随机五子棋/德州 + 麻将六变种
# （guangdong / hongzhong / blood / sichuan / changsha / taiwan，v5.2
# variants）+ UNO 六变体。三个消费注册点必须同步，否则各自漂移（曾漏挂
# 四川/长沙/台湾，导致文档承诺六变种但大厅只有三个）：
#   - 平台：本文件（平台 /api/games → 大厅；UNO 已注册但前端
#     FAMILY_BOARDS 尚无 "uno" 条目——LobbyPage 对 uno 族卡片置灰标注
#     「前端界面开发中」，BattlePage 有占位守卫防崩溃）
#   - 训练：train-cli/games.py `_mahjong_spec`（六变体 × MARL 管线）
#   - 文档：docs/user/play_mahjong.md（六变体 × 默认 4 人）
GAMES: dict[str, GameSpec] = {
    "moon_chess": GameSpec(
        game_id="moon_chess",
        display_name="月亮棋",
        description="3×3 经典月亮棋：三子连珠即胜，棋盘满时最旧的棋子被挤出。",
        kind="board",
        board_size=3,
        seat_options=("p_black", "p_white"),
        seat_label="颜色",
        difficulty_budgets={"easy": 200, "normal": 800, "hard": 2000},
        create_engine=_moon_create_engine,
        create_solver=_moon_create_solver,
        resolve_start=_noop_resolve_start,
        ai_opens=_board_ai_opens,
        parse_human_action=_moon_parse_human_action,
        apply_human=_moon_apply_human,
        run_ai=_moon_run_ai,
        build_snapshot=_moon_snapshot,
        describe_action=_moon_describe_action,
    ),
    "stochastic_gomoku": GameSpec(
        game_id="stochastic_gomoku",
        display_name="随机五子棋",
        description="9×9 五子棋变体：每次落子后棋子有 50% 概率被随机抹去。",
        kind="board",
        board_size=9,
        seat_options=("p_black", "p_white"),
        seat_label="颜色",
        difficulty_budgets={"easy": 300, "normal": 1500, "hard": 4000},
        create_engine=_gomoku_create_engine,
        create_solver=_gomoku_create_solver,
        resolve_start=_noop_resolve_start,
        ai_opens=_board_ai_opens,
        parse_human_action=_gomoku_parse_human_action,
        apply_human=_gomoku_apply_human,
        run_ai=_gomoku_run_ai,
        build_snapshot=_gomoku_snapshot,
        describe_action=_gomoku_describe_action,
    ),
    "texas_holdem": GameSpec(
        game_id="texas_holdem",
        display_name="德州扑克",
        description="双人德州扑克：翻前/翻牌/转牌/河牌四轮下注，AI 使用混合求解器。",
        kind="poker",
        board_size=None,
        seat_options=("p_sb", "p_bb"),  # declared in rules players
        seat_label="座位",
        difficulty_budgets={"easy": 150, "normal": 500, "hard": 1200},
        create_engine=_poker_create_engine,
        create_solver=_poker_create_solver,
        resolve_start=_poker_resolve_start,
        ai_opens=_poker_ai_opens,
        parse_human_action=_poker_parse_human_action,
        apply_human=_poker_apply_human,
        run_ai=_poker_run_ai,
        build_snapshot=_poker_snapshot,
        describe_action=_poker_describe_action,
    ),
    "mahjong_guangdong": GameSpec(
        game_id="mahjong_guangdong",
        display_name="广东麻将（鸡胡）",
        description="四人广东鸡胡：吃碰杠、自摸荣和、清一色等番种，AI 使用启发式策略。",
        kind="mahjong",
        board_size=None,
        seat_options=("p0", "p1", "p2", "p3"),
        seat_label="座位",
        # 麻将标准 4 人（规则默认与训练注册表一致；2 人仅引擎层显式可选）
        player_counts=(4,),
        # 启发式求解器与搜索预算无关，难度档仅作展示（审查 Minor 12）
        difficulty_budgets={"easy": 1, "normal": 1, "hard": 1},
        create_engine=_make_mahjong_engine("guangdong"),
        create_solver=_make_mahjong_solver("mahjong_guangdong"),
        resolve_start=_mahjong_resolve_start,
        ai_opens=lambda session: session.player_pid != "p0",
        parse_human_action=_mahjong_parse_human_action,
        apply_human=_mahjong_apply_human,
        run_ai=_mahjong_run_ai,
        build_snapshot=_mahjong_snapshot,
        describe_action=_mahjong_describe_action,
    ),
    "mahjong_hongzhong": GameSpec(
        game_id="mahjong_hongzhong",
        display_name="红中麻将",
        description="红中万能牌：红中可代任意牌凑搭子，其余规则同鸡胡。",
        kind="mahjong",
        board_size=None,
        seat_options=("p0", "p1", "p2", "p3"),
        seat_label="座位",
        # 麻将标准 4 人（规则默认与训练注册表一致；2 人仅引擎层显式可选）
        player_counts=(4,),
        difficulty_budgets={"easy": 1, "normal": 1, "hard": 1},
        create_engine=_make_mahjong_engine("hongzhong"),
        create_solver=_make_mahjong_solver("mahjong_hongzhong"),
        resolve_start=_mahjong_resolve_start,
        ai_opens=lambda session: session.player_pid != "p0",
        parse_human_action=_mahjong_parse_human_action,
        apply_human=_mahjong_apply_human,
        run_ai=_mahjong_run_ai,
        build_snapshot=_mahjong_snapshot,
        describe_action=_mahjong_describe_action,
    ),
    "mahjong_blood": GameSpec(
        game_id="mahjong_blood",
        # 规则引擎里 blood 变体的语义是血流成河（胡家不退场继续摸打、
        # 可多点累计胡牌），与四川麻将的血战到底（胡家退场）相区分——
        # 旧名「血战到底」与 mahjong_sichuan 撞名，大厅出现两个血战到底。
        display_name="血流成河",
        description="血流成河：胡牌后不退场继续摸打（不能重复胡），可多次胡牌累计番分，直到三家胡牌或牌墙摸空；108 张无字牌、缺一门、禁吃，与四川麻将同源。",
        kind="mahjong",
        board_size=None,
        seat_options=("p0", "p1", "p2", "p3"),
        seat_label="座位",
        # 麻将标准 4 人（规则默认与训练注册表一致；2 人仅引擎层显式可选）
        player_counts=(4,),
        difficulty_budgets={"easy": 1, "normal": 1, "hard": 1},
        create_engine=_make_mahjong_engine("blood"),
        create_solver=_make_mahjong_solver("mahjong_blood"),
        resolve_start=_mahjong_resolve_start,
        ai_opens=lambda session: session.player_pid != "p0",
        parse_human_action=_mahjong_parse_human_action,
        apply_human=_mahjong_apply_human,
        run_ai=_mahjong_run_ai,
        build_snapshot=_mahjong_snapshot,
        describe_action=_mahjong_describe_action,
    ),
    "mahjong_sichuan": GameSpec(
        game_id="mahjong_sichuan",
        display_name="四川麻将（血战到底）",
        description="四人四川麻将（血战到底）：108 张无字牌，缺一门才能胡（硬门槛），禁吃，胡牌后胡家退场、剩余玩家继续，直到三家胡牌或牌墙摸空；番种：平胡 1/对对胡 2/清一色 4/七对 4/龙七对 8/将对 8。",
        kind="mahjong",
        board_size=None,
        seat_options=("p0", "p1", "p2", "p3"),
        seat_label="座位",
        # 麻将标准 4 人（规则默认与训练注册表一致；2 人仅引擎层显式可选）
        player_counts=(4,),
        difficulty_budgets={"easy": 1, "normal": 1, "hard": 1},
        create_engine=_make_mahjong_engine("sichuan"),
        create_solver=_make_mahjong_solver("mahjong_sichuan"),
        resolve_start=_mahjong_resolve_start,
        ai_opens=lambda session: session.player_pid != "p0",
        parse_human_action=_mahjong_parse_human_action,
        apply_human=_mahjong_apply_human,
        run_ai=_mahjong_run_ai,
        build_snapshot=_mahjong_snapshot,
        describe_action=_mahjong_describe_action,
    ),
    "mahjong_changsha": GameSpec(
        game_id="mahjong_changsha",
        display_name="长沙麻将（258将）",
        description="四人长沙麻将（258将）：108 张无字牌，小胡必须 2/5/8 为将，大胡（碰碰胡/清一色/七对/将将胡）乱将豁免；番制：小胡 1 番→10 / 大胡 6 番→60 / 番上番 12 番→120。",
        kind="mahjong",
        board_size=None,
        seat_options=("p0", "p1", "p2", "p3"),
        seat_label="座位",
        # 麻将标准 4 人（规则默认与训练注册表一致；2 人仅引擎层显式可选）
        player_counts=(4,),
        difficulty_budgets={"easy": 1, "normal": 1, "hard": 1},
        create_engine=_make_mahjong_engine("changsha"),
        create_solver=_make_mahjong_solver("mahjong_changsha"),
        resolve_start=_mahjong_resolve_start,
        ai_opens=lambda session: session.player_pid != "p0",
        parse_human_action=_mahjong_parse_human_action,
        apply_human=_mahjong_apply_human,
        run_ai=_mahjong_run_ai,
        build_snapshot=_mahjong_snapshot,
        describe_action=_mahjong_describe_action,
    ),
    "mahjong_taiwan": GameSpec(
        game_id="mahjong_taiwan",
        display_name="台湾麻将（16张）",
        description="四人台湾麻将（16张）：无花简化 136 张，庄 17 张闲 16 张，5 副+将成胡，呖咕呖咕（八对半）可胡；台数：平胡 2/门清 1/自摸 1/碰碰胡 4/混一色 4/清一色 8。",
        kind="mahjong",
        board_size=None,
        seat_options=("p0", "p1", "p2", "p3"),
        seat_label="座位",
        # 麻将标准 4 人（规则默认与训练注册表一致；2 人仅引擎层显式可选）
        player_counts=(4,),
        difficulty_budgets={"easy": 1, "normal": 1, "hard": 1},
        create_engine=_make_mahjong_engine("taiwan"),
        create_solver=_make_mahjong_solver("mahjong_taiwan"),
        resolve_start=_mahjong_resolve_start,
        ai_opens=lambda session: session.player_pid != "p0",
        parse_human_action=_mahjong_parse_human_action,
        apply_human=_mahjong_apply_human,
        run_ai=_mahjong_run_ai,
        build_snapshot=_mahjong_snapshot,
        describe_action=_mahjong_describe_action,
    ),
    # ── UNO：六变体（与 train-cli/games.py `_uno_spec` 对齐；同一 uno.json
    # 的 variants 声明选择，默认 4 人）。求解器与德州同为不完全信息 Hybrid。
    # 三个消费注册点同步：平台（本文件）/ 训练（train-cli）/ 文档
    # （docs/user/play_uno.md）。
    "uno": GameSpec(
        game_id="uno",
        display_name="UNO（经典）",
        description="四人经典 UNO：108 张牌，同色或同符号接牌，先清空手牌者胜。",
        kind="uno",
        board_size=None,
        seat_options=("p0", "p1", "p2", "p3"),
        seat_label="座位",
        player_counts=(4,),
        difficulty_budgets={"easy": 25, "normal": 50, "hard": 100},
        create_engine=_make_uno_engine("classic"),
        create_solver=_make_uno_solver("uno"),
        resolve_start=_uno_resolve_start,
        ai_opens=_uno_ai_opens,
        parse_human_action=_uno_parse_human_action,
        apply_human=_uno_apply_human,
        run_ai=_uno_run_ai,
        build_snapshot=_uno_snapshot,
        describe_action=_uno_describe_action,
    ),
    "uno_seven_zero": GameSpec(
        game_id="uno_seven_zero",
        display_name="UNO 7-0（换手/移交）",
        description="UNO 7-0 变体：打出 7 可与任一玩家换手，打出 0 全场手牌按方向移交。",
        kind="uno",
        board_size=None,
        seat_options=("p0", "p1", "p2", "p3"),
        seat_label="座位",
        player_counts=(4,),
        difficulty_budgets={"easy": 25, "normal": 50, "hard": 100},
        create_engine=_make_uno_engine("seven_zero"),
        create_solver=_make_uno_solver("uno_seven_zero"),
        resolve_start=_uno_resolve_start,
        ai_opens=_uno_ai_opens,
        parse_human_action=_uno_parse_human_action,
        apply_human=_uno_apply_human,
        run_ai=_uno_run_ai,
        build_snapshot=_uno_snapshot,
        describe_action=_uno_describe_action,
    ),
    "uno_jump_in": GameSpec(
        game_id="uno_jump_in",
        display_name="UNO 抢牌",
        description="UNO 抢牌变体：他人出牌后，持同色同数字牌者可抢出（jump-in）。",
        kind="uno",
        board_size=None,
        seat_options=("p0", "p1", "p2", "p3"),
        seat_label="座位",
        player_counts=(4,),
        difficulty_budgets={"easy": 25, "normal": 50, "hard": 100},
        create_engine=_make_uno_engine("jump_in"),
        create_solver=_make_uno_solver("uno_jump_in"),
        resolve_start=_uno_resolve_start,
        ai_opens=_uno_ai_opens,
        parse_human_action=_uno_parse_human_action,
        apply_human=_uno_apply_human,
        run_ai=_uno_run_ai,
        build_snapshot=_uno_snapshot,
        describe_action=_uno_describe_action,
    ),
    "uno_stacking": GameSpec(
        game_id="uno_stacking",
        display_name="UNO +2 叠加",
        description="UNO 叠加变体：+2/+4 可被下一家继续叠加，累计罚牌由无力叠加者全部吃下。",
        kind="uno",
        board_size=None,
        seat_options=("p0", "p1", "p2", "p3"),
        seat_label="座位",
        player_counts=(4,),
        difficulty_budgets={"easy": 25, "normal": 50, "hard": 100},
        create_engine=_make_uno_engine("stacking"),
        create_solver=_make_uno_solver("uno_stacking"),
        resolve_start=_uno_resolve_start,
        ai_opens=_uno_ai_opens,
        parse_human_action=_uno_parse_human_action,
        apply_human=_uno_apply_human,
        run_ai=_uno_run_ai,
        build_snapshot=_uno_snapshot,
        describe_action=_uno_describe_action,
    ),
    "uno_draw_until": GameSpec(
        game_id="uno_draw_until",
        display_name="UNO 摸到能打",
        description="UNO 摸到能打变体：无牌可接时持续摸牌，直到摸到可打的牌（牌堆空则停）。",
        kind="uno",
        board_size=None,
        seat_options=("p0", "p1", "p2", "p3"),
        seat_label="座位",
        player_counts=(4,),
        difficulty_budgets={"easy": 25, "normal": 50, "hard": 100},
        create_engine=_make_uno_engine("draw_until"),
        create_solver=_make_uno_solver("uno_draw_until"),
        resolve_start=_uno_resolve_start,
        ai_opens=_uno_ai_opens,
        parse_human_action=_uno_parse_human_action,
        apply_human=_uno_apply_human,
        run_ai=_uno_run_ai,
        build_snapshot=_uno_snapshot,
        describe_action=_uno_describe_action,
    ),
    "uno_strict_wild4": GameSpec(
        game_id="uno_strict_wild4",
        display_name="UNO 严格+4",
        description="UNO 严格+4 变体：仍有台面颜色可接时禁止出 +4，违规质疑成立罚 4 张。",
        kind="uno",
        board_size=None,
        seat_options=("p0", "p1", "p2", "p3"),
        seat_label="座位",
        player_counts=(4,),
        difficulty_budgets={"easy": 25, "normal": 50, "hard": 100},
        create_engine=_make_uno_engine("strict_wild4"),
        create_solver=_make_uno_solver("uno_strict_wild4"),
        resolve_start=_uno_resolve_start,
        ai_opens=_uno_ai_opens,
        parse_human_action=_uno_parse_human_action,
        apply_human=_uno_apply_human,
        run_ai=_uno_run_ai,
        build_snapshot=_uno_snapshot,
        describe_action=_uno_describe_action,
    ),
}
