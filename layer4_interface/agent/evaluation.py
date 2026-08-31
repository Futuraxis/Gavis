"""evaluate — 通用兜底局面评估（Layer 4，确定性、不泄露隐藏信息）.

终局直接读 :meth:`GameEngine.get_utility`；非终局用 ``lastPlacedCell``
邻域启发式（占角 / 连线计数等通用启发，不做游戏特供）。只读取公开
棋盘的 ``board`` 数组与公开 ``env`` 字段，绝不触碰底牌 / 手牌 / 身份 /
未翻牌堆等隐藏数组。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

#: 连续棋子成线时每多一枚的加分权重（通用启发，非游戏特供）。
_LINE_WEIGHT = 0.5
#: 达到该连续长度才开始计入加分。
_MIN_LINE = 2
#: 评估值高于此视为占优、低于其相反数视为落后（用于中文摘要措辞）。
_ADVANTAGE = 1.5


def evaluate(state: dict[str, Any], viewer: str, engine: Any) -> dict[str, Any]:
    """返回通用兜底评估 ``{score, summary, mechanical_text}``.

    Args:
        state: 引擎状态（ground arrays + env）。
        viewer: 被评估的玩家 id。
        engine: :class:`layer2_engine.core.engine.GameEngine`。

    Returns:
        ``score`` 为数值评估（终局取 utility，非终局取邻域启发式），
        ``summary`` / ``mechanical_text`` 为机械中文文本。
    """
    if engine.is_terminal(state):
        utility = engine.get_utility(state, viewer)
        score = float(utility)
        # 摘要为 viewer 相对、不含原始 pid（pid 如 p_sb 会经对话载荷渗入 LLM
        # 文本，被复述成「p_sb 赢了」——见 audit B 修复）。「本方」= 评估视角
        # （companion/teaching 取玩家、opponent 取 AI），LLM 据人设自然成文。
        if score > 0:
            summary = "本方获胜"
        elif score < 0:
            summary = "本方落败"
        else:
            summary = "平局"
        mechanical_text = f"终局，本方效用 {score:+.1f}"
        return {"score": score, "summary": summary, "mechanical_text": mechanical_text}

    score, summary = _board_heuristic(state, viewer)
    mechanical_text = f"本方当前评估 {score:+.2f}，{summary}"
    return {"score": score, "summary": summary, "mechanical_text": mechanical_text}


def _board_heuristic(state: dict[str, Any], viewer: str) -> tuple[float, str]:
    """基于公开棋盘与 ``lastPlacedCell`` 的通用邻域启发式."""
    board = state.get("_arrays", {}).get("board")
    if not isinstance(board, list) or not board:
        return 0.0, "局面暂时难分高下"

    size = int(round(len(board) ** 0.5))
    if size * size != len(board):
        # 非方阵棋盘（当前 P0 游戏都是方阵），退回中性评估。
        return 0.0, "局面暂时难分高下"

    pieces: dict[str, int] = defaultdict(int)
    for cell in board:
        if cell is not None:
            pieces[str(cell)] += 1

    viewer_count = pieces.get(viewer, 0)
    others = sum(count for pid, count in pieces.items() if pid != viewer)
    last = state.get("env", {}).get("lastPlacedCell")
    line_bonus = _line_bonus(board, size, last, viewer)
    score = float(viewer_count - others) + line_bonus

    if score > _ADVANTAGE:
        summary = "本方略占上风"
    elif score < -_ADVANTAGE:
        summary = "本方稍处下风"
    else:
        summary = "局面胶着"
    return score, summary


def _line_bonus(board: list[Any], size: int, last: Any, viewer: str) -> float:
    """统计 ``lastPlacedCell`` 四周 ``viewer`` 的连续棋子长度并加分."""
    if not last:
        return 0.0
    try:
        _, row, col = str(last).split("_")
        row, col = int(row), int(col)
    except (ValueError, TypeError):
        return 0.0

    index = row * size + col
    if index >= len(board) or board[index] != viewer:
        return 0.0

    bonus = 0.0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        rr, cc = row + dr, col + dc
        while 0 <= rr < size and 0 <= cc < size and board[rr * size + cc] == viewer:
            count += 1
            rr += dr
            cc += dc
        rr, cc = row - dr, col - dc
        while 0 <= rr < size and 0 <= cc < size and board[rr * size + cc] == viewer:
            count += 1
            rr -= dr
            cc -= dc
        if count >= _MIN_LINE:
            bonus += _LINE_WEIGHT * count
    return bonus
