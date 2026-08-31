"""result — faction-aware win/loss resolution (Layer 4).

社交推理游戏（undercover / werewolf）的 ``env.winner`` 是**阵营 id**
（``"undercover"`` / ``"civilian"`` / ``"blank"`` / ``"wolf"`` / ``"good"``），
从来不是玩家 pid；其余游戏族是 pid（网格/扑克/UNO）或 ``winners`` pid 列表
（麻将多胡局）。因此任何回答「玩家赢没赢」的层都必须把胜者与**观测者的身份**
做阵营比对，而不是 ``winner == player_pid``——否则卧底获胜的一局会被全部下游
（对话引擎 outcome、平台聊天结果标签、复盘摘要、前端结算弹窗）误报成
「AI 获胜（玩家落败）」。

本模块是零依赖的纯函数集，供 agent 对话引擎 / platform 聊天与会话 / review
复盘分析器 / 前端共享，保证「相对观测者谁赢了」只有一处判据：

- ``winner == viewer`` 或 ``viewer ∈ winners`` → 胜；
- ``winner is None``：
  - ``winners`` 非空（麻将多胡）→ 以 ``viewer ∈ winners`` 判定；
  - 否则 → ``None``（平局 / 无胜者）；
- 其余（胜者是阵营或他人的 pid）：
  - 快照含 ``final_roles``（social 族终局公开的身份表）→ ``viewer`` 的身份
    等于胜者阵营才判胜（卧底局玩家是卧底、winner=undercover → 玩家胜）；
  - 快照无身份表、但调用方有引擎 per-viewer 终局效用（``engine.get_utility``
    的符号 ±1/0，见 ``agent/evaluation.py``）→ 用其符号；
  - 都不满足 → ``False``（单一胜者的常规对局：胜者不是玩家即玩家落败）。

``final_roles`` 是公开终局信息（social 族快照在 ``over=True`` 时揭晓全员身份，
复盘记录最后一手快照同样携带），读取它不触碰隐藏信息红线。
"""

from __future__ import annotations

from typing import Any

__all__ = ["faction_matches", "player_won", "role_of"]


def faction_matches(role: str, winner: Any) -> bool:
    """``role`` 是否属于 ``winner`` 阵营（社交族终局身份表比对用）。

    - 身份名 == 胜者阵营（卧底局：undercover/civilian/blank 词表同源）→ 胜；
    - 狼人杀：胜者阵营是 ``good``/``wolf``，而身份表里是具体的
      ``villager/seer/witch/hunter/guard``（好人侧）/ ``wolf``——``winner=good``
      时任何非狼身份获胜，``winner=wolf`` 时只有狼身份获胜；
    - 其余 → 不属于。
    """
    if role == winner:
        return True
    if winner == "good":
        return role != "wolf"
    if winner == "wolf":
        return role == "wolf"
    return False


def role_of(snap: Any, pid: str) -> str | None:
    """从快照的终局身份表解析 ``pid`` 的身份（无表/无行 → ``None``）。

    ``snap`` 可以是 social 族快照（含 ``final_roles``）、复盘记录里最后一手
    的 ``snapshot`` dict、或任意带 ``final_roles`` 的字典；非 dict 直接返回
    ``None``，绝不在缺失时抛异常。
    """
    if not isinstance(snap, dict):
        return None
    rows = snap.get("final_roles")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("pid") != pid:
            continue
        role = row.get("role")
        if role not in (None, ""):
            return str(role)
    return None


def player_won(
    winner: Any,
    player_pid: str,
    winners: Any = None,
    snap: Any = None,
    *,
    score: Any = None,
) -> bool | None:
    """判定 ``player_pid`` 是否赢得已结束的对局。

    Args:
        winner: ``env.winner``（pid 或阵营名；``None``/``""`` = 无胜者）。
        player_pid: 观测者（玩家 / 对手模式下的 AI）pid。
        winners: ``env.winners`` 列表（麻将多胡局；``None``/空 = 无）。
        snap: 快照 / 复盘最后一手 snapshot（可含 ``final_roles``）。
        score: 可选，引擎 per-viewer 终局效用（``engine.get_utility``），
            正/负号作为身份表缺失时的兜底判据。

    Returns:
        ``True`` / ``False`` 可判定时；``None`` = 平局（无胜者且无 winners）
        或「胜者为阵营但身份信息与效用都不可得」。
    """
    if winner is None or winner == "":
        wins = winners if winners is not None else (snap.get("winners") if isinstance(snap, dict) else None)
        if isinstance(wins, list) and wins:
            return player_pid in wins
        return None

    if winner == player_pid or player_pid in (winners or []):
        return True

    role = role_of(snap, player_pid)
    if role is not None:
        return faction_matches(role, winner)

    if score is not None:
        try:
            s = float(score)
        except (TypeError, ValueError):
            s = 0.0
        if s > 0:
            return True
        if s < 0:
            return False

    # 单一胜者的常规对局：胜者不是玩家 → 玩家落败。
    return False