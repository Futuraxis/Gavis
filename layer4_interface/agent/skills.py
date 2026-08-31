"""skills — 确定性技能层（Layer 4 陪伴后端）.

:class:`Skills` 从 :class:`SkillContext`（唯一数据入口 ``Skills.build``
构造）产出机械事实 dict，供对话引擎成文 / 复盘 / 提示使用。技能层只读
投影观测与公开评估，绝不 import ``layer3_solvers``；求解器只在
:meth:`Skills.suggest_hint` 的 ``specific`` / ``demo`` 级别经由注入的
``SolverProvider``（或会话 ``SolverHandle``）获取，不直接依赖 Layer 3。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..frontend.engine_helpers import canonical_family_text, game_family
from ..solver_provider import SolverHandle, SolverProvider
from .evaluation import evaluate
from .hidden_guard import assert_no_hidden, infer_game_id

#: 视作"好棋"的评估下限；低于其相反数视作"失误"。
_GOOD_THRESHOLD = 1.0


@dataclass
class SkillContext:
    """技能层唯一数据入口产出：一次 build 快照的确定性上下文.

    Attributes:
        human_pid: 人类玩家 id（观测视角）。
        observation: ``engine.project_observation(state, human_pid)`` 投影。
        legal_actions: ``engine.get_legal_actions(state)`` 结果。
        evaluation: ``evaluation.evaluate(state, human_pid, engine)`` 结果。
        revealed: 隐藏信息放行门（对局结束且已 reveal，如德州 showdown）。
    """

    human_pid: str
    observation: dict[str, Any]
    legal_actions: list[Any]
    evaluation: dict[str, Any]
    revealed: bool


class Skills:
    """确定性技能：只读公开信息，产出机械事实."""

    @staticmethod
    def build(state: dict[str, Any], human_pid: str, engine: Any) -> SkillContext:
        """从状态构建 :class:`SkillContext`（唯一数据入口，遵守红线）.

        只调用 ``engine.project_observation`` 与 ``engine.get_legal_actions``，
        并校验观测不含隐藏字段。
        """
        observation = engine.project_observation(state, human_pid)
        legal_actions = engine.get_legal_actions(state)
        evaluation = evaluate(state, human_pid, engine)
        revealed = bool(engine.is_terminal(state)) and state.get("env", {}).get("last_action") == "showdown"
        ctx = SkillContext(
            human_pid=human_pid,
            observation=observation,
            legal_actions=legal_actions,
            evaluation=evaluation,
            revealed=revealed,
        )
        assert_no_hidden(ctx)
        return ctx

    @staticmethod
    def evaluate_position(ctx: SkillContext, engine: Any) -> dict[str, Any]:
        """返回当前局面评估 ``{score, summary, mechanical_text}``."""
        return dict(ctx.evaluation)

    @staticmethod
    def detect_good_move(ctx: SkillContext, engine: Any) -> dict[str, Any] | None:
        """识别玩家好棋（评估占优则给出正向反馈事实），否则 ``None``."""
        score = float(ctx.evaluation.get("score", 0.0))
        if score < _GOOD_THRESHOLD:
            return None
        return {
            "kind": "good_move",
            "score": score,
            "summary": ctx.evaluation.get("summary", ""),
            "mechanical_text": ctx.evaluation.get("mechanical_text", ""),
        }

    @staticmethod
    def detect_blunder(ctx: SkillContext, engine: Any) -> dict[str, Any] | None:
        """识别玩家失误（评估明显落后则给出风险事实），否则 ``None``."""
        score = float(ctx.evaluation.get("score", 0.0))
        if score > -_GOOD_THRESHOLD:
            return None
        return {
            "kind": "blunder",
            "score": score,
            "summary": ctx.evaluation.get("summary", ""),
            "mechanical_text": ctx.evaluation.get("mechanical_text", ""),
        }

    @staticmethod
    def suggest_hint(
        ctx: SkillContext,
        level: str,
        provider: SolverProvider | SolverHandle | None,
        engine: Any,
    ) -> dict[str, Any]:
        """按提示级别给方向 / 具体建议 / 演示（不直接 import L3）.

        ``direction`` 纯评估成文；``specific`` / ``demo`` 从合法动作里
        确定性选一手描述（会话级接线持有实时 state，会用注入的
        ``provider`` 求解真实走法，本确定性半边只保证结构与无泄露）。
        """
        score = float(ctx.evaluation.get("score", 0.0))
        direction = _direction_hint(score)
        result: dict[str, Any] = {
            "level": level,
            "direction": direction,
            "mechanical_text": ctx.evaluation.get("mechanical_text", ""),
        }
        if level == "direction":
            result["hint"] = direction
            return result

        action = _pick_legal_action(ctx.legal_actions)
        if action is not None:
            key = action.canonical_key if hasattr(action, "canonical_key") else str(action)
            result["action"] = key
            # 机器键保留在 ``result["action"]`` 供校验/回放；给 LLM 的提示文案
            # 用中文描述（``canonical_family_text``）：LLM 读“打出 一条”而不是
            # “discard:s1”。
            family = game_family(infer_game_id(ctx.observation))
            human = canonical_family_text(family, key)
            result["hint"] = f"演示走法：{human}" if level == "demo" else f"具体建议：{human}"
        else:
            result["hint"] = direction
        return result

    @staticmethod
    def summarize_result(ctx: SkillContext, engine: Any, winner: str, player_pid: str) -> dict[str, Any]:
        """赛后胜负机械事实."""
        won = winner == player_pid
        # 不含原始 pid：摘要经对话载荷渗入 LLM 文本时，pid 会被复述成
        # 「p_sb 赢了」（见 evaluation.py 同源修复）。「本方」= 玩家视角。
        summary = "本方获胜" if won else "本方落败"
        return {
            "winner": winner,
            "player_pid": player_pid,
            "won": won,
            "summary": summary,
            "mechanical_text": "对局结束",
        }

    @staticmethod
    def explain_illegal(ctx: SkillContext, engine: Any, attempted: dict[str, Any]) -> dict[str, Any]:
        """违规操作机械事实（说明原因，不责备）."""
        return {
            "attempted": attempted,
            "reason": "该操作不符合当前阶段的规则",
            "summary": "这一步暂时不能走，规则不允许",
            "mechanical_text": f"非法操作：{attempted}",
        }

    @staticmethod
    def idle_reminder(ctx: SkillContext) -> dict[str, Any]:
        """长时间未操作提醒（先等待后提醒，不催促）."""
        return {
            "summary": "还在想吗？不着急，慢慢来。",
            "mechanical_text": "玩家长时间未操作",
        }

    @staticmethod
    def greet(ctx: SkillContext, profile: dict[str, Any] | None) -> dict[str, Any]:
        """开局问候机械事实（可提及昵称 / 上次战绩）."""
        profile = profile or {}
        nickname = str(profile.get("nickname", "") or "")
        recent = profile.get("recent", {}) if isinstance(profile.get("recent"), dict) else {}
        parts = ["greet"]
        if nickname:
            parts.append(f"昵称：{nickname}")
        if recent:
            parts.append(f"最近战绩：{recent}")
        summary = f"欢迎{nickname}" if nickname else "欢迎"
        return {
            "kind": "greet",
            "nickname": nickname,
            "recent": recent,
            "summary": summary,
            "mechanical_text": "；".join(parts),
        }


def _direction_hint(score: float) -> str:
    """按评估值给出方向性建议（通用、无游戏特供）."""
    if score > _GOOD_THRESHOLD:
        return "当前占优，可以稳扎稳打"
    if score < -_GOOD_THRESHOLD:
        return "当前落后，先补强防守"
    return "局面胶着，优先占住关键位置"


def _pick_legal_action(legal_actions: list[Any]) -> Any | None:
    """从合法动作里确定性选一手（按 canonical_key 排序取中位）."""
    if not legal_actions:
        return None
    keyed = [(getattr(action, "canonical_key", str(action)), action) for action in legal_actions]
    keyed.sort(key=lambda pair: pair[0])
    return keyed[len(keyed) // 2][1]
