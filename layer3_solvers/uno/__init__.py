"""UNO 求解器组件。

当前仅提供 ``UnoRolloutPolicy``（MCTS rollout 启发式先验）——给裸 MCTS
的随机 rollout 注入 UNO 出牌常识，提升统计信号质量。与麻将的
``MahjongHeuristicAI`` 同属"启发式兜底"家族，但本策略是 rollout 先验
（注入 ``hybrid.mcts.rollout_policy``），而非独立 SolverBase。
"""

from .heuristic import UnoRolloutPolicy, _play_score, _split_card

__all__ = ["UnoRolloutPolicy", "_play_score", "_split_card"]
