"""导入桥 — 让 ``train-cli/`` 文件夹可被模块化导入.

``train-cli`` 目录名含连字符，Python 无法把它当包导入（``import train-cli``
非法）。本模块是唯一别名：把 ``train-cli/`` 加入 ``sys.path`` 并再导出其公共
API，使以下用法成立：

- ``from train_cli import GAMES, create_solver, default_provider, ...``
- ``python -m train_cli`` → 等价于 ``python train-cli/train.py``

它不是适配器/装配器——只是连字符目录的命名空间桥；全部装配逻辑仍在
``train-cli/games.py`` 的注册表数据里。
"""

from __future__ import annotations

import sys
from pathlib import Path

_TRAIN_CLI_DIR = Path(__file__).resolve().parent / "train-cli"
if str(_TRAIN_CLI_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_CLI_DIR))

from games import (  # noqa: E402
    GAMES,
    SOLVER_FACTORY,
    DefaultSolverProvider,
    EngineSpec,
    GameSpec,
    RandomSolver,
    SolverPipeline,
    create_solver,
    default_provider,
    registered_game_ids,
    registered_solver_names,
)
from train import EVAL_MCTS_BUDGET, MAX_EVAL_STEPS, build_engine, evaluate, play_episode  # noqa: E402

__all__ = [
    "GAMES",
    "GameSpec",
    "EngineSpec",
    "SolverPipeline",
    "SOLVER_FACTORY",
    "RandomSolver",
    "DefaultSolverProvider",
    "default_provider",
    "create_solver",
    "registered_game_ids",
    "registered_solver_names",
    "build_engine",
    "evaluate",
    "play_episode",
    "MAX_EVAL_STEPS",
    "EVAL_MCTS_BUDGET",
]


if __name__ == "__main__":
    from train import main  # noqa: E402  (train-cli/train.py)

    main()
