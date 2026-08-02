# 随机五子棋 · 人机对弈（使用说明）

人类与 AI 在浏览器里下随机五子棋。AI 使用 MCTS，与月亮棋应用共用
同一套应用架构（独立目录、独立服务器、`frontend/common` 共享工具）。

## 快速开始

```bash
python -m layer4_interface.frontend.play_gomoku.server
```

浏览器打开 **http://127.0.0.1:8767/**。

## 游戏规则

- **棋盘**：9×9
- **落子**：黑先白后，交替落子
- **随机消失**：每手落子后，该子有 **50% 概率消失**（棋盘上抹去）
- **获胜**：任意一方五子连成一线（横/竖/斜）
- **平局**：棋盘下满无人五连

## 界面操作

| 操作 | 说明 |
|------|------|
| 执子选择 | 黑棋先手 / 白棋后手 / 随机。选白棋时 AI 先落子 |
| AI 难度 | 简单 / 正常 / 困难（MCTS 预算 300 / 1500 / 4000） |
| 落子 | 点击空格落子，AI 随后回应（AI 落子蓝色高亮） |
| 消失提示 | 落子被抹去时：格子红色脉冲闪烁 + 状态栏提示 |
| 再来一局 | 终局后点击重新开局 |

## 难度与耗时（9×9 棋盘，实测参考）

| 难度 | MCTS 预算 | 每步耗时 |
|------|-----------|----------|
| 简单 | 300 | ~0.5 秒 |
| 正常 | 1500 | ~2.4 秒 |
| 困难 | 4000 | ~6.5 秒 |

## API

与月亮棋应用同构：

- `POST /api/start` — `{playerColor, difficulty}` → session（`board` 为 81 元素数组）
- `POST /api/move` — `{gameId, cellIndex}` → session（AI 回应 + vanish 结果）
- `POST /api/state` — `{gameId}` → session

session 字段：`board`、`turn`、`winner`、`over`、`round`、
`last_ai_move`、**`last_vanish`**（最近一手消失的格子索引，`null`=未消失）。

## 目录结构

```
layer4_interface/frontend/play_gomoku/
├── server.py            # HTTP 入口
├── session.py           # 对局会话（含 chance/vanish 处理）
└── static/              # index.html + css/gomoku.css + js/gomoku.js
```
