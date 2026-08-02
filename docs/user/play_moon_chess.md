# 月亮棋 · 人机对弈（使用说明）

人类与 AI 在浏览器里下月亮棋。AI 使用 MCTS（蒙特卡洛树搜索），通过
`layer3_solvers` 与 `layer2_engine` 的引擎交互，无需截图或 VLM——纯交互对弈。

## 快速开始

```bash
python -m layer4_interface.frontend.play_moon_chess.server
```

启动后浏览器打开 **http://127.0.0.1:8765/** 即可开始对局。

自定义端口：

```bash
python -m layer4_interface.frontend.play_moon_chess.server --host 0.0.0.0 --port 9000
```

## 游戏规则（月亮棋）

- **棋盘**：3×3，共 9 格
- **棋子**：每方最多 **3 子**，落第 4 子时**最老的棋子被驱逐**（FIFO）
- **获胜**：任意一方**三子连成一线**（横/竖/斜）即胜
- **平局**：50 手内无人三连，判平局
- 先手为黑（●），后手为白（○）

## 界面操作

| 操作 | 说明 |
|------|------|
| 执子选择 | 黑棋先手 / 白棋后手 / 随机。选白棋时 AI 先落一子 |
| AI 难度 | 简单 / 正常 / 困难（对应 MCTS 预算 300 / 1500 / 5000） |
| 开始对局 | 进入对局，轮到你可落子时格子可点击 |
| 落子 | 点击空格即落子，随后 AI 思考并回应（AI 落子带蓝色高亮） |
| 棋子序号 | 棋子上的数字 1=最新、3=最老，直观展示 FIFO 驱逐顺序 |
| 再来一局 | 终局后点击，回到设置页重新开局 |

对局进行中 AI 思考时棋盘会禁用（不可点击）。

## 难度与耗时

3×3 棋盘较小，AI 每步耗时大致为：

| 难度 | MCTS 预算 | 每步耗时（参考） |
|------|-----------|------------------|
| 简单 | 200 | ~0.2 秒 |
| 正常 | 800 | ~0.5 秒 |
| 困难 | 2000 | ~1.3 秒 |

## API 参考

接口均为 `POST` + JSON，地址前缀 `http://127.0.0.1:8765`。

### `POST /api/start` — 新建对局

请求：

```json
{ "playerColor": "p_black", "difficulty": "normal" }
```

`playerColor`：`p_black` / `p_white` / `random`；`difficulty`：`easy` / `normal` / `hard`。

响应（`session` 字段）：

```json
{
  "ok": true,
  "session": {
    "game_id": "c880f654",
    "player_color": "p_black",
    "difficulty": "normal",
    "board": ["p_black", null, null, "p_white", null, null, null, null, null],
    "round_age": { "0": 1, "3": 1 },
    "turn": "p_black",
    "winner": null,
    "over": false,
    "last_ai_move": null,
    "round": 1
  }
}
```

字段说明：

| 字段 | 说明 |
|------|------|
| `board` | 9 元素数组，`null`=空，`p_black`/`p_white`=棋子 |
| `round_age` | 棋子序号：格子索引 → 年龄（1=最新，3=最老） |
| `turn` | 当前轮到谁（`null` = 已终局） |
| `over` | 是否终局 |
| `last_ai_move` | AI 最近一次落子的格子索引 |
| `round` | 当前手数 |

### `POST /api/move` — 人类落子

请求：

```json
{ "gameId": "c880f654", "cellIndex": 4 }
```

服务端先应用人类落子，再让 AI 回应，响应即最新 `session`（结构同上）。
`cellIndex` 为 0–8 的格子索引。

错误（HTTP 400）：对局不存在、已终局、非自己回合、格子上已有棋子。

### `POST /api/state` — 查询局面

请求：`{ "gameId": "c880f654" }` → 返回当前 `session`（刷新页面后可恢复对局）。

## 常见问题

**端口被占用？** 换端口启动：`--port 9000`，浏览器访问对应端口。

**对局中断了（浏览器关闭/刷新）？** 服务端会话仍在内存中，刷新后用
`POST /api/state` + `game_id` 可恢复；服务重启后会话丢失，需重新开局。

**AI 每步要等多久？** 见上方难度表；难度越高思考越久，普通对局建议"正常"。

**想换个执子/难度？** 终局后点"再来一局"，或直接刷新页面重新设置。

## 目录结构

```
layer4_interface/frontend/play_moon_chess/
├── server.py            # HTTP 入口（/api/start、/api/move、/api/state）
├── session.py           # 对局会话管理（game_id → 引擎 + MCTS）
└── static/              # 前端资源
    ├── index.html
    ├── css/board.css
    └── js/game.js
```

其他前端应用（如视觉识别 `vision`，端口 8766）相互独立，互不依赖，
公共后端工具位于 `layer4_interface/frontend/common/`。
