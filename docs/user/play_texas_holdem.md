# 德州扑克 · 人机对弈（使用说明）

人类与 AI 在浏览器里打德州扑克（单挑无限注）。AI 使用 **HybridSolver**
（`layer3_solvers/hybrid/`）的对手建模搜索：在采样世界（`sample_hidden`
补全对手底牌）上进行 PIMC 式树搜索，对手节点按对手模型（默认均匀）采样，
不在树中泄漏对手底牌；与 `layer2_engine` 的 v5.0 规则引擎交互
（`rules/texas_holdem.json`），发牌为引擎 chance 节点，
底池结算与牌型判定由引擎内置函数完成。

## 快速开始

```bash
python -m layer4_interface.frontend.play_texas_holdem.server
```

启动后浏览器打开 **http://127.0.0.1:8768/** 即可开始对局。

自定义端口：

```bash
python -m layer4_interface.frontend.play_texas_holdem.server --host 0.0.0.0 --port 9000
```

## 游戏规则（单挑无限注德州扑克）

- **盲注**：小盲 1，大盲 2，初始筹码各 **100**
- **行动**：弃牌 / 跟注（下注为 0 时即过牌）/ 加注（可全下）
- **加注下限**：标准最小加注规则（上一个加注幅度，翻前为大盲）
- **发牌**：各 2 张底牌 → 翻牌 3 张 → 转牌 → 河牌 → 摊牌
- **摊牌**：7 张选 5 定牌型（同花顺 / 四条 / 葫芦 / 同花 / 顺子 /
  三条 / 两对 / 一对 / 高牌），平分底池时按牌型平摊
- **全下**：筹码不足时按可跟注额度结算，超额部分退回（heads-up 规则）

## 界面操作

| 操作 | 说明 |
|------|------|
| 座位选择 | 小盲位 / 大盲位 / 随机。选大盲位时 AI（小盲）先行动 |
| AI 难度 | 简单 / 正常 / 困难（对应 Hybrid 搜索预算 150 / 500 / 1200） |
| 弃牌 | 放弃本局，底池归对方 |
| 过牌 / 跟注 | 无需跟注时显示「过牌」，需要时显示「跟注 N」 |
| 加注 | 选择目标数额（下拉），点击「加注」；「全下」一键推 100 |
| 摊牌 | 终局显示双方底牌与牌型、胜负与筹码盈亏 |

对局中 AI 思考时操作按钮会禁用。

## 终端演示

```bash
python train-cli/train.py --game texas_holdem --solver hybrid --skip-eval  # 训练 Hybrid（CFR 先验 + 不完全信息搜索）
```

## API（与其他对弈应用一致）

| 接口 | 请求 | 说明 |
|------|------|------|
| `POST /api/start` | `{playerColor, difficulty}` | 开局（自动发牌，AI 先行动则先走） |
| `POST /api/move` | `{gameId, choice, amount}` | 人类行动（弃牌/跟注/加注），AI 随即回应 |
| `POST /api/state` | `{gameId}` | 查询当前局面快照 |
