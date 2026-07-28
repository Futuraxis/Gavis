# 月亮棋 Binding Layer 与 PPO Demo

这个仓库只实现当前分工内的两个模块：

- Binding Layer：把外部画面或 mock 输入转换为 `Observation`
- PPO：读取统一 `GameState`，输出合法 `GameAction`

没有实现输入层、DSL、CFR、PSRO，也没有试图替代团队后续的完整游戏引擎。

## 当前目录结构

- `binding/`
  负责 `Observation`、`MockBinding`、`ImageBinding`、`VisionLLMBinding`、`CellClassifier` 接口、`StateTracker`
- `encoding/`
  负责 `GameStateAdapter` 和 `MoonStateEncoder`
- `algorithms/`
  负责 `ActorCriticNetwork`、`RolloutBuffer`、`PPOAgent`
- `training/`
  提供最小训练与评估脚本
- `tests/mock_moon_env.py`
  仅用于本模块联调的最小环境，不是正式游戏模块

## 环境

- Python 3.11+
- PyTorch
- OpenCV
- pytest
- numpy
- pydantic v2

建议先安装依赖：

```bash
python3 -m pip install pytest numpy pydantic torch opencv-python
```

## ImageBinding 限制

`ImageBinding` 第一版只支持：

- 已经裁剪好的正方形棋盘截图
- 固定 3x3 布局
- 棋盘背景和棋子样式相对稳定
- 当前默认分类器只适合演示用途，不保证迁移到其他游戏界面仍然有效

图片要求：

- 建议至少 `90x90`
- 每个格子尽量完整可见
- 背景不要过于复杂
- 棋子边缘需要较清晰

后续如果团队接入更稳健的视觉模型，可以直接替换 `CellClassifier`

## 纯视觉大模型路线

如果你想把“整张前端页面截图”直接交给某个视觉大模型识别，可以使用 `VisionLLMBinding`：

```python
from binding import VisionLLMBinding


class MyVisionClient:
    def infer_observation(self, *, image_bytes: bytes, mime_type: str, prompt: str):
        # 这里接你的视觉模型 API
        # 返回 dict 或 JSON 字符串都可以
        return {
            "boardObservation": [
                ["X", None, "O"],
                [None, "X", None],
                ["O", None, None],
            ],
            "confidence": [
                [0.98, 0.93, 0.96],
                [0.91, 0.97, 0.95],
                [0.96, 0.94, 0.92],
            ],
        }


binding = VisionLLMBinding(client=MyVisionClient())
observation = binding.parse_image("screenshots/moon_page.png", frame_seq=0)
print(observation.model_dump())
```

这个版本是纯视觉路线：

- 输入可以是整张页面截图，不要求你先手动裁棋盘
- Prompt 会明确要求模型只识别中间 3x3 棋盘
- 最终统一返回 `Observation`

当前限制：

- 单张截图通常仍然不能可靠恢复完整 `pieceOrder`
- 顺序信息仍建议交给 `StateTracker` 基于连续帧推断
- 真正接入哪个大模型，由你的 `client.infer_observation(...)` 决定

## 运行测试

```bash
pytest
```

## 最小训练

```bash
python3 -m training.train_ppo --episodes 20 --save-path artifacts/ppo_agent.pt
python3 -m training.evaluate_ppo --model-path artifacts/ppo_agent.pt --episodes 10
```

## 编码说明

`MoonStateEncoder.FEATURE_DIM == 38`

- 27 维：9 个格子的占用 one-hot
- 9 维：棋子年龄编码
- 1 维：当前是否轮到视角玩家
- 1 维：归一化步数

这样可以保证：

- 同一棋盘占用但不同 `pieceOrder` 会得到不同编码
- PPO 可以通过 `action_mask` 避免选择非法动作

# key
cd /Users/anon/Desktop/月亮棋
source .venv/bin/activate
export DASHSCOPE_API_KEY="sk-ws-H.EHLELHL.Aicy.MEQCIBdZ9S9ceFSo9nucEIPoYoFoe5smI_T54bITGBbw3qcvAiAT5xjY5mhlAeYmfR79s0rTPchd216781i19bc6Oab2Jg"
export QWEN_BASE_URL="https://llm-celowj0p67i3zz4v.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
export QWEN_MODEL="qwen3-vl-plus"
export QWEN_SKIP_SSL_VERIFY=1
python app_server.py