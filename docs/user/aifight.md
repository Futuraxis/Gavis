# AIFight 接入

AIFight Bridge 调用的是模型 API。Gavis 的 Layer 3/Layer 4 solver 本身不是 LLM API，所以项目提供一个本地 OpenAI-compatible 包装服务，让 AIFight 把 Gavis 当作一个兼容模型调用。

## 启动本地 API

```bash
export GAVIS_AIFIGHT_TOKEN='gavis-local-aifight-20260830'

.venv-2/bin/python -m layer4_interface.aifight.openai_compat \
  --host 127.0.0.1 \
  --port 8789 \
  --token "$GAVIS_AIFIGHT_TOKEN" \
  --model gavis-local
```

API base URL：

```text
http://127.0.0.1:8789/v1
```

API key：

```text
gavis-local-aifight-20260830
```

`gavis-local-aifight-20260830` 是本地服务校验用的 bearer token，只要启动服务和 AIFight 配置里一致即可。

## 配置 AIFight Bridge

在 AIFight Bridge 里新增一个 OpenAI-compatible/compat provider：

```text
base_url = http://127.0.0.1:8789/v1
model = gavis-local
api_key = gavis-local-aifight-20260830
```

如果使用 AIFight CLI，可按它的 `compat`/OpenAI-compatible provider 配置，把 base URL、model、API key 填成上面的值。

这些值对应关系如下：

| AIFight 要填的字段 | 当前项目里填什么 | 从哪里来 |
| --- | --- | --- |
| `base_url` | `http://127.0.0.1:8789/v1` | 本机 Gavis API 地址 |
| `model` | `gavis-local` | 启动服务时的 `--model`，可自定义 |
| `api_key` | `gavis-local-aifight-20260830` | 启动服务时的 `--token`，两边一致即可 |
| `profile` | `gavis` | AIFight 本地配置名，自己命名 |

例如：

```bash
export GAVIS_AIFIGHT_TOKEN='gavis-local-aifight-20260830'

.venv-2/bin/python -m layer4_interface.aifight.openai_compat \
  --host 127.0.0.1 \
  --port 8789 \
  --token "$GAVIS_AIFIGHT_TOKEN" \
  --model gavis-local
```

另开一个终端配置 AIFight：

```bash
aifight config add gavis \
  --protocol compat \
  --base-url http://127.0.0.1:8789/v1 \
  --model gavis-local \
  --env GAVIS_AIFIGHT_TOKEN

aifight config test --profile gavis
```

如果还没有安装或连接 AIFight：

```bash
npm install -g @aifight/aifight
aifight setup
```

如果你已经在 AIFight 网页创建了 agent，它会给一个 pairing code 或连接命令，按它显示的命令运行：

```bash
aifight connect <PAIRING_CODE>
```

## 决策格式

这个包装 API 会读取 chat prompt 里的 JSON：

- 如果 JSON 是 Botzone/Gavis envelope，例如 `{"requests":[...],"responses":[...]}`，会直接走 `layer4_interface.botzone.runner.decide()`，再调用对应的 Layer 4 和 Layer 3。
- 如果 JSON 里只有 `legal_actions`/`legalActions`/`actions`，会从合法动作里选择一个兜底动作，保证返回值可解析。

当前更完整的游戏适配仍在 Botzone 协议侧：

- 国标/国际麻将：Botzone 字符串协议 -> Layer 4 Mahjong 适配 -> Layer 3 Mahjong solver。
- 双人德州扑克：Botzone JSON 协议 -> Layer 4 Texas 适配 -> Layer 3 MCTS。
