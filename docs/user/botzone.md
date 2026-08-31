# Botzone 接入

Botzone 有两种适合 Gavis 的接入方式：

1. 本地 AI 长轮询：Botzone 给一个包含用户 ID 和密钥的 `/localai` 地址，本机进程主动去拉取请求，并在下一次请求头里提交回复。推荐先用这个。
2. 上传 zip 薄客户端：Botzone 运行 Python 3.6 兼容入口，入口再转发到本机或服务器的 Gavis 接口。

## 本地 AI 长轮询

在 Botzone 页面里创建或进入游戏桌，选择“用本地 AI 代替我”，复制平台给出的 Local AI URL，格式类似：

```text
https://www.botzone.org/api/<用户ID>/<密钥>/localai
```

然后在完整 Gavis 项目环境里运行：

```bash
export BOTZONE_LOCALAI_URL='https://www.botzone.org/api/<用户ID>/<密钥>/localai'
python -m layer4_interface.botzone.localai
```

这个客户端会：

- `GET /localai` 拉取 Botzone 给出的 match 请求。
- 用项目内 `layer4_interface.botzone.runner.decide()` 识别麻将字符串协议或德州 JSON 协议。
- 调用 Layer 4 适配和 Layer 3 solver。
- 在下一次 `GET /localai` 时通过 `X-Match-<matchid>` 请求头提交动作。

也可以让客户端调用 Botzone 的 `/runmatch` 创建测试桌：

```bash
python -m layer4_interface.botzone.localai \
  --create-game TexasHoldem \
  --player me \
  --player <另一个BotID>
```

`--player` 中必须且只能有一个 `me`，对应 Botzone 文档里的本地 AI 玩家位置。

## 上传 zip 薄客户端

Botzone 的 Python 运行环境可能是 3.6，不能直接上传完整 Gavis 项目。薄客户端方式的流程是：

1. 本机或服务器运行完整 Gavis 项目，提供决策接口。
2. Botzone 上传一个 Python 3.6 兼容的 zip 薄客户端。
3. 薄客户端把 Botzone 的 `requests` 转发到 Gavis 接口，接口再调用 Layer 4 适配和 Layer 3 solver。
4. 网络失败时，薄客户端会返回合法兜底动作，避免 Botzone 因崩溃/超时判负。

## 启动完整项目接口

```bash
python -m layer4_interface.botzone.server \
  --host 0.0.0.0 \
  --port 8788 \
  --token YOUR_TOKEN
```

接口路径：

```text
POST /botzone/decide
Authorization: Bearer YOUR_TOKEN
```

`YOUR_TOKEN` 是你自己设置并写入上传包的密钥。Botzone 若提供平台侧密钥，可以把同一个值填到 `--token` 和构建参数 `--remote-token` 中。

## 构建上传包

```bash
python scripts/build_botzone_zip.py \
  --remote-url https://YOUR_DOMAIN/botzone/decide \
  --remote-token YOUR_TOKEN \
  --remote-timeout 0.75
```

上传文件：

```text
dist/gavis_botzone.zip
```

## 当前支持

- 国标/国际麻将：Botzone 字符串协议 -> `mahjong_international` -> `MahjongHeuristicAI`。
- 双人德州扑克：Botzone JSON 协议 -> `texas_holdem` -> Layer 3 Hybrid 决策
  （不完全信息搜索，`BOTZONE_LAYER3_BUDGET=35` 覆盖搜索预算；六人德州走
  L4 保守启发式，见下文）。
- 六人德州扑克：不作为完整项目接入目标；上传端仅保留合法兜底，防止误传时崩溃。
