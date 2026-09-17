# LLM 服务配置（端点与模型）

Gavis 的 LLM 访问统一走 `layer2_engine/core/llm.py` 的 OpenAI 兼容客户端
（请求路径 `{base_url}/v1/chat/completions`）。端点与模型的解析优先级：

```
显式代码参数（LLMClient(model=..., base_url=...)） > 平台持久化配置 > 环境变量 > 内置默认
内置默认: base_url = http://127.0.0.1:11434（本地 Ollama）, model = qwen3:8b
```

## 方式一：环境变量（无需启动平台页面）

启动任何使用 LLM 的进程前设置：

```bash
export LLM_BASE_URL="https://api.deepseek.com"   # 或本地 vLLM / Ollama
export LLM_MODEL="deepseek-chat"                 # 或 qwen3:8b / 其他模型名
export LLM_API_KEY="sk-..."                      # 云端必填；本地 Ollama 可留空
```

命中范围：平台聊天 / Agent 对话 / 规则翻译（L1 默认客户端）/ 社交类求解器
（狼人杀、谁是卧底的 AI 座位，经 `train-cli/games.py` 注册表）以及其它直接
构造 `LLMClient()` 的调用点。显式传参的调用点仍以显式参数优先。

### 规则翻译的两个调优旋钮（创建自定义游戏）

规则翻译是**长输出**任务，与聊天尺度不同，单独有两个环境变量：

```bash
export LLM_MAX_TOKENS="32768"   # 输出预算（默认 32768；夹在 1024..262144）
export LLM_TIMEOUT_S="300"      # 传输超时秒数（默认 300；夹在 5..3600）
```

- **为什么预算要这么大**：推理模型（如 deepseek 系列带思维链的型号）的
  `reasoning_tokens` **也计入** `max_tokens`。实测预算 8192 时整个预算被思维链
  吃光、正文为空（`finish_reason=length`），表现为「LLM 未返回内容」→ 创建
  游戏失败。客户端现在会把这种失败明说成「输出预算被思维链/长度截断耗尽，
  请提高 LLM_MAX_TOKENS」，不再让你猜。
- **模型上限更小时自动降档**：端点若以 HTTP 400 拒绝该预算（老模型上限常为
  8192），统一客户端会自动用 8192 重试一次，不需要你手改配置。
- **超时**：聊天默认 30s 结束一次请求，但一份完整 rules.json 生成常需 1-3
  分钟；翻译路径单独使用 `LLM_TIMEOUT_S`（默认 300s）。

## 方式二：平台配置页面（运行时改，持久化）

8770 平台 → 侧边栏「LLM 配置」（`/llm`）→ 填端点 / 模型 / 密钥 →

- **保存配置**：写入 `data/llm_config.json`（原子写），立即生效——聊天、翻译、
  社交 AI 与 Agent 对话同步切换；聊天客户端缓存自动失效并重建。
- **测试连接**：保存前先探测 `{端点}/v1/models`（可带预览密钥）。
- **恢复默认**：清空平台配置，回退到环境变量 / 内置默认（还原启动时的进程环境）。

平台配置 > 环境变量：页面保存的值会覆盖进程内对应环境变量；清空后还原。

## 视觉识别（独立配置，不属本页范围）

`layer4_interface/binding/qwen_vision.py` 走 DashScope，另用环境变量：

```bash
export DASHSCOPE_API_KEY="sk-..."
export QWEN_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
export QWEN_MODEL="qwen-vl-plus"
```

## 行为说明

- 失败兜底：端点不可达 / API 错误时客户端返回空串（fail-soft），调用方走
  模板 / 随机兜底，平台不崩溃；失败原因记录在 `LLMClient.last_error` 并打
  warning 日志。需要「必须成功」的调用可设 `fail_hard=True` 抛
  `LLMClientError`。
- 密钥仅写不回显：`GET /api/llm/config` 只返回 `has_api_key`；页面密码框
  留空 = 保持不变，填空串（配合保存）= 清除。

## 从零翻译的提示词是「接地」的

L1 从零翻译的系统提示里带**方言速查 + 一份完整可运行的参考规则**
（`rules/stochastic_gomoku.json`，经 `layer1_translator/prompt_builder.py`
注入）。没有这份锚点时模型会自创方言（`effects`/`phases` 顶层键、
`hasFour(...)` 之类不存在的函数、字符串表达式当布尔用），产物永远过不了
`engine_validator`（schema + L2 冒烟），用户等几分钟只拿到「规则校验未通过」。
换成接地提示后同一模型能稳定产出通过校验、且能在平台上真正对弈的规则。