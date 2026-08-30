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