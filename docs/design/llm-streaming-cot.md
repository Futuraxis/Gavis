# LLM 流式 API 与思维链（CoT）支持 — 设计记录

> 2026（流式/思维链改造）。在统一客户端
> `layer2_engine/core/llm.py::LLMClient` 上新增 SSE 流式传输与思维链提取，
> 平台聊天界面（`/api/chat` → ChatPage）改为流式渲染并可折叠展示思维链。

## 背景与目标

- 统一客户端此前是整块响应：`"stream": False` 硬编码，只读
  `choices[0].message.content`，不提取 `reasoning_content`，不剥离
  ` think…/think` 标签。所有 LLM 回合（对话、陪玩成文、求解决策、规则
  翻译）都是「整块等待」。
- 目标：
  1. `LLMClient` 提供流式方法（SSE 增量迭代），失败语义与既有 fail-soft
     完全一致；
  2. 流式与非流式都能提取思维链（`reasoning_content` / `reasoning` 字段、
     Ollama-legacy ` think…/think` 内容夹层）；
  3. 平台 `/api/chat` 支持 SSE 模式（增量正文 + 思维链增量），前端逐字
     渲染并将思维链以折叠块展示、存档持久化；
  4. 向后兼容：不带 `Accept: text/event-stream` 的请求保持原 JSON 信封，
     旧 `dist` 与既有测试不受影响。

## 传输层（layer2_engine/core/llm.py）

### 新增公开面

- `ChatReply.reasoning: str = ""` — 非流式回复的思维链（默认空，零破坏）。
- `StreamChunk` dataclass：
  - `text` / `reasoning` — 增量（拼接得全量）；
  - `done` — 流结束标志；
  - `tool_calls` — 终态块上的完整解析结果（`finish_reason: "tool_calls"`）；
  - `error` — 传输失败原因（fail-soft：`done=True` + `error` 非空）。
- `complete_stream(messages, *, tools=None, max_tokens=None, temperature=None)
  -> Iterator[StreamChunk]` — SSE 逐行解析：
  - payload `"stream": True`；
  - 逐 `readline()` 只处理 `data:` 行（跳过注释/空行/非 data 行）；
  - 三种终态：`[DONE]`、`choices[0].finish_reason ∈ {stop, tool_calls}`、
    Ollama native `done: true`；
  - delta 取 `choices[0].delta`（OpenAI 系）或 `message`（兼容形态）；
  - reasoning 双键防御：`reasoning_content` 优先、`reasoning` 兜底；
  - 工具调用按 `index` 累积 `function.name/arguments` 分片，流结束统一
    解析成 `ToolCall`（兼容「单块完整 tool_calls」与「逐帧分片」两种
    端点行为）；
  - 失败分类复用 `_record_failure`：HTTP 4xx/5xx、不可达/超时、累计超
    `_MAX_RESPONSE_BYTES`、畸形 `data:` JSON → 产出错误终态块
    （fail-soft）或抛 `LLMClientError`（`fail_hard`）；迭代永不悬挂。
- `complete_chat_reply(system, user, max_tokens=None) -> ChatReply` —
  供成文侧（DialogueEngine）透传 reasoning。

### think 标签剥离（_ThinkTagSplitter）

- 常量 `_THINK_OPEN = "<think>"` / `_THINK_CLOSE = "</think>"`（Ollama /
  llama.cpp legacy 形态：`--reasoning-format think` 的内容夹层）；严格
  配对才剥离，普通文本零处理；尾部挂起前缀覆盖 `<think`/`</think` 的
  所有真前缀（`<`、`<t`、…、`</think`），跨 chunk 切标签不丢字。
- 状态机支持跨 chunk 分片：块尾是不完整标签前缀（`<`, `<t`, … `<think`,
  `</`, …, `</think`）时挂起等待补全；`flush()` 对未闭合标签按字面文本
  处理。
- 流式与非流式共用同一 splitter：`delta.reasoning_content` 直取，
  `content` 里的夹层切到 reasoning 通道。

### 上限与安全

- `_MAX_RESPONSE_BYTES`（8 MiB）对整条流累计生效；
- 每段增量过 `sanitize_text`（控制字符清洗，统一 prompt 注入防护）。

## 编排层（layer4_interface/frontend/platform/chat.py）

- `_prepare(...)` / `_assemble_llm_prompt(...)` — `chat_turn` 与
  `chat_turn_stream` 共享前置（清洗文本、定位 session、收集目录/活跃
  会话/最近对局、拼 tools+messages），两出口上下文一致。
- `chat_turn_stream(...) -> Iterator[dict]` — SSE 事件生成器：
  - 事件契约：

    | event     | data                         | 说明 |
    |-----------|------------------------------|------|
    | `reasoning` | `{"delta": str}`           | 思维链增量（上限 `_REASONING_MAX_CHARS = 8000`）|
    | `text`     | `{"delta": str}`            | 回复正文增量 |
    | `intent`   | `ChatTurnResult`            | 最终意图（text 为全量回复）|
    | `error`    | `{"error": str}`            | 流中失败（随后仍发兜底 intent）|
    | `done`     | `{}`                        | 结束 |

  - 循环骨架与 `chat_turn` 的 `_MAX_TOOL_ROUNDS` 有界工具循环一致：
    信息工具就地执行、`role:"tool"` 回传；动作工具/携带 intent 与
    JSON 模式同语义（模型亲笔增量优先作最终文案）；
  - `llm=None`、空文本、流失败、预算耗尽 → 与 JSON 模式相同的兜底
    收口（正则 intent / last_tool_result），保证前端总能拿到 `intent`。

## 服务端（server.py / common/http_utils.py）

- `http_utils`：`start_sse(handler)`（`text/event-stream` + 禁止缓存头、
  `Connection: close`，stdlib HTTP/1.0 连发即断）+ `send_sse_event(...)`
  逐事件 flush。
- `_wants_stream(handler)` — 协商：`Accept` 含 `text/event-stream` 或
  query `stream=1` 才走流式；否则原 JSON 信封（向后兼容）。
- `_handle_chat_stream(...)` 遍历 `chat_turn_stream` 逐帧写出；异常兜
  `error` + `done`，连接绝不悬挂。

## 应用端（platform-frontend）

- `src/chat/sse.ts` — 纯 SSE 帧解析（跨块缓冲、注释行容错、`finish()`
  收尾），无运行时依赖，可被 node 测试直接覆盖。
- `src/api/client.ts::chatTurnStream(text, gameId?, history?, {onText,
  onReasoning}) -> Promise<ChatTurnResult>` — 带 `Accept:
  text/event-stream` POST `/api/chat?stream=1`；`resp.body.getReader()`
  增量解码；`reasoning`/`text` 事件实时回调；`intent` 事件 resolve；
  `error`/无 intent 的 `done` 以 `ApiError` reject。
- `src/chat/useChatRuntime.ts::send` — 先落一条 `pending` 草稿消息，
  增量原地更新（正文 + `reasoning`）；`chatTurnStream` resolve 后
  `dispatch(result)`，其内部 `pushAgent` 命中 `draftTargetRef` 把草稿
  原地定稿（流式正文优先，兜底文案后备）——不产生第二条消息、不丢
  亲笔文本。流中断：草稿已有内容则保留并附中断提示；完全无内容则移除
  草稿、走本地正则兜底（旧行为不变）。`persistable` 排除 `pending` 草稿，
  定稿后才入档（conversations + localStorage 镜像），避免中间态进存档。
- `types.ts::ChatMessage.reasoning?: string`；`MessageBubble` 在正文下
  渲染可折叠「🧠 思维链」`<details>`（默认收起，`max-height: 320px`
  滚动承载长文本）；`chat.css` 新增 `.chat-msg-reasoning` 弱化样式。
- 陪伴/教练通道（E）：`AgentMessage.reasoning` + `DialogueEngine` 走
  `complete_chat_reply` 透传；`session.pending_chat`（快照 `chat` 增量）
  与 `/api/agent/say` 响应携带 `reasoning`；`SnapshotChatEntry.reasoning?`
  与 `snapshotChatToMessages` 映射 → 同一折叠块渲染。该通道保持单次
  成文（不流式；流式化归属后续项）。

## 存档（layer4_interface/frontend/platform/conversations.py）

- 消息字段白名单新增可选 `reasoning`（剔控制字符 + 上限
  `_REASONING_MAX = 4000`，非字符串/超限丢弃，fail-soft）；
- 旧记录缺字段天然兼容（可选字段）。

## 边界与失败模式

- 流中途 HTTP 错误/超时/畸形行 → 错误终态块（fail-soft），前端对
  `error` 事件记录原因、以 `intent`（兜底）收口，草稿合并规则见上文。
- 工具分片两种端点形态 → 统一「流结束再解析」，解析失败按空参
  `ToolCall` 丢弃（与既有非流式语义一致）。
- reasoning 出现在 `message`（非流）与 `delta`（流）双包装 → 双键防御。
- `think` 严格配对；无标签模型零开销。
- 生成器中途 close（前端中断）→ GeneratorExit 经由 with 块正常释放连接。
- 旧 `dist` 未带 Accept 头 → JSON 模式，行为不变。

## 测试

- 后端 pytest：
  - `test_llm_client.py`：SSE 分片/三种终态、reasoning 双键、think 标签
    剥离（含跨 chunk）、工具分片与单块形态、流中失败 fail-soft /
    `fail_hard`、`_FakeStreamResponse.readline` 注入；
  - `test_chat.py`：`chat_turn_stream` 事件序列（纯文本/携带 intent/
    信息工具多轮/动作工具/无 LLM/预算耗尽/流失败兜底）；
  - `test_platform_server.py`：Accept 协商 → SSE 头与帧序列；无头 → JSON
    回归；
  - `test_conversations.py`：reasoning 持久化/超限/缺失兼容。
- 前端 node 测试：`chat.sse.test.ts`（解析器）、`chat.client-api.test.ts`
  （chatTurnStream 请求形态/增量回调/跨块分片/error/无 intent/非 2xx/
  断连）；`npm run test:frontend` 全绿。
- 类型/构建：`npm run build`（`tsc --noEmit` + vite build）。tsconfig
  增 `allowImportingTsExtensions`（配合 `noEmit`）以让
  `src/api/client.ts` 的运行时 import 带 `.ts` 扩展名——node
  `--experimental-strip-types` 的 ESM 解析要求显式扩展名，Vite 同样兼容。