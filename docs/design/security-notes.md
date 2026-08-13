# 安全与性能决策记录（审计 3.6）

本文档记录 2026-08-13 架构审计 3.6 节（安全与性能隐患）各项的处理决策与
状态，作为后续维护的参照。修复项以代码注释 `（审计 3.6）` 标记。

## 已修复

| 条目 | 等级 | 修复内容 |
|------|------|----------|
| 路径遍历 | Critical | `platform/history.py` 对 `match_id` 做白名单校验（`[A-Za-z0-9_-]{1,64}`），`record/get/delete` 统一拦截 |
| 请求体无限制 | Major | `common/http_utils.py` `read_json_body` 限制 `MAX_BODY_BYTES=10MB`，超出抛 `BodyTooLargeError`；六个 server 统一响应 413 |
| Prompt 注入 | Major | LLM 输出发言统一清洗：长度上限（200）+ 剔除控制字符（`ollama_solver._sanitize_speech`、`social/llm_policy._complete`） |
| 线程不安全 | Major | `state_tracker` / `image_binding` / `mock_binding` / `feedback_collector` 的共享状态加 `threading.Lock` |
| 资源泄漏 | Minor | `benchmark.BenchmarkRunner` 增加 `MAX_JOBS=500` 上限与完成态清理 |
| 硬编码 API key | Minor | 统一读取流程：`layer2_engine/interfaces/api_key.py` `resolve_api_key`（显式参数 > 环境变量 > 默认），接入 `social/llm_policy` 与 `binding/qwen_vision`；key 为空时不发送 Authorization 头 |
| infoset key 性能 | Major | `engine.get_info_set_key` 改为紧凑序列化（无 sort_keys）+ sha256 哈希，key 恒 64 字符 |
| 无并发环境（PSRO） | Minor | `gamescape`/`exploitability` 用 `ThreadPoolExecutor` 并行评估对局（`GymAdapter.clone()` 提供每线程独立环境，`PSROConfig.num_workers` 控制，1=串行） |

**注意**：infoset key 格式已变更——按旧格式（全量 JSON）保存的 CFR 策略表
（Hybrid 的 `cfr_table_path` JSON 文件）不再兼容，需重新训练生成。

## 暂不处理（决策记录）

| 条目 | 等级 | 决策 | 理由与前置条件 |
|------|------|------|----------------|
| SSL 禁用（`QWEN_SKIP_SSL_VERIFY`） | Critical | 保留，仅本地开发用 | 本地自签证书/代理环境需要；代码注释明确生产必须走证书校验（`QWEN_CA_BUNDLE`）。对外开放前移除 |
| CORS 通配 + 无认证 | Major | 保留 | 平台服务定位为本机开发工具（默认绑定 127.0.0.1）。对外网/局域网暴露前：收紧 CORS 到同源 + 引入鉴权（token 或 session），属 P2 平台工程化 |
| 阻塞 I/O（LLM 调用） | Major | 保留 | 本地单人演示可接受；对外服务前需线程池/任务队列 + 超时重试（P2） |

## 未排期（P2 候选）

- RL 求解器多环境并行（PPO/MARL 的 SubprocVecEnv 式采集）——本轮只做了
  PSRO 元博弈评估并行化。
- 平台服务工程化：认证/鉴权、请求限流、job 队列、持久化存储。
