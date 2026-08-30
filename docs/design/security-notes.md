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
| CORS 通配（2026-09 体验审计 B6 修复） | Major | 平台服务移除全部 ACA* 响应头：默认同源（Python 托管 dist，页面与 `/api` 同源；dev 模式经 Vite 代理也不跨域；`--host 0.0.0.0` LAN 场景仍同源）。测试锚：`test_platform_server.py::test_no_cors_wildcard_headers`。对外网暴露前仍需鉴权（见未排期） |
| 在线学习捕获无法关闭（体验审计 B5 修复） | Major | `PlayManager.start` 以 `LearningManager.enabled(game_id)` 门控轨迹捕获——UI 关闭后新会话不再落盘（旧实现只在 apply 阶段检查，捕获照常写入）。测试锚：`test_online_learning.py::test_disabled_game_means_no_capture` |
| 轨迹落盘 god-view 全量状态（体验审计 B7-lite 修复） | Major | `TrajectoryRecorder._decision` 改存 `engine.project_observation(state, player)`（决策者自己的信息集投影），不再明文落盘含对手底牌的 `_arrays` 全量。测试锚：`test_online_learning.py::test_texas_holdem_captures_info_keys_and_ai_loop` |
| 轨迹无限增长（体验审计 B8 修复） | Minor | `LearningManager.apply` 发布成功后调用 `store.trim(game_id)`（默认保留最近 500 局）——设计承诺落地 |

**注意**：infoset key 格式已变更——按旧格式（全量 JSON）保存的 CFR 策略表
（Hybrid 的 `cfr_table_path` JSON 文件）不再兼容，需重新训练生成。

## 暂不处理（决策记录）

| 条目 | 等级 | 决策 | 理由与前置条件 |
|------|------|------|----------------|
| SSL 禁用（`QWEN_SKIP_SSL_VERIFY`） | Critical | 保留，仅本地开发用 | 本地自签证书/代理环境需要；代码注释明确生产必须走证书校验（`QWEN_CA_BUNDLE`）。对外开放前移除 |
| 阻塞 I/O（LLM 调用） | Major | 保留 | 本地单人演示可接受；对外服务前需线程池/任务队列 + 超时重试（P2） |

## 既定偏差登记（2026-08-22 第二轮审查）

| 条目 | 等级 | 位置 | 决策与前置条件 |
|------|------|------|----------------|
| `eval()`/`exec()` 直接使用（与 coding-standards §2.4 明文禁止冲突） | Major | `core/expr_eval.py:1084,1087`（`_eval_arithmetic` 算术串求值）；`core/state_graph.py:116`（`_eval_length_expr`）；`core/rules_compiler.py:479`（`_safe_eval`，noqa S307）、`:808`（`exec(compile(...))` codegen，noqa S102） | **保留**。缓解：输入为受信静态规则 JSON（`rules/` 顶层 + 生成的 v5.1 规则）；`eval` 均剥离 `__builtins__`；codegen 产物经 probe 验证兜底。**前置条件（已落地，本轮）**：Layer 1 纳入平台工作流后，自定义/变体规则不再属于「受信静态规则」——约束已生效：(a) Layer 1 产出强制 `schema_validator` + L2 冒烟校验（`layer1_translator/engine_validator.py`，schema 校验模块 `layer1_translator/schema_validator.py`）；(b) `GameEngine(rules, allow_codegen=False)` 开关已实现（`layer2_engine/core/engine.py`），平台自定义游戏族构造引擎一律 `allow_codegen=False` 强制纯解释器路径（`layer4_interface/frontend/platform/families/helpers.py::engine_from_rules_dict` 默认即关闭；解释器可处理全部 v5.1 表达式，`test_expr_eval.py::_assert_consistent` 已系统验证双路径一致）。长期方向：算术串求值替换为自研 tokenizer/求值器或 `ast` 节点白名单求值，彻底消除 `eval` |

## 未排期（P2 候选）

- RL 求解器多环境并行（PPO/MARL 的 SubprocVecEnv 式采集）——本轮只做了
  PSRO 元博弈评估并行化。
- 平台服务工程化：认证/鉴权、请求限流、job 队列、持久化存储。
