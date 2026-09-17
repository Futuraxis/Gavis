// API 客户端 — 解包 {"ok": ...} 信封, 失败时抛出 ApiError

import { SseParser, type SseEvent } from '../chat/sse.ts'
import { buildHistoryQuery, type ResultKind } from '../history.ts'

const BASE = '/api'

export class ApiError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

interface Envelope {
  ok: boolean
  error?: string
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response
  try {
    resp = await fetch(BASE + path, init)
  } catch {
    throw new ApiError('无法连接服务器，请确认平台服务已启动 (python -m layer4_interface.frontend.platform.server)')
  }
  let data: T & Envelope
  try {
    data = (await resp.json()) as T & Envelope
  } catch {
    throw new ApiError(`服务器返回异常 (HTTP ${resp.status})`)
  }
  if (!data.ok) {
    throw new ApiError(data.error ?? `请求失败 (HTTP ${resp.status})`)
  }
  return data
}

export function apiGet<T>(path: string): Promise<T> {
  return request<T>(path)
}

export function apiPost<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

// ── Chat-first (agent 聊天模式) ─────────────────────────────────
// POST /api/chat — 一句话 → 意图+参数（LLM function calling + 正则兜底）。
// history: 之前若干轮 user/assistant 文本（最新的在后），让 LLM 有对话上下文。
export type ChatHistoryTurn = { role: 'user' | 'assistant'; content: string }

export function chatTurn(
  text: string,
  gameId?: string,
  history?: ChatHistoryTurn[],
): Promise<import('../types').ChatTurnResult> {
  return apiPost<import('../types').ChatTurnResult>('/chat', {
    text,
    ...(gameId ? { game_id: gameId } : {}),
    ...(history && history.length > 0 ? { history } : {}),
  })
}

// ── Chat-first 流式模式（SSE）───────────────────────────────────
// 同一 /api/chat 路由，带 Accept: text/event-stream（+ ?stream=1），
// 后端按 chat_turn_stream 事件契约发流：
//   reasoning{delta} / text{delta} / intent{ChatTurnResult} / error{error} / done{}
// 回调在事件到达时同步触发（onText/onReasoning 供前端逐字渲染与思维链展示）；
// 最终以 intent 事件 resolve；error 事件或流意外中断以 ApiError reject。
export interface ChatStreamHandlers {
  onText?: (delta: string) => void
  onReasoning?: (delta: string) => void
}

export function chatTurnStream(
  text: string,
  gameId?: string,
  history?: ChatHistoryTurn[],
  handlers: ChatStreamHandlers = {},
): Promise<import('../types').ChatTurnResult> {
  return new Promise<import('../types').ChatTurnResult>((resolve, reject) => {
    void (async () => {
      try {
        let resp: Response
        try {
          resp = await fetch(BASE + '/chat?stream=1', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Accept: 'text/event-stream',
            },
            body: JSON.stringify({
              text,
              ...(gameId ? { game_id: gameId } : {}),
              ...(history && history.length > 0 ? { history } : {}),
            }),
          })
        } catch {
          throw new ApiError(
            '无法连接服务器，请确认平台服务已启动 (python -m layer4_interface.frontend.platform.server)',
          )
        }
        if (!resp.ok || !resp.body) {
          let detail = `服务器返回异常 (HTTP ${resp.status})`
          const ct = resp.headers.get('Content-Type') ?? ''
          if (ct.includes('application/json')) {
            try {
              const data = (await resp.json()) as { error?: string }
              if (data?.error) detail = data.error
            } catch {
              /* 保持默认文案 */
            }
          }
          throw new ApiError(detail)
        }
        const reader = resp.body.getReader()
        const decoder = new TextDecoder('utf-8')
        const parser = new SseParser()
        let intent: import('../types').ChatTurnResult | null = null
        let streamError: string | null = null
        const handle = (ev: SseEvent): void => {
          if (ev.event === 'reasoning') {
            const d = JSON.parse(ev.data) as { delta?: string }
            handlers.onReasoning?.(d.delta ?? '')
          } else if (ev.event === 'text') {
            const d = JSON.parse(ev.data) as { delta?: string }
            handlers.onText?.(d.delta ?? '')
          } else if (ev.event === 'intent') {
            intent = JSON.parse(ev.data) as import('../types').ChatTurnResult
          } else if (ev.event === 'error') {
            const d = JSON.parse(ev.data) as { error?: string }
            streamError = d.error ?? 'LLM 对话流失败'
          }
          // done 无业务载荷；无 intent 的 done 由收尾处判失败。
        }
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          for (const ev of parser.push(decoder.decode(value, { stream: true }))) handle(ev)
        }
        for (const ev of parser.finish()) handle(ev)
        if (intent) {
          resolve(intent)
          return
        }
        throw new ApiError(streamError ?? '对话流意外结束（未收到 intent 事件）')
      } catch (err) {
        reject(err instanceof ApiError ? err : new ApiError((err as Error).message))
      }
    })()
  })
}

// ── 对局流式（SSE，全游戏通用）───────────────────────────────────
// 同一 /match/start 与 /match/move 路由，带 Accept: text/event-stream
//（+ ?stream=1），后端按事件契约发流：
//   progress{session} / snapshot{session} / error{error} / done{}
// progress = 每份可见变化后的玩家投影快照——人类行动落地一帧（发言/落子
// 即刻上屏），AI 每走一步再一帧（社交逐条发言、棋盘逐手落子），前端逐帧
// 更新棋盘，不再等服务端把整轮 AI 循环跑完才看到结果。
// 最终以 snapshot 事件 resolve；error 或流意外中断以 ApiError reject。
export interface MatchStreamHandlers {
  onProgress?: (session: import('../types').Snapshot) => void
}

function matchStream<T>(path: string, body: unknown, handlers: MatchStreamHandlers): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    void (async () => {
      try {
        let resp: Response
        try {
          resp = await fetch(BASE + path + '?stream=1', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Accept: 'text/event-stream',
            },
            body: JSON.stringify(body),
          })
        } catch {
          throw new ApiError(
            '无法连接服务器，请确认平台服务已启动 (python -m layer4_interface.frontend.platform.server)',
          )
        }
        if (!resp.ok || !resp.body) {
          let detail = `服务器返回异常 (HTTP ${resp.status})`
          const ct = resp.headers.get('Content-Type') ?? ''
          if (ct.includes('application/json')) {
            try {
              const data = (await resp.json()) as { error?: string }
              if (data?.error) detail = data.error
            } catch {
              /* 保持默认文案 */
            }
          }
          throw new ApiError(detail)
        }
        const reader = resp.body.getReader()
        const decoder = new TextDecoder('utf-8')
        const parser = new SseParser()
        let snap: T | null = null
        let streamError: string | null = null
        const handle = (ev: SseEvent): void => {
          if (ev.event === 'progress') {
            const d = JSON.parse(ev.data) as { session?: import('../types').Snapshot }
            if (d.session) handlers.onProgress?.(d.session)
          } else if (ev.event === 'snapshot') {
            snap = JSON.parse(ev.data) as T
          } else if (ev.event === 'error') {
            const d = JSON.parse(ev.data) as { error?: string }
            streamError = d.error ?? '对局流式推送失败'
          }
          // done 无业务载荷；无 snapshot 的 done 由收尾处判失败。
        }
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          for (const ev of parser.push(decoder.decode(value, { stream: true }))) handle(ev)
        }
        for (const ev of parser.finish()) handle(ev)
        if (snap) {
          resolve(snap)
          return
        }
        throw new ApiError(streamError ?? '对局流意外结束（未收到 snapshot 事件）')
      } catch (err) {
        reject(err instanceof ApiError ? err : new ApiError((err as Error).message))
      }
    })()
  })
}

export function matchStartStream(
  body: unknown,
  handlers: MatchStreamHandlers = {},
): Promise<{ session: import('../types').Snapshot }> {
  return matchStream<{ session: import('../types').Snapshot }>('/match/start', body, handlers)
}

export function matchMoveStream(
  gameId: string,
  action: unknown,
  handlers: MatchStreamHandlers = {},
): Promise<{ session: import('../types').Snapshot }> {
  return matchStream<{ session: import('../types').Snapshot }>(
    '/match/move',
    { game_id: gameId, action },
    handlers,
  )
}

/**
 * 流式对局请求 + 旧后端回退（``/match/start`` 与 ``/match/move`` 共用）。
 *
 * 为什么必须走流式：非流式请求要等服务端把整轮 AI 循环跑完才返回一次快照。
 * 谁是卧底 8 人桌一轮 describe 有 7 个 AI 座位，每位一次 LLM 调用（实测单次
 * 1~20s，取决于模型是否已加载）——整轮下来数十秒里前端**一帧都拿不到**，
 * 表现就是「轮不到自己发言，界面一直卡着」。流式下人类行动落地即一帧、
 * AI 每走一步再一帧，界面全程在动。
 *
 * 回退红线（**只有一帧都没收到就失败**才回退一次性 JSON 请求）：旧后端不认
 * SSE / 中间层吞流 / 连接建不起来时，退回原来的 JSON 信封保证还能玩；但一旦
 * 已经推进过任何一帧，说明服务端状态已经变了，此时再发一次同样的动作会把
 * 人类行动重复落地（本该走一步的回合走两步），所以宁可把错误抛给上层。
 *
 * @param stream  流式请求（``matchStartStream`` / ``matchMoveStream`` 的绑定）。
 * @param fallback 无帧可得时的 JSON 兜底请求。
 * @param onProgress 每帧进度快照回调（逐帧上屏）。
 */
export async function matchStreamOrFallback(
  stream: (handlers: MatchStreamHandlers) => Promise<{ session: import('../types').Snapshot }>,
  fallback: () => Promise<{ session: import('../types').Snapshot }>,
  onProgress: (session: import('../types').Snapshot) => void,
): Promise<{ session: import('../types').Snapshot }> {
  let progressed = false
  try {
    return await stream({
      onProgress: (session) => {
        progressed = true
        onProgress(session)
      },
    })
  } catch (err) {
    if (progressed) throw err
    return await fallback()
  }
}

export function apiPut<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

// ── 对话管理与存档 API (与后端 conversations.py 契约对齐) ────────

export function listConversations(): Promise<{ conversations: import('../types').ConversationMeta[] }> {
  return apiGet<{ conversations: import('../types').ConversationMeta[] }>('/conversations')
}

export function getConversation(convId: string): Promise<import('../types').Conversation> {
  return apiGet<{ conversation: import('../types').Conversation }>(
    `/conversations/${encodeURIComponent(convId)}`,
  ).then((d) => d.conversation)
}

export function createConversation(
  init?: { title?: string; messages?: import('../types').ChatMessage[] },
): Promise<import('../types').Conversation> {
  return apiPost<{ conversation: import('../types').Conversation }>('/conversations', init ?? {}).then(
    (d) => d.conversation,
  )
}

export function appendConversationMessages(
  convId: string,
  messages: import('../types').ChatMessage[],
): Promise<import('../types').ConversationMeta> {
  return apiPost<{ conversation: import('../types').ConversationMeta }>(
    `/conversations/${encodeURIComponent(convId)}/messages`,
    { messages },
  ).then((d) => d.conversation)
}

export function updateConversation(
  convId: string,
  patch: { title?: string; archived?: boolean },
): Promise<import('../types').ConversationMeta> {
  return apiPost<{ conversation: import('../types').ConversationMeta }>(
    `/conversations/${encodeURIComponent(convId)}`,
    patch,
  ).then((d) => d.conversation)
}

export function deleteConversation(convId: string): Promise<{ ok: boolean }> {
  return request(`/conversations/${encodeURIComponent(convId)}`, { method: 'DELETE' })
}

// ── 在线学习 API ────────────────────────────────────────────────

export function getLearningStatus(): Promise<{ learning: import('../types').LearningStatus[] }> {
  return apiGet<{ learning: import('../types').LearningStatus[] }>('/learning/status')
}

export function applyLearning(gameId?: string): Promise<{ result: import('../types').LearningApplyResult } | { results: import('../types').LearningApplyResult[] }> {
  return apiPost<{ result: import('../types').LearningApplyResult } | { results: import('../types').LearningApplyResult[] }>(
    '/learning/apply',
    gameId ? { game_id: gameId } : {},
  )
}

export function setLearningConfig(gameId: string, enabled: boolean): Promise<{ learning: import('../types').LearningStatus }> {
  return apiPost<{ learning: import('../types').LearningStatus }>('/learning/config', { game_id: gameId, enabled })
}

// ── Agent 陪伴 / 偏好 / 复盘 API ────────────────────────────────
// 这些路由由集成阶段接线 (D.1)；前端只按冻结契约的 JSON 结构调用。

// 后端要求 game_id 作为「会话 id」（与 /match/state、/match/move 一致），
// 响应仍套 {\"ok\": ..., key: ...} 信封，这里负责解包。
export function agentSay(
  gameId: string,
  scenario: string,
  extra?: Record<string, unknown>,
): Promise<import('../types').AgentMessage> {
  return apiPost<{ message: import('../types').AgentMessage }>('/agent/say', {
    game_id: gameId,
    scenario,
    ...(extra ?? {}),
  }).then((d) => d.message)
}

export function matchHint(gameId: string, level: import('../types').HintLevel): Promise<import('../types').AgentMessage> {
  return apiPost<{ hint: { hint: string } }>('/match/hint', { game_id: gameId, level }).then((d) => ({
    text: d.hint.hint,
    mood: 'thinking',
  }))
}

// 后端 /profile 将档案嵌套在 {"ok": ..., "profile": {...}} 信封里
// （与 /games → games、/match/active → sessions 一致），这里负责解包。
export function getProfile(): Promise<import('../types').Profile> {
  return apiGet<{ profile: import('../types').Profile }>('/profile').then((d) => d.profile)
}

export function saveProfile(profile: import('../types').Profile): Promise<import('../types').Profile> {
  return apiPut<{ profile: import('../types').Profile }>('/profile', { profile }).then((d) => d.profile)
}

// 只提交变更字段 —— 后端以 `{**load(), **patch}` 合并，所以对话里改偏好
// （"换成高冷竞技"）可以只写 `{default_persona: 'cold'}`，不必先读全量档案。
export function patchProfile(
  patch: Partial<import('../types').Profile>,
): Promise<import('../types').Profile> {
  return apiPut<{ profile: import('../types').Profile }>('/profile', { profile: patch }).then((d) => d.profile)
}

export function clearProfile(): Promise<{ ok: boolean }> {
  return apiPost<{ ok: boolean }>('/profile/clear', {})
}

// ── LLM 配置 API ────────────────────────────────────────────────
// 平台持久化配置（data/llm_config.json）> 环境变量 > 内置默认；保存后
// 聊天 / Agent 对话 / 规则翻译 / 社交 AI 立即使用新端点与模型。
// 密钥只写不回显：GET 只给 has_api_key；保存时省略 api_key = 保持不变，
// 传空串 = 清除。

export interface LlmConfigInfo {
  base_url: string
  model: string
  has_api_key: boolean
  effective_base_url: string
  effective_model: string
  source: 'platform' | 'env' | 'default'
  available?: boolean
}

export function getLlmConfig(): Promise<{ config: LlmConfigInfo }> {
  return apiGet<{ config: LlmConfigInfo }>('/llm/config')
}

export function saveLlmConfig(patch: {
  base_url?: string
  model?: string
  api_key?: string
}): Promise<{ config: LlmConfigInfo }> {
  return apiPut<{ config: LlmConfigInfo }>('/llm/config', patch)
}

export function testLlmConnection(patch: {
  base_url?: string
  api_key?: string
}): Promise<{ reachable: boolean; error: string; base_url: string }> {
  return apiPost<{ reachable: boolean; error: string; base_url: string }>('/llm/test', patch)
}

export function getReview(matchId: string): Promise<import('../types').ReviewReport> {
  return apiGet<import('../types').ReviewReport>(`/review/${matchId}`)
}

// ── 对局记录 API（平台「对局记录」页）────────────────────────────
// 后端 GET /api/history 契约（见 layer4_interface/frontend/platform/server.py）：
//   {ok, matches, total, has_more}；matches 为 meta 列表（含注入的 seat_names），
//   limit 默认 100、上限 500，offset 分页，其余筛选参数非法时后端按默认值处理。
// 与 client 内其他函数一致：内部解包命名 key，调用方只拿业务对象。

export interface HistoryQueryParams {
  gameId?: string | null
  /** 玩家视角结果过滤（win/lose/draw）；空 = 全部。 */
  result?: ResultKind | '' | null
  /** 关键字（匹配游戏名 / 座位称呼）。 */
  q?: string | null
  /** ISO-8601 起始日期（含）。 */
  since?: string | null
  limit?: number
  offset?: number
}

export interface HistoryPage {
  matches: import('../types').MatchMeta[]
  total: number
  has_more: boolean
}

export function listHistory(params: HistoryQueryParams = {}): Promise<HistoryPage> {
  const query = buildHistoryQuery({
    gameId: params.gameId,
    result: params.result,
    q: params.q,
    since: params.since,
    limit: params.limit,
    offset: params.offset,
  })
  return apiGet<HistoryPage>(`/history${query ? `?${query}` : ''}`).then((d) => ({
    matches: d.matches ?? [],
    total: typeof d.total === 'number' ? d.total : (d.matches ?? []).length,
    has_more: d.has_more === true,
  }))
}

// ── 自定义游戏 API (A2 后端契约 / A3 前端) ──────────────────────

export interface CustomCreateBody {
  mode: 'from_scratch' | 'variant'
  rule_text?: string
  base_game_id?: string
  change_text?: string
  game_name?: string
  source_lang?: string
  use_llm?: boolean
}

interface CustomCreateErrorBody {
  ok: boolean
  error?: string
  validation?: { valid?: boolean; errors?: string[]; warnings?: string[] }
}

/** 创建自定义游戏 — 失败时把 validation errors 并入 ApiError 消息，便于页面展示。 */
export async function createCustomGame(body: CustomCreateBody): Promise<import('../types').CustomCreateResult> {
  let resp: Response
  try {
    resp = await fetch(BASE + '/custom/games', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch {
    throw new ApiError('无法连接服务器，请确认平台服务已启动 (python -m layer4_interface.frontend.platform.server)')
  }
  let data: import('../types').CustomCreateResult | CustomCreateErrorBody
  try {
    data = (await resp.json()) as import('../types').CustomCreateResult
  } catch {
    throw new ApiError(`服务器返回异常 (HTTP ${resp.status})`)
  }
  if (!data.ok) {
    throw new ApiError(customCreateErrorText(data as CustomCreateErrorBody, `请求失败 (HTTP ${resp.status})`))
  }
  return data as import('../types').CustomCreateResult
}

/** 创建失败的展示文案：主原因 + 校验错误明细（后端已给中文原因）。 */
function customCreateErrorText(err: CustomCreateErrorBody, fallback: string): string {
  const validationErrors = err.validation?.errors ?? []
  const detail = validationErrors.length > 0 ? `：${validationErrors.join('；')}` : ''
  return `${err.error ?? fallback}${detail}`
}

export interface CustomCreateStage {
  stage: string
  detail: string
}

export interface CustomCreateStreamHandlers {
  /** 创建阶段进度（正在模板翻译 / 正在用 LLM 翻译 / 校验 / 注册）。 */
  onStage?: (stage: CustomCreateStage) => void
}

/**
 * 流式创建自定义游戏（SSE）——创建过程对用户可见。
 *
 * 为什么必须走流式：勾选「LLM 生成」后一次翻译要跑 1-3 分钟（推理模型还要
 * 先想再答），非流式请求在这段时间里**一帧都拿不到**，用户看到的就是
 * 「等了半天，什么也没出现」。流式下服务端每个阶段推一条 ``stage``，
 * 创建页据此显示「正在用 LLM 翻译…（已等 42s）」，最后以 ``result`` 收口。
 *
 * 旧后端（不认 SSE）回退到一次性 JSON 请求：创建是幂等的「新建一局」动作，
 * 没有「重复落子」那种回退红线（失败即失败，不会产生半成品）。
 */
export async function createCustomGameStream(
  body: CustomCreateBody,
  handlers: CustomCreateStreamHandlers = {},
): Promise<import('../types').CustomCreateResult> {
  let resp: Response
  try {
    resp = await fetch(BASE + '/custom/games?stream=1', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(body),
    })
  } catch {
    return createCustomGame(body)
  }
  const contentType = resp.headers.get('Content-Type') ?? ''
  if (!resp.ok || !resp.body || !contentType.includes('text/event-stream')) {
    // 旧后端 / 非流式响应：直接按 JSON 路径解析（复用同一份错误文案）。
    if (resp.ok) return createCustomGame(body)
    let detail = `服务器返回异常 (HTTP ${resp.status})`
    try {
      const data = (await resp.json()) as CustomCreateErrorBody
      detail = customCreateErrorText(data, detail)
    } catch {
      /* 保持默认文案 */
    }
    throw new ApiError(detail)
  }
  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  const parser = new SseParser()
  let result: import('../types').CustomCreateResult | null = null
  let streamError: string | null = null
  const handle = (ev: SseEvent): void => {
    if (ev.event === 'stage') {
      const d = JSON.parse(ev.data) as CustomCreateStage
      handlers.onStage?.({ stage: d.stage ?? '', detail: d.detail ?? '' })
    } else if (ev.event === 'result') {
      result = JSON.parse(ev.data) as import('../types').CustomCreateResult
    } else if (ev.event === 'error') {
      const d = JSON.parse(ev.data) as CustomCreateErrorBody
      streamError = customCreateErrorText(d, '创建失败')
    }
  }
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    for (const ev of parser.push(decoder.decode(value, { stream: true }))) handle(ev)
  }
  for (const ev of parser.finish()) handle(ev)
  if (result) return result
  throw new ApiError(streamError ?? '创建流意外结束（未收到结果）')
}

export function listCustomGames(): Promise<{ games: import('../types').GameInfo[] }> {
  return apiGet<{ games: import('../types').GameInfo[] }>('/custom/games')
}

export function deleteCustomGame(gameId: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/custom/games/${encodeURIComponent(gameId)}`, { method: 'DELETE' })
}
