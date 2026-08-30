// API 客户端 — 解包 {"ok": ...} 信封, 失败时抛出 ApiError

import { SseParser, type SseEvent } from '../chat/sse.ts'

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
    const err = data as CustomCreateErrorBody
    const validationErrors = err.validation?.errors ?? []
    const detail = validationErrors.length > 0 ? `：${validationErrors.join('；')}` : ''
    throw new ApiError(`${err.error ?? `请求失败 (HTTP ${resp.status})`}${detail}`)
  }
  return data as import('../types').CustomCreateResult
}

export function listCustomGames(): Promise<{ games: import('../types').GameInfo[] }> {
  return apiGet<{ games: import('../types').GameInfo[] }>('/custom/games')
}

export function deleteCustomGame(gameId: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/custom/games/${encodeURIComponent(gameId)}`, { method: 'DELETE' })
}
