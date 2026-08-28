// API 客户端 — 解包 {"ok": ...} 信封, 失败时抛出 ApiError

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
export function chatTurn(
  text: string,
  gameId?: string,
  history?: { role: 'user' | 'assistant'; content: string }[],
): Promise<import('../types').ChatTurnResult> {
  return apiPost<import('../types').ChatTurnResult>('/chat', {
    text,
    ...(gameId ? { game_id: gameId } : {}),
    ...(history && history.length > 0 ? { history } : {}),
  })
}

export function apiPut<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
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
