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

export function agentSay(scenario: string, extra?: Record<string, unknown>): Promise<import('../types').AgentMessage> {
  return apiPost<import('../types').AgentMessage>('/agent/say', { scenario, ...(extra ?? {}) })
}

export function matchHint(level: import('../types').HintLevel): Promise<import('../types').AgentMessage> {
  return apiPost<import('../types').AgentMessage>('/match/hint', { level })
}

export function getProfile(): Promise<import('../types').Profile> {
  return apiGet<import('../types').Profile>('/profile')
}

export function saveProfile(profile: import('../types').Profile): Promise<import('../types').Profile> {
  return apiPut<import('../types').Profile>('/profile', profile)
}

export function clearProfile(): Promise<{ ok: boolean }> {
  return apiPost<{ ok: boolean }>('/profile/clear', {})
}

export function getReview(matchId: string): Promise<import('../types').ReviewReport> {
  return apiGet<import('../types').ReviewReport>(`/review/${matchId}`)
}
