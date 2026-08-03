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
