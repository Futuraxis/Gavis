// /api/chat 客户端契约测试（chat/client.ts 的 chatTurn 封装）。
//
// 后端 POST /api/chat 返回 {ok, intent, text, mood, params}——这是聊天前端
// 的主路径：一句话 → 意图 → 平台动作。这里锁定请求体与解包形态，
// 防止「会话 id 丢失 / 信封当结果存」这类前端事故复发。

import assert from 'node:assert/strict'
import { test } from 'node:test'

import { chatTurn, chatTurnStream } from '../src/api/client.ts'

type FetchCall = { url: string; init?: RequestInit }

function mockFetch(payload: unknown): FetchCall[] {
  const calls: FetchCall[] = []
  globalThis.fetch = async (url: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    calls.push({ url: String(url), init })
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  }
  return calls
}

function bodyOf(call: FetchCall): Record<string, unknown> {
  assert.ok(call.init?.body !== undefined, '请求应带 JSON body')
  return JSON.parse(String(call.init.body)) as Record<string, unknown>
}

const REPLY = {
  ok: true,
  intent: 'play',
  text: '好，来一局月亮棋！对局正在创建…',
  mood: 'happy',
  params: { game_id: 'moon_chess' },
}

test('chatTurn：POST /api/chat；有会话时带 game_id（会话 id）', async () => {
  const calls = mockFetch(REPLY)
  const result = await chatTurn('我想玩月亮棋', 'sess-1')
  assert.equal(calls.length, 1)
  assert.equal(calls[0].url, '/api/chat')
  assert.equal((calls[0].init?.method ?? 'GET').toUpperCase(), 'POST')
  const body = bodyOf(calls[0])
  assert.equal(body.text, '我想玩月亮棋')
  assert.equal(body.game_id, 'sess-1')
  assert.equal(result.intent, 'play')
  assert.equal(result.mood, 'happy')
  assert.deepEqual(result.params, { game_id: 'moon_chess' })
})

test('chatTurn：无活跃会话时不带 game_id', async () => {
  const calls = mockFetch(REPLY)
  await chatTurn('来一局')
  const body = bodyOf(calls[0])
  assert.equal(body.text, '来一局')
  assert.equal('game_id' in body, false)
  assert.equal('history' in body, false)
})

test('chatTurn：带 history 时原样入请求体（user/assistant 轮，最新在后）', async () => {
  const calls = mockFetch(REPLY)
  const history = [
    { role: 'user' as const, content: '我想玩德州扑克' },
    { role: 'assistant' as const, content: '好，来一局德州扑克！' },
  ]
  await chatTurn('那月亮棋呢', undefined, history)
  const body = bodyOf(calls[0])
  assert.deepEqual(body.history, history)
  assert.deepEqual(body.text, '那月亮棋呢')
})

test('chatTurn：clarify 意图的 chips 参数原样透传', async () => {
  const clarify = {
    ok: true,
    intent: 'clarify',
    text: '想玩哪一款？',
    mood: 'neutral',
    params: { chips: ['月亮棋', '德州扑克'] },
  }
  const calls = mockFetch(clarify)
  const result = await chatTurn('来一局')
  assert.equal(result.intent, 'clarify')
  assert.deepEqual(result.params.chips, ['月亮棋', '德州扑克'])
})

test('chatTurn：网络失败抛 ApiError（页面据此走本地兜底分类）', async () => {
  globalThis.fetch = async (): Promise<Response> => {
    throw new TypeError('fetch failed: network is down')
  }
  await assert.rejects(() => chatTurn('你好'), /无法连接服务器/)
})

// ── chatTurnStream（SSE 流式模式）───────────────────────────────

/** SSE intent 事件载荷 = ChatTurnResult（无 ok 信封；与后端 chat_turn_stream 契约一致）。 */
const STREAM_REPLY = {
  intent: 'play' as const,
  text: '好，来一局月亮棋！对局正在创建…',
  mood: 'happy' as const,
  params: { game_id: 'moon_chess' },
}

/** 用一段 SSE 帧构造 Response（整体一包；分块场景另行显式构造）。 */
function sseResponse(body: string): Response {
  const encoder = new TextEncoder()
  return new Response(encoder.encode(body), {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

function frame(event: string, data: string): string {
  return `event: ${event}\ndata: ${data}\n\n`
}

function mockSseFetch(body: string): FetchCall[] {
  const calls: FetchCall[] = []
  globalThis.fetch = async (url: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    calls.push({ url: String(url), init })
    return sseResponse(body)
  }
  return calls
}

test('chatTurnStream：请求形态（POST /api/chat?stream=1 + Accept: text/event-stream）', async () => {
  const calls = mockSseFetch(frame('intent', JSON.stringify(STREAM_REPLY)) + frame('done', '{}'))
  await chatTurnStream('我想玩月亮棋', 'sess-1', undefined, {})
  assert.equal(calls.length, 1)
  assert.equal(calls[0].url, '/api/chat?stream=1')
  assert.equal((calls[0].init?.method ?? 'GET').toUpperCase(), 'POST')
  const accept = new Headers(calls[0].init?.headers).get('accept')
  assert.ok(accept && accept.includes('text/event-stream'))
  const body = bodyOf(calls[0])
  assert.equal(body.text, '我想玩月亮棋')
  assert.equal(body.game_id, 'sess-1')
})

test('chatTurnStream：增量事件按序回调，intent 事件 resolve', async () => {
  const stream = mockSseFetch(
    frame('reasoning', '{"delta":"先看中心"}') +
      frame('text', '{"delta":"这一步建议"}') +
      frame('text', '{"delta":"占中心。"}') +
      frame('intent', JSON.stringify(STREAM_REPLY)) +
      frame('done', '{}'),
  )
  const texts: string[] = []
  const reasonings: string[] = []
  const result = await chatTurnStream('怎么走', undefined, undefined, {
    onText: (d) => texts.push(d),
    onReasoning: (d) => reasonings.push(d),
  })
  assert.deepEqual(texts, ['这一步建议', '占中心。'])
  assert.deepEqual(reasonings, ['先看中心'])
  assert.equal(result.intent, 'play')
  assert.equal(result.text, STREAM_REPLY.text)
  assert.equal(stream.length, 1)
})

test('chatTurnStream：跨块分片的帧也能正确解析', async () => {
  const raw =
    frame('text', '{"delta":"你好"}') + frame('intent', JSON.stringify(STREAM_REPLY)) + frame('done', '{}')
  const mid = Math.floor(raw.length / 2)
  const encoder = new TextEncoder()
  const reader = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(raw.slice(0, mid)))
      controller.enqueue(encoder.encode(raw.slice(mid)))
      controller.close()
    },
  })
  globalThis.fetch = async (): Promise<Response> => new Response(reader, { status: 200 })
  const result = await chatTurnStream('你好', undefined, undefined, {})
  assert.equal(result.intent, 'play')
  assert.equal(result.text, STREAM_REPLY.text)
})

test('chatTurnStream：error 事件 → 未收到 intent 时 reject ApiError', async () => {
  mockSseFetch(frame('error', '{"error":"LLM 端点不可达/超时"}') + frame('done', '{}'))
  await assert.rejects(() => chatTurnStream('你好'), /LLM 端点不可达\/超时/)
})

test('chatTurnStream：done 但无 intent → reject（流意外结束）', async () => {
  mockSseFetch(frame('done', '{}'))
  await assert.rejects(() => chatTurnStream('你好'), /未收到 intent/)
})

test('chatTurnStream：HTTP 非 2xx 且带 JSON 错误体 → ApiError 带服务端文案', async () => {
  globalThis.fetch = async (): Promise<Response> =>
    new Response(JSON.stringify({ ok: false, error: '请求体超限' }), {
      status: 413,
      headers: { 'Content-Type': 'application/json' },
    })
  await assert.rejects(() => chatTurnStream('你好'), /请求体超限/)
})

test('chatTurnStream：网络失败 → ApiError（无法连接服务器）', async () => {
  globalThis.fetch = async (): Promise<Response> => {
    throw new TypeError('fetch failed: network is down')
  }
  await assert.rejects(() => chatTurnStream('你好'), /无法连接服务器/)
})
