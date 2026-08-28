// /api/chat 客户端契约测试（chat/client.ts 的 chatTurn 封装）。
//
// 后端 POST /api/chat 返回 {ok, intent, text, mood, params}——这是聊天前端
// 的主路径：一句话 → 意图 → 平台动作。这里锁定请求体与解包形态，
// 防止「会话 id 丢失 / 信封当结果存」这类前端事故复发。

import assert from 'node:assert/strict'
import { test } from 'node:test'

import { chatTurn } from '../src/api/client.ts'

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
