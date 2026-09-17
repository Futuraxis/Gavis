// 对局流式客户端契约测试（/api/match/start|move 的 SSE + 旧后端回退策略）。
//
// 回归背景（2026-09）：对话页（默认界面）曾经**只**用 apiPost('/match/move')
// 打非流式接口，服务端明明支持 SSE 也拿不到逐帧进度——谁是卧底 8 人桌一轮
// describe 有 7 个 AI 座位、每位一次 LLM 调用，玩家「轮不到自己发言」时界面
// 数十秒一帧不动，看起来就是卡死。本文件锁定三件事：
//   1) 请求形态真的是流式（?stream=1 + Accept: text/event-stream）；
//   2) progress 帧逐帧回调（AI 每走一步都要能上屏）；
//   3) 回退只在「一帧都没收到」时发生——收到过帧再回退会把人类行动重放。
//
// 运行：cd platform-frontend && npm run test:frontend

import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  ApiError,
  matchMoveStream,
  matchStartStream,
  matchStreamOrFallback,
} from '../src/api/client.ts'
import type { Snapshot } from '../src/types.ts'

type FetchCall = { url: string; init?: RequestInit }

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
}

/** 用若干 SSE 帧构造响应（可指定分块，验证跨块解析）。 */
function sseResponse(chunks: string[]): Response {
  const encoder = new TextEncoder()
  const reader = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c))
      controller.close()
    },
  })
  return new Response(reader, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
}

function mockSse(chunks: string[]): FetchCall[] {
  const calls: FetchCall[] = []
  globalThis.fetch = async (url: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    calls.push({ url: String(url), init })
    return sseResponse(chunks)
  }
  return calls
}

function snap(turn: string | null, extra: Record<string, unknown> = {}): Snapshot {
  return { game_id: 'sess-1', player_pid: 'p0', turn, over: false, ...extra } as unknown as Snapshot
}

// ── 请求形态 ─────────────────────────────────────────────────

test('matchStartStream：POST /api/match/start?stream=1 + Accept: text/event-stream', async () => {
  const calls = mockSse([frame('snapshot', { session: snap('p0') }), frame('done', {})])
  await matchStartStream({ game_id: 'undercover' })
  assert.equal(calls.length, 1)
  assert.equal(calls[0].url, '/api/match/start?stream=1')
  assert.equal((calls[0].init?.method ?? 'GET').toUpperCase(), 'POST')
  const accept = new Headers(calls[0].init?.headers).get('accept')
  assert.ok(accept && accept.includes('text/event-stream'), '必须协商 SSE')
  assert.deepEqual(JSON.parse(String(calls[0].init?.body)), { game_id: 'undercover' })
})

test('matchMoveStream：POST /api/match/move?stream=1，body 为 {game_id, action}', async () => {
  const calls = mockSse([frame('snapshot', { session: snap('p0') }), frame('done', {})])
  await matchMoveStream('sess-1', { type: 'speak', text: '你好' })
  assert.equal(calls[0].url, '/api/match/move?stream=1')
  assert.deepEqual(JSON.parse(String(calls[0].init?.body)), {
    game_id: 'sess-1',
    action: { type: 'speak', text: '你好' },
  })
})

// ── 逐帧进度（卧底「轮不到自己」时界面在动的唯一来源）────────────

test('matchMoveStream：progress 帧按序逐帧回调（AI 每走一步一帧）', async () => {
  const calls = mockSse([
    frame('progress', { session: snap('p1', { phase: 'describe' }) }),
    frame('progress', { session: snap('p2', { phase: 'describe' }) }),
    frame('progress', { session: snap('p3', { phase: 'describe' }) }),
    frame('snapshot', { session: snap('p0', { phase: 'vote' }) }),
    frame('done', {}),
  ])
  const seen: (string | null)[] = []
  const data = await matchMoveStream('sess-1', { type: 'speak', text: '我先说' }, {
    onProgress: (s) => seen.push(s.turn),
  })
  assert.deepEqual(seen, ['p1', 'p2', 'p3'], 'AI 发言必须逐条可见，不能等整轮跑完')
  assert.equal(data.session.turn, 'p0')
  assert.equal(calls.length, 1)
})

test('matchStartStream：跨块分片的帧也能逐帧回调并 resolve', async () => {
  const raw =
    frame('progress', { session: snap('p0') }) +
    frame('progress', { session: snap('p1') }) +
    frame('snapshot', { session: snap('p2') }) +
    frame('done', {})
  const mid = Math.floor(raw.length / 2)
  mockSse([raw.slice(0, mid), raw.slice(mid)])
  const seen: (string | null)[] = []
  const data = await matchStartStream({ game_id: 'undercover' }, { onProgress: (s) => seen.push(s.turn) })
  assert.deepEqual(seen, ['p0', 'p1'])
  assert.equal(data.session.turn, 'p2')
})

// ── 失败形态 ─────────────────────────────────────────────────

test('error 事件且无 snapshot → reject ApiError 带服务端文案', async () => {
  mockSse([frame('error', { error: '未知对局: deadbeef' }), frame('done', {})])
  await assert.rejects(
    () => matchMoveStream('sess-1', { cell_index: 0 }),
    (err: unknown) => err instanceof ApiError && err.message.includes('未知对局'),
  )
})

test('done 但无 snapshot → reject（流意外结束，界面据此解除卡住状态）', async () => {
  mockSse([frame('done', {})])
  await assert.rejects(() => matchMoveStream('sess-1', { cell_index: 0 }), /意外结束/)
})

// ── 回退策略（matchStreamOrFallback）──────────────────────────

test('一帧都没收到就失败 → 回退一次性 JSON 请求（旧后端不认 SSE 也能玩）', async () => {
  let streamCalls = 0
  let fallbackCalls = 0
  const data = await matchStreamOrFallback(
    async () => {
      streamCalls += 1
      throw new ApiError('服务器返回异常 (HTTP 404)')
    },
    async () => {
      fallbackCalls += 1
      return { session: snap('p0') }
    },
    () => undefined,
  )
  assert.equal(streamCalls, 1)
  assert.equal(fallbackCalls, 1, '无帧可得的失败必须回退，不能把整局玩挂')
  assert.equal(data.session.turn, 'p0')
})

test('已收到进度帧后失败 → 不回退，直接抛出（否则人类行动会被重放）', async () => {
  let fallbackCalls = 0
  await assert.rejects(
    () =>
      matchStreamOrFallback(
        async (handlers) => {
          handlers.onProgress?.(snap('p1'))
          throw new ApiError('对局流意外结束')
        },
        async () => {
          fallbackCalls += 1
          return { session: snap('p1') }
        },
        () => undefined,
      ),
    /意外结束/,
  )
  assert.equal(fallbackCalls, 0, '推进过帧说明服务端状态已改，重发动作会重复落子')
})

test('回退成功路径：流式未推进帧时把 progress 回调留给上层逐帧上屏', async () => {
  const seen: (string | null)[] = []
  const data = await matchStreamOrFallback(
    async (handlers) => {
      handlers.onProgress?.(snap('p1'))
      handlers.onProgress?.(snap('p2'))
      return { session: snap('p0') }
    },
    async () => {
      throw new ApiError('不该走到这里')
    },
    (s) => seen.push(s.turn),
  )
  assert.deepEqual(seen, ['p1', 'p2'])
  assert.equal(data.session.turn, 'p0')
})
