// 前端 API 客户端契约测试（零依赖：node:test + fetch mock）。
//
// 回归背景（2026-08 前端事故，均由此文件锁定，防止复发）：
//   1) 后端所有业务接口都套 {ok, <key>} 信封（/profile → profile、
//      /games → games、/match/active → sessions、/agent/say → message、
//      /match/hint → hint、/match/start|move|state → session）。
//      客户端必须解包命名 key；不解包会把整个信封当业务对象存，
//      profile.recent 变 undefined → HomePage 整页崩溃。
//   2) /agent/say 与 /match/hint 的后端参数是「会话 id」（= session.game_id），
//      缺了会 400 Bad Request。
//   3) 失败形态：ok=false / 网络失败 / 非 JSON 都应抛 ApiError，页面据此兜底。
//
// 运行：cd platform-frontend && npm run test:frontend
// （node --experimental-strip-types --test，无需安装任何依赖）

import assert from 'node:assert/strict'
import { test } from 'node:test'

import { agentSay, ApiError, deleteCustomGame, getProfile, listCustomGames, matchHint, saveProfile } from '../src/api/client.ts'
import { recentOf } from '../src/profile.ts'
import type { Profile } from '../src/types.ts'

type FetchCall = { url: string; init?: RequestInit }

interface MockOptions {
  status?: number
  raw?: string // 非 JSON 响应体（模拟后端出错/网关页面）
}

function mockFetch(payload: unknown, opts: MockOptions = {}): FetchCall[] {
  const calls: FetchCall[] = []
  globalThis.fetch = async (url: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    calls.push({ url: String(url), init })
    const status = opts.status ?? 200
    if (opts.raw !== undefined) {
      return new Response(opts.raw, { status, headers: { 'Content-Type': 'text/html; charset=utf-8' } })
    }
    return new Response(JSON.stringify(payload), {
      status,
      headers: { 'Content-Type': 'application/json' },
    })
  }
  return calls
}

function mockFetchFailure(): FetchCall[] {
  const calls: FetchCall[] = []
  globalThis.fetch = async (url: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    calls.push({ url: String(url), init })
    throw new TypeError('fetch failed: network is down')
  }
  return calls
}

function bodyOf(call: FetchCall): unknown {
  assert.ok(call.init?.body !== undefined, '请求应带 JSON body')
  return JSON.parse(String(call.init.body))
}

function methodOf(call: FetchCall): string {
  return (call.init?.method ?? 'GET').toUpperCase()
}

/** 与后端 DEFAULT_PROFILE 同构的最小档案（含 recent）。 */
const FULL_PROFILE: Profile = {
  nickname: '阿远',
  agent_call: '小G',
  default_persona: 'gentle',
  default_difficulty: 'normal',
  hint_level: 'direction',
  pacing: 'standard',
  adaptive: true,
  difficulty_locked: false,
  learning_enabled: true,
  theme: 'light',
  recent: { moon_chess: { wins: 3, plays: 5 } },
}

// ── /profile ─────────────────────────────────────────────────

test('getProfile：解包 {ok, profile} 信封，返回裸档案（回归: 信封当档案存 → recent undefined → 整页崩溃）', async () => {
  const calls = mockFetch({ ok: true, profile: FULL_PROFILE })
  const profile = await getProfile()

  assert.equal(calls.length, 1)
  assert.equal(calls[0].url, '/api/profile')
  assert.equal(methodOf(calls[0]), 'GET')
  assert.equal(profile.nickname, '阿远')
  assert.deepEqual(profile.recent, { moon_chess: { wins: 3, plays: 5 } })
  assert.equal('ok' in (profile as object), false, '信封键不得泄漏进业务对象')
  assert.equal('profile' in (profile as object), false, '信封键不得泄漏进业务对象')
})

test('saveProfile：请求体为 {profile}（后端按 payload["profile"] 读），响应解包', async () => {
  const calls = mockFetch({ ok: true, profile: { ...FULL_PROFILE, nickname: '玩家2', theme: 'dark' } })
  const saved = await saveProfile({ ...FULL_PROFILE, nickname: '玩家2', theme: 'dark' })

  assert.equal(methodOf(calls[0]), 'PUT')
  assert.deepEqual(bodyOf(calls[0]), { profile: { ...FULL_PROFILE, nickname: '玩家2', theme: 'dark' } })
  assert.equal(saved.nickname, '玩家2')
  assert.equal(saved.theme, 'dark')
})

// ── /agent/say ───────────────────────────────────────────────

test('agentSay：必须携带 game_id（会话 id）与 scenario，并解包 {ok, message}', async () => {
  const calls = mockFetch({ ok: true, message: { scenario: 'chat', text: '我在的，你继续说。', mood: 'happy' } })
  const msg = await agentSay('session-abc', 'chat', { message: '在吗？' })

  assert.equal(methodOf(calls[0]), 'POST')
  assert.deepEqual(bodyOf(calls[0]), { game_id: 'session-abc', scenario: 'chat', message: '在吗？' })
  assert.equal(msg.text, '我在的，你继续说。')
  assert.equal(msg.mood, 'happy')
})

// ── /match/hint ──────────────────────────────────────────────

test('matchHint：携带 game_id/level，将 {ok, hint} 映射为聊天气泡 {text, mood}', async () => {
  const calls = mockFetch({
    ok: true,
    hint: { level: 'direction', direction: '当前落后，先补强防守', mechanical_text: '…', hint: '当前落后，先补强防守' },
  })
  const hint = await matchHint('session-abc', 'direction')

  assert.equal(methodOf(calls[0]), 'POST')
  assert.deepEqual(bodyOf(calls[0]), { game_id: 'session-abc', level: 'direction' })
  assert.equal(hint.text, '当前落后，先补强防守')
  assert.equal(hint.mood, 'thinking')
})

// ── 失败形态 ─────────────────────────────────────────────────

test('信封 ok=false：抛 ApiError 且携带后端 error 文案', async () => {
  mockFetch({ ok: false, error: '未知对局: deadbeef' })
  await assert.rejects(
    () => getProfile(),
    (err: unknown) => err instanceof ApiError && err.message.includes('未知对局'),
  )
})

test('网络不可达：抛 ApiError（页面据此走本地兜底分支）', async () => {
  mockFetchFailure()
  await assert.rejects(
    () => agentSay('session-abc', 'chat'),
    (err: unknown) => err instanceof ApiError && err.message.includes('无法连接服务器'),
  )
  await assert.rejects(
    () => getProfile(),
    (err: unknown) => err instanceof ApiError && err.message.includes('无法连接服务器'),
  )
})

test('非 JSON 响应：抛 ApiError（不会把 HTML 当业务数据继续渲染）', async () => {
  mockFetch({}, { raw: '<html><body>Bad Gateway</body></html>', status: 502 })
  await assert.rejects(
    () => getProfile(),
    (err: unknown) => err instanceof ApiError && err.message.includes('服务器返回异常'),
  )
})

// ── 页面兜底（第二道防线）────────────────────────────────────

test('recentOf：档案缺 recent 字段时返回空表（HomePage/ProfilePage 整页崩溃的兜底）', () => {
  assert.deepEqual(recentOf(undefined), {})
  assert.deepEqual(recentOf(null), {})
  assert.deepEqual(recentOf({}), {})
  assert.deepEqual(recentOf({ nickname: 'x' }), {})
  assert.deepEqual(recentOf(FULL_PROFILE), { moon_chess: { wins: 3, plays: 5 } })
})

// ── 自定义游戏管理（删除/列表）：平台端「自定义变体不能删除」回归 ────

test('deleteCustomGame：DELETE /api/custom/games/{id}，URL 编码 game_id，解包 {ok}', async () => {
  const calls = mockFetch({ ok: true })
  // 空格必须编码为 %20；! 依 encodeURIComponent 规范保持原样（后端 unquote 均能还原）。
  const res = await deleteCustomGame('my variant')

  assert.equal(calls.length, 1)
  assert.equal(methodOf(calls[0]), 'DELETE')
  assert.equal(calls[0].url, '/api/custom/games/my%20variant')
  assert.deepEqual(res, { ok: true })
})

test('deleteCustomGame：ok=false 抛 ApiError（404 不存在 / 服务未启用时页面可展示错误）', async () => {
  mockFetch({ ok: false, error: '自定义游戏不存在: my_game' })
  await assert.rejects(
    () => deleteCustomGame('my_game'),
    (err: unknown) => err instanceof ApiError && err.message.includes('自定义游戏不存在'),
  )
})

test('listCustomGames：GET /api/custom/games 解包 {ok, games}（管理列表渲染用）', async () => {
  const calls = mockFetch({
    ok: true,
    games: [
      {
        game_id: 'my_game',
        display_name: '我的变体',
        description: '…',
        kind: 'board',
        board_size: 8,
        seat_options: ['人机'],
        seat_label: '座位',
        player_counts: [2],
        difficulties: ['easy'],
        solver_options: ['mcts'],
        family: 'grid',
        custom: true,
        created_at: '2026-08-01T10:00:00+08:00',
      },
    ],
  })
  const data = await listCustomGames()

  assert.equal(calls.length, 1)
  assert.equal(calls[0].url, '/api/custom/games')
  assert.equal(methodOf(calls[0]), 'GET')
  assert.equal(data.games.length, 1)
  assert.equal(data.games[0].game_id, 'my_game')
  assert.equal(data.games[0].family, 'grid')
  assert.equal(data.games[0].created_at, '2026-08-01T10:00:00+08:00')
})