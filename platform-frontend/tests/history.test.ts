// 对局元数据纯函数测试（src/history.ts）—— 平台「对局记录」页与复盘页共用。
//
// 回归锚（这几种都真实出现过或极易写错）：
//   1) 玩家视角胜负必须与后端一致：卧底/狼人获胜时 `won=false` 但
//      `winner` 不是 AI 的 pid，不能再按 pid 猜成「胜」；`won` 缺省的旧记录
//      才回退 pid 比较；
//   2) 旧记录缺 seed / finished_at / teaching 时，详情行必须整体略去，
//      不能渲染出 "undefined"；
//   3) 「加载更多」在分页期间新打完一局时，重复记录不得渲染两次。
//
// 运行：cd platform-frontend && npm run test:frontend

import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  buildHistoryQuery,
  detailRows,
  difficultyLabel,
  durationLabel,
  gameLabel,
  matchResult,
  mergePage,
  movesLabel,
  relativeWhen,
  seatLabel,
  summarize,
  toCsv,
  variantLabel,
  winRateLabel,
  type MatchMetaLike,
} from '../src/history.ts'

function meta(overrides: Partial<MatchMetaLike> = {}): MatchMetaLike {
  return {
    match_id: 'm1',
    game_id: 'moon_chess',
    player_pid: 'p_black',
    difficulty: 'normal',
    winner: 'p_black',
    moves: 12,
    ...overrides,
  }
}

// ── 玩家视角胜负 ───────────────────────────────────────────────

test('matchResult：won=true → 胜', () => {
  const r = matchResult(meta({ won: true }))
  assert.equal(r.kind, 'win')
  assert.equal(r.badge, 'win')
})

test('matchResult：won=false 且 winner 非 AI pid（阵营胜者）→ 负，不按 pid 猜胜', () => {
  // 卧底获胜：winner='undercover'，玩家的 pid 与它不相等 → 后端已判 won=false。
  const r = matchResult(meta({ won: false, winner: 'undercover', player_pid: 'p0' }))
  assert.equal(r.kind, 'lose')
  assert.equal(r.badge, 'lose')
})

test('matchResult：旧记录缺 won → 回退 pid 比较', () => {
  assert.equal(matchResult(meta({ won: undefined, winner: 'p_black', player_pid: 'p_black' })).kind, 'win')
  assert.equal(matchResult(meta({ won: undefined, winner: 'p_white', player_pid: 'p_black' })).kind, 'lose')
})

test('matchResult：无胜者 → 平局（无徽标）', () => {
  const r = matchResult(meta({ winner: null, won: null }))
  assert.equal(r.kind, 'draw')
  assert.equal(r.badge, '')
})

// ── 标签 ───────────────────────────────────────────────────────

test('gameLabel：已知 id → 中文名；未知 id → 回退 id 本身（绝不空串）', () => {
  assert.equal(gameLabel('moon_chess'), '月亮棋')
  assert.equal(gameLabel('mahjong_international'), '国际麻将（国标）')
  assert.equal(gameLabel('my_custom_game'), 'my_custom_game')
  assert.equal(gameLabel(null), '')
})

test('seatLabel：后端 seat_names 优先 → 内置字典 → 座位N → pid', () => {
  assert.equal(seatLabel(meta({ seat_names: { p0: '东家' } }), 'p0'), '东家')
  assert.equal(seatLabel(meta(), 'p_white'), '白棋')
  assert.equal(seatLabel(meta(), 'p2'), '座位3')
  assert.equal(seatLabel(meta(), 'weird_pid'), 'weird_pid')
  assert.equal(seatLabel(meta(), ''), '')
})

test('difficultyLabel：普通 / 自适应（缺 ai_strength → 破折号）/ 未知档位', () => {
  assert.equal(difficultyLabel(meta({ difficulty: 'hard' })), '困难')
  assert.equal(difficultyLabel(meta({ difficulty: 'normal', adaptive: true, ai_strength: 400 })), '自适应 ⚙ 强度 400')
  assert.equal(difficultyLabel(meta({ difficulty: 'normal', adaptive: true })), '自适应 ⚙ 强度 —')
  assert.equal(difficultyLabel('custom_tier'), 'custom_tier')
})

test('variantLabel：字符串收录；缺省 → 空串', () => {
  assert.equal(variantLabel(meta({ variant: 'guangdong' })), 'guangdong')
  assert.equal(variantLabel(meta()), '')
})

test('movesLabel：列表 meta（数字）与复盘 MatchLog（数组）都能算步数', () => {
  assert.equal(movesLabel(meta({ moves: 12 })), '12 步')
  assert.equal(movesLabel(meta({ moves: [{}, {}, {}] })), '3 步')
})

test('durationLabel：正常时长 / 缺 finished_at / 时间是垃圾值 → 空串', () => {
  const base = { started_at: '2026-01-01T10:00:00Z' }
  assert.equal(durationLabel(meta({ ...base, finished_at: '2026-01-01T10:20:30Z' })), '20 分 30 秒')
  assert.equal(durationLabel(meta({ ...base, finished_at: '2026-01-01T10:00:30Z' })), '30 秒')
  assert.equal(durationLabel(meta({ ...base })), '')
  assert.equal(durationLabel(meta({ started_at: 'garbage', finished_at: 'garbage' })), '')
  // 结束早于开始（时钟回拨/脏记录）→ 不显示负数时长。
  assert.equal(durationLabel(meta({ started_at: '2026-01-01T10:10:00Z', finished_at: '2026-01-01T10:00:00Z' })), '')
})

test('relativeWhen：未来时间不显示负数（回退绝对时间）', () => {
  const future = new Date(Date.now() + 3_600_000).toISOString()
  assert.equal(relativeWhen(future), relativeWhen(future)) // 不抛异常
  assert.ok(!relativeWhen(future).includes('-'))
  assert.equal(relativeWhen(''), '')
})

// ── 详情行 ─────────────────────────────────────────────────────

test('detailRows：旧记录无 seed/teaching/finished_at → 这些行整体略去，不出 undefined', () => {
  const rows = detailRows(meta({ started_at: '2026-01-01T10:00:00Z' }))
  const labels = rows.map((r) => r.label)
  assert.ok(!labels.includes('随机种子'))
  assert.ok(!labels.includes('教学对局'))
  assert.ok(!labels.includes('结束时间'))
  assert.ok(!labels.includes('时长'))
  for (const row of rows) {
    assert.ok(!row.value.includes('undefined'), `${row.label} 不应渲染 undefined`)
    assert.ok(!row.value.includes('null'), `${row.label} 不应渲染 null`)
  }
})

test('detailRows：完整记录 → 关键元数据齐全（含中文性格名与人数）', () => {
  const rows = detailRows(
    meta({
      started_at: '2026-01-01T10:00:00Z',
      finished_at: '2026-01-01T10:20:00Z',
      seed: 43,
      family: 'mahjong',
      player_count: 4,
      variant: 'guangdong',
      persona: 'gentle',
      hinted: false,
      teaching: true,
      custom: false,
      ai_strength: 200,
    }),
  )
  const map = new Map(rows.map((r) => [r.label, r.value]))
  assert.equal(map.get('随机种子'), '43')
  assert.equal(map.get('规则族'), '麻将')
  assert.equal(map.get('人数'), '4 人')
  assert.equal(map.get('变体'), 'guangdong')
  assert.equal(map.get('性格'), '温柔陪伴')
  assert.equal(map.get('用过提示'), '否')
  assert.equal(map.get('教学对局'), '是')
  assert.equal(map.get('AI 强度'), '200')
  assert.ok(!map.has('自定义游戏')) // custom=false 不占一行
})

// ── 汇总 ───────────────────────────────────────────────────────

test('summarize：胜场按玩家视角统计，并按游戏分组', () => {
  const s = summarize([
    meta({ match_id: 'a', game_id: 'moon_chess', won: true }),
    meta({ match_id: 'b', game_id: 'moon_chess', won: false }),
    meta({ match_id: 'c', game_id: 'werewolf', won: true, winner: 'good', player_pid: 'p0' }),
    meta({ match_id: 'd', game_id: 'werewolf', winner: null, won: null }),
  ])
  assert.equal(s.plays, 4)
  assert.equal(s.wins, 2)
  assert.equal(s.winRate, 0.5)
  assert.equal(winRateLabel(s.winRate), '50%')
  assert.deepEqual(
    s.byGame.map((g) => [g.game_id, g.plays, g.wins]),
    [
      ['werewolf', 2, 1],
      ['moon_chess', 2, 1],
    ],
  )
})

test('summarize：无对局 → 胜率 0（不做除零）', () => {
  const s = summarize([])
  assert.equal(s.plays, 0)
  assert.equal(s.winRate, 0)
  assert.equal(winRateLabel(s.winRate), '0%')
})

// ── 查询串与分页 ───────────────────────────────────────────────

test('buildHistoryQuery：空参数省略，中文关键字 URL 编码', () => {
  assert.equal(buildHistoryQuery({}), '')
  assert.equal(buildHistoryQuery({ gameId: 'uno', result: '', q: '', since: '', limit: 20, offset: 0 }), 'game_id=uno&limit=20')
  assert.equal(buildHistoryQuery({ q: '东家' }), `q=${encodeURIComponent('东家')}`)
})

test('mergePage：按 match_id 去重（加载更多期间新打完一局不重复渲染）', () => {
  const first = [meta({ match_id: 'a' }), meta({ match_id: 'b' })]
  const next = [meta({ match_id: 'b' }), meta({ match_id: 'c' })]
  assert.deepEqual(
    mergePage(first, next).map((m) => m.match_id),
    ['a', 'b', 'c'],
  )
})

// ── CSV 导出 ───────────────────────────────────────────────────

test('toCsv：表头 + BOM + 逗号/换行值被引号包裹', () => {
  const csv = toCsv([meta({ match_id: 'a', game_id: 'custom,weird' })])
  const lines = csv.split('\r\n')
  assert.ok(csv.startsWith('\ufeff'))
  assert.ok(lines[0].replace(/^\ufeff/, '').startsWith('开始时间,游戏,变体'))
  assert.ok(csv.includes('"custom,weird"'))
  assert.equal(lines.length, 2)
})
