// 对局元数据的单一来源（平台「对局记录」页 / 复盘页共用）。
//
// 背景：同一条对局记录此前在 HistoryPage 与 ReviewPage 里各有一份游戏名
// 字典、各写一遍「玩家视角胜负」判断，且历史页只知道 3 个游戏名。这里把
// **标签 + 判定 + 格式化 + 查询构造 + CSV** 收敛成纯函数（零 React / 零
// fetch 依赖），既让两个页面口径一致，也让 node --experimental-strip-types
// 下的 tests/history.test.ts 能直接覆盖（与 profile.ts / matchResult.ts 同款）。
//
// 契约：后端 GET /api/history 的 meta（见 layer4_interface/frontend/platform/
// history.py 的 record()）；未知字段一律缺省容忍——旧记录没有 seed /
// teaching / ai_strength，展示层必须「缺就不显示」而不是渲染 undefined。

import type { MatchMeta } from './types'

/**
 * 元数据函数的最小输入契约 —— `MatchMeta`（列表页）与 `MatchLog`（复盘页）
 * 都满足它，于是标签/判定/格式化函数两边共用，不必各写一遍或做类型断言。
 */
export interface MatchMetaLike {
  match_id: string
  game_id: string
  player_pid: string
  difficulty: string
  winner: string | null
  won?: boolean | null
  moves: number | { length: number }
  started_at?: string | null
  finished_at?: string | null
  persona?: string | null
  hinted?: boolean | null
  ai_strength?: number | null
  teaching?: boolean | null
  adaptive?: boolean | null
  seed?: number | null
  family?: string | null
  custom?: boolean | null
  player_count?: number | null
  variant?: string | null
  seat_names?: Record<string, string>
}

// ── 标签字典 ───────────────────────────────────────────────────

/** 注册表游戏 id → 中文显示名（与 layer4_interface/.../games.py 的 GAMES 对齐）。 */
export const GAME_LABELS: Record<string, string> = {
  moon_chess: '月亮棋',
  stochastic_gomoku: '随机五子棋',
  texas_holdem: '德州扑克',
  mahjong_guangdong: '广东麻将（鸡胡）',
  mahjong_hongzhong: '红中麻将',
  mahjong_blood: '血流成河',
  mahjong_sichuan: '四川麻将（血战到底）',
  mahjong_changsha: '长沙麻将（258将）',
  mahjong_taiwan: '台湾麻将（16张）',
  mahjong_international: '国际麻将（国标）',
  uno: 'UNO（经典）',
  uno_seven_zero: 'UNO 7-0（换手/移交）',
  uno_jump_in: 'UNO 抢牌',
  uno_stacking: 'UNO +2 叠加',
  uno_draw_until: 'UNO 摸到能打',
  uno_strict_wild4: 'UNO 严格+4',
  undercover: '谁是卧底',
  werewolf: '狼人杀',
}

/** 座位 pid → 中文称呼（后端 seat_names 缺失时的兜底；仅内置游戏常见座位）。 */
export const SEAT_LABELS: Record<string, string> = {
  p_black: '黑棋',
  p_white: '白棋',
  p_sb: '小盲位',
  p_bb: '大盲位',
}

export const DIFFICULTY_LABELS: Record<string, string> = { easy: '简单', normal: '普通', hard: '困难' }

/** 规则族 → 中文（`/api/history` 的 meta.family；旧记录缺省）。 */
export const FAMILY_LABELS: Record<string, string> = {
  grid: '棋类',
  poker: '扑克',
  mahjong: '麻将',
  social: '语言社交',
  uno: 'UNO',
}

/** 性格 key → 显示名（与 layer4_interface/agent 的 PERSONAS 对齐）。 */
export const PERSONA_LABELS: Record<string, string> = {
  gentle: '温柔陪伴',
  teacher: '认真教学',
  banter: '轻松吐槽',
  cold: '高冷竞技',
}

/** 游戏名（未知 id 回退 id 本身，绝不返回空串）。 */
export function gameLabel(gameId: string | null | undefined): string {
  const id = String(gameId ?? '')
  return GAME_LABELS[id] ?? id
}

/** 规则族中文名（未知/缺省 → 空串，调用方按「无此项」处理）。 */
export function familyLabel(family: string | null | undefined): string {
  const key = String(family ?? '')
  return FAMILY_LABELS[key] ?? key
}

/** 座位中文称呼：后端注入的 seat_names 优先，其次内置字典，最后 `座位N`。 */
export function seatLabel(meta: MatchMetaLike, pid: string): string {
  if (!pid) return ''
  const named = meta.seat_names?.[pid]
  if (named) return named
  if (SEAT_LABELS[pid]) return SEAT_LABELS[pid]
  const numbered = /^p(\d+)$/.exec(pid)
  return numbered ? `座位${Number(numbered[1]) + 1}` : pid
}

// ── 判定 ───────────────────────────────────────────────────────

export type ResultKind = 'win' | 'lose' | 'draw'

export interface MatchResult {
  kind: ResultKind
  label: string
  /** 结果徽标 class（与 global.css 的 .badge.win/.lose 对齐）。 */
  badge: string
}

/**
 * 玩家视角胜负 — 全平台唯一判定点。
 *
 * 口径：优先信后端已解析的 `won`（社交阵营胜者如卧底/狼人获胜已正确归边，
 * 见 layer4_interface/result.player_won）；旧记录缺省时按 pid 比较兜底；
 * 无胜者 → 平局。`won === false` 必须走 pid 兜底以外的「负」，不能被
 * `??` 的 truthiness 误判。
 */
export function matchResult(meta: MatchMetaLike): MatchResult {
  const winner = meta.winner ?? null
  if (winner == null) return { kind: 'draw', label: '平局', badge: '' }
  const won = meta.won ?? winner === meta.player_pid
  return won ? { kind: 'win', label: '胜利 🎉', badge: 'win' } : { kind: 'lose', label: '失败', badge: 'lose' }
}

// ── 格式化 ─────────────────────────────────────────────────────

/** 本地化时间（无效/缺失 → 空串）。 */
export function formatWhen(iso: string | null | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleString('zh-CN')
}

/** 相对时间（"3 分钟前" / "2 天前"；超 30 天回退到日期）。 */
export function relativeWhen(iso: string | null | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const diffMs = Date.now() - d.getTime()
  if (diffMs < 0) return formatWhen(iso)
  const minutes = Math.floor(diffMs / 60000)
  if (minutes < 1) return '刚刚'
  if (minutes < 60) return `${minutes} 分钟前`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours} 小时前`
  const days = Math.floor(hours / 24)
  if (days <= 30) return `${days} 天前`
  return formatWhen(iso).slice(0, 10)
}

/** 难度文案：自适应局显示实际强度档（旧记录缺 ai_strength → 「—」）。 */
export function difficultyLabel(meta: MatchMetaLike | string): string {
  if (typeof meta === 'string') return DIFFICULTY_LABELS[meta] ?? meta
  const base = DIFFICULTY_LABELS[meta.difficulty] ?? meta.difficulty ?? ''
  if (!meta.adaptive) return base
  return `自适应 ⚙ 强度 ${meta.ai_strength ?? '—'}`
}

/** 手数（列表 meta.moves 是步数；复盘 MatchLog.moves 是明细数组）。 */
export function movesLabel(meta: MatchMetaLike): string {
  const count = typeof meta.moves === 'number' ? meta.moves : meta.moves?.length
  return typeof count === 'number' ? `${count} 步` : ''
}

/** 对局时长（缺 finished_at / 时间无效 → 空串）。 */
export function durationLabel(meta: MatchMetaLike): string {
  const start = meta.started_at ? new Date(meta.started_at).getTime() : NaN
  const end = meta.finished_at ? new Date(meta.finished_at).getTime() : NaN
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return ''
  const total = Math.round((end - start) / 1000)
  if (total < 60) return `${total} 秒`
  const minutes = Math.floor(total / 60)
  const seconds = total % 60
  if (minutes < 60) return seconds ? `${minutes} 分 ${seconds} 秒` : `${minutes} 分钟`
  const hours = Math.floor(minutes / 60)
  return `${hours} 小时 ${minutes % 60} 分`
}

/** 变体名（后端 meta.variant；旧记录/无变体 → 空串）。 */
export function variantLabel(meta: MatchMetaLike): string {
  return typeof meta.variant === 'string' ? meta.variant : ''
}

export interface DetailRow {
  label: string
  value: string
}

/**
 * 展开详情行 —— **空值行整体略去**（旧记录不该出现「种子：undefined」）。
 */
export function detailRows(meta: MatchMetaLike): DetailRow[] {
  const rows: DetailRow[] = []
  const push = (label: string, value: unknown) => {
    if (value === null || value === undefined || value === '') return
    rows.push({ label, value: String(value) })
  }
  push('对局 ID', meta.match_id)
  push('开始时间', formatWhen(meta.started_at))
  push('结束时间', formatWhen(meta.finished_at))
  push('时长', durationLabel(meta))
  push('变体', variantLabel(meta))
  push('规则族', familyLabel(meta.family))
  push('人数', typeof meta.player_count === 'number' ? `${meta.player_count} 人` : '')
  push('随机种子', meta.seed)
  push('AI 强度', meta.ai_strength)
  push('性格', meta.persona ? (PERSONA_LABELS[meta.persona] ?? meta.persona) : '')
  push('用过提示', meta.hinted === true ? '是' : meta.hinted === false ? '否' : '')
  push('教学对局', meta.teaching === true ? '是' : meta.teaching === false ? '否' : '')
  push('自定义游戏', meta.custom === true ? '是' : '')
  return rows
}

// ── 汇总 ───────────────────────────────────────────────────────

export interface GameTally {
  game_id: string
  label: string
  plays: number
  wins: number
}

export interface MatchSummary {
  plays: number
  wins: number
  /** 胜率（0..1；无对局 → 0）。 */
  winRate: number
  byGame: GameTally[]
}

/** 汇总当前已加载集合（顶部战绩条 + 游戏筛选下拉都用它）。 */
export function summarize(matches: MatchMeta[]): MatchSummary {
  let wins = 0
  const per = new Map<string, GameTally>()
  for (const m of matches) {
    const result = matchResult(m)
    if (result.kind === 'win') wins += 1
    const gameId = String(m.game_id ?? '')
    const tally = per.get(gameId) ?? { game_id: gameId, label: gameLabel(gameId), plays: 0, wins: 0 }
    tally.plays += 1
    if (result.kind === 'win') tally.wins += 1
    per.set(gameId, tally)
  }
  const plays = matches.length
  return {
    plays,
    wins,
    winRate: plays > 0 ? wins / plays : 0,
    byGame: [...per.values()].sort((a, b) => b.plays - a.plays || a.label.localeCompare(b.label, 'zh-CN')),
  }
}

/** 胜率百分比文案（「62%」）。 */
export function winRateLabel(winRate: number): string {
  return `${Math.round(winRate * 100)}%`
}

// ── 查询与分页 ─────────────────────────────────────────────────

export interface HistoryQuery {
  gameId?: string | null
  result?: ResultKind | '' | null
  q?: string | null
  since?: string | null
  limit?: number
  offset?: number
}

/** 构造 `/history` 查询串（空参数与零值一律省略，后端按默认值处理）。 */
export function buildHistoryQuery(query: HistoryQuery): string {
  const parts: string[] = []
  const add = (key: string, value: string | number | null | undefined) => {
    if (value === null || value === undefined || value === '' || value === 0) return
    parts.push(`${key}=${encodeURIComponent(String(value))}`)
  }
  add('game_id', query.gameId)
  add('result', query.result)
  add('q', query.q)
  add('since', query.since)
  add('limit', query.limit)
  add('offset', query.offset)
  return parts.join('&')
}

/** 按 match_id 去重合并两页（「加载更多」期间新打完一局不会重复渲染）。 */
export function mergePage(prev: MatchMeta[], next: MatchMeta[]): MatchMeta[] {
  const seen = new Set(prev.map((m) => m.match_id))
  const merged = [...prev]
  for (const m of next) {
    if (!m || seen.has(m.match_id)) continue
    seen.add(m.match_id)
    merged.push(m)
  }
  return merged
}

// ── 导出 ───────────────────────────────────────────────────────

const CSV_HEADERS = [
  '开始时间',
  '游戏',
  '变体',
  '你的座位',
  '难度',
  'AI强度',
  '步数',
  '时长',
  '结果',
  '随机种子',
  '性格',
  '用过提示',
  '教学对局',
  '对局ID',
]

function csvCell(value: unknown): string {
  const text = value === null || value === undefined ? '' : String(value)
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

/** 当前筛选结果 → CSV 文本（含 BOM，Excel 打开中文不乱码）。 */
export function toCsv(matches: MatchMetaLike[]): string {
  const lines = [CSV_HEADERS.join(',')]
  for (const m of matches) {
    const yesNo = (v: boolean | null | undefined) => (v === true ? '是' : v === false ? '否' : '')
    lines.push(
      [
        formatWhen(m.started_at),
        gameLabel(m.game_id),
        variantLabel(m),
        seatLabel(m, m.player_pid),
        difficultyLabel(m),
        m.ai_strength ?? '',
        m.moves ?? '',
        durationLabel(m),
        matchResult(m).label,
        m.seed ?? '',
        m.persona ? (PERSONA_LABELS[m.persona] ?? m.persona) : '',
        yesNo(m.hinted),
        yesNo(m.teaching),
        m.match_id,
      ]
        .map(csvCell)
        .join(','),
    )
  }
  return `\ufeff${lines.join('\r\n')}`
}
