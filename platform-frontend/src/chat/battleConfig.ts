// battleConfig — 开局配置的默认值 / 合并 / 无 LLM 解析（单一事实来源）。
//
// 背景：对话流里原有一张「开局配置卡」（SetupCard → BattleSetup 九项表单），
// 用户得逐项点选才能开局。现在偏好由**工具调用**带过来（后端 play_game 的
// 偏好参数 → ``params.config``），本模块负责：
//
// - defaultBattleConfig：= 现行表单初值（平台 BattleSetup 表单与对话开局共用
//   同一个来源，避免两处默认值慢慢漂移）；
// - mergeBattleConfig：只认白名单键并校验类型/枚举，非法项一律丢弃——
//   一句拿不准的话绝不该让开局失败（fail-soft），缺省即"照默认开"；
// - parseBattleOptions：后端 ``chat.py::_parse_battle_options`` 的孪生实现，
//   给无 LLM / 断连时的本地兜底用（"三人局、困难、教学对局" 也听得懂）。
//
// 取值一律用前端字段名（``playerCount`` / ``hintLevel`` …）：后端
// ``chat.py::_PLAY_CONFIG_FIELDS`` 已把工具参数名映射到同一组字段名，
// 因此 ``params.config`` 可以直接喂给 mergeBattleConfig。

import type { HintLevel, Pacing, PersonaKey } from '../types'
import type { BattleConfig } from '../components/BattleSetup'

/**
 * 开局相关的可配置面（只看这几个字段的最小形态）。
 *
 * ``GameInfo``（完整目录条目）与 ``LocalContext.games`` 的条目都能满足它，
 * 因此平台表单、对话开局与离线兜底三处可以共用同一套默认值与校验。
 */
export interface BattleGameOptions {
  player_counts?: number[]
  difficulties?: string[]
  seat_options?: string[]
  variant_themes?: string[] | null
}

export const PERSONA_VALUES: readonly PersonaKey[] = ['gentle', 'teacher', 'banter', 'cold']
export const HINT_LEVEL_VALUES: readonly HintLevel[] = ['off', 'direction', 'specific', 'demo']
export const PACING_VALUES: readonly Pacing[] = ['fast', 'standard', 'slow']

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function pick<T extends string>(value: unknown, allowed: readonly T[]): T | undefined {
  return typeof value === 'string' && (allowed as readonly string[]).includes(value) ? (value as T) : undefined
}

/** 现行表单初值（BattleSetup 的九项 useState 初值；对话开局同源）。 */
export function defaultBattleConfig(game: BattleGameOptions | null | undefined): BattleConfig {
  return {
    playerPid: 'random',
    difficulty: 'normal',
    theme: game?.variant_themes?.[0],
    playerCount: game?.player_counts?.[0] ?? 2,
    persona: 'gentle',
    hintLevel: 'off',
    pacing: 'standard',
    // 服务端缺省是 false（``/api/match/start`` 的 adaptive），而现行表单初值是
    // 开启 → 对话开局显式带上 true，体验与旧配置卡完全一致。
    adaptive: true,
    teaching: false,
  }
}

/**
 * 该取值对该游戏是否合法（合并与解析过滤的**同一判定**）。
 *
 * 游戏目录信息缺失时（离线、目录未加载）不做事后过滤，交给后端校验；
 * 布尔键严格取 boolean（不认 ``"false"`` 这类字符串）。
 */
function supports(key: keyof BattleConfig, value: unknown, game?: BattleGameOptions | null): boolean {
  const counts = game?.player_counts ?? []
  const tiers = game?.difficulties ?? []
  const seats = game?.seat_options ?? []
  const themes = game?.variant_themes ?? []
  switch (key) {
    case 'playerCount':
      return typeof value === 'number' && Number.isInteger(value) && (counts.length === 0 || counts.includes(value))
    case 'difficulty':
      return typeof value === 'string' && (tiers.length === 0 || tiers.includes(value))
    case 'theme':
      // 只有**声明了主题**的游戏才接受主题：后端见到 theme 就会按
      // ``f"{theme}_{tier}"`` 拼 variant 传给引擎，给无主题的游戏塞主题
      // 会变成一个非法变体。目录没给主题信息时也按"不支持"处理。
      return typeof value === 'string' && themes.includes(value)
    case 'playerPid':
      return typeof value === 'string' && (seats.length === 0 || seats.includes(value))
    case 'persona':
      return pick(value, PERSONA_VALUES) !== undefined
    case 'hintLevel':
      return pick(value, HINT_LEVEL_VALUES) !== undefined
    case 'pacing':
      return pick(value, PACING_VALUES) !== undefined
    case 'adaptive':
    case 'teaching':
      return typeof value === 'boolean'
    default:
      return false
  }
}

/**
 * 把（后端已校验的 / 本地解析出的）偏好合并进基础配置。
 *
 * 只认白名单键、只接受该游戏真的支持的取值：人数不在 ``player_counts``、
 * 难度不在 ``difficulties``、主题不在 ``variant_themes``、座位不在
 * ``seat_options`` 内的一律丢弃。
 */
export function mergeBattleConfig(base: BattleConfig, raw: unknown, game?: BattleGameOptions | null): BattleConfig {
  if (!isRecord(raw)) return base
  const out: BattleConfig = { ...base }
  for (const key of Object.keys(raw) as (keyof BattleConfig)[]) {
    const value = raw[key as string]
    if (value === undefined) continue
    if (supports(key, value, game)) (out as unknown as Record<string, unknown>)[key as string] = value
  }
  return out
}

/** 默认值 + 偏好合并（dispatch 开局的唯一入口）。 */
export function battleConfigFor(game: BattleGameOptions | null | undefined, raw?: unknown): BattleConfig {
  return mergeBattleConfig(defaultBattleConfig(game), raw, game)
}

// ── 无 LLM 兜底：从一句话里抠偏好 ────────────────────────────────

/** 中文数词 → 数字（"三人局" 与 "3 人" 等价；与后端同表）。 */
const CN_NUMERALS: Record<string, number> = {
  二: 2,
  两: 2,
  三: 3,
  四: 4,
  五: 5,
  六: 6,
  七: 7,
  八: 8,
  九: 9,
  十: 10,
  十一: 11,
  十二: 12,
}

const COUNT_RE = /(\d{1,2}|[二两三四五六七八九十]{1,3})\s*个?\s*人/

/**
 * 短语 → 偏好（与后端 ``_BATTLE_OPTION_RULES`` 同表：同样的措辞、同样的取值；
 * 后端写工具参数名、这里写前端字段名，一一对应）。按序覆盖，同键后命中者胜。
 * 刻意保守：只在措辞明确时命中（"标准节奏" 不该被当成难度档）。
 */
const OPTION_RULES: [keyof BattleConfig, unknown, RegExp][] = [
  ['difficulty', 'easy', /(?:简单|容易)/],
  ['difficulty', 'hard', /(?:困难|高难)/],
  ['difficulty', 'normal', /(?:普通|中等)/],
  ['teaching', true, /(?:教学对局|教学|教练|带我打|带打)/],
  ['persona', 'gentle', /(?:温柔|贴心|陪玩)/],
  ['persona', 'banter', /(?:吐槽|幽默|搞笑)/],
  ['persona', 'cold', /(?:高冷|严肃|竞技)/],
  ['hintLevel', 'demo', /(?:演示|示范)/],
  ['hintLevel', 'specific', /(?:具体建议|具体提示|详细建议|详细提示)/],
  ['hintLevel', 'direction', /(?:方向提示|给我方向|指个方向)/],
  ['hintLevel', 'off', /(?:关闭提示|不要提示|别提示)/],
  ['pacing', 'fast', /(?:快棋|快节奏|快点下|下快点)/],
  ['pacing', 'slow', /(?:慢棋|慢节奏|慢慢来)/],
  ['theme', 'fruit', /水果/],
  ['theme', 'food', /美食/],
  ['theme', 'animal', /动物/],
  ['theme', 'object', /物品/],
  ['theme', 'place', /地点/],
  ['theme', 'plant', /植物/],
  ['adaptive', true, /自适应/],
  ['adaptive', false, /(?:不要|不用|别|关闭).{0,4}自适应/],
]

/**
 * 从一句话里解析开局偏好（无 LLM 兜底）。
 *
 * 与后端 ``_parse_battle_options`` + ``_validated_play_config`` 同语义：
 * 解析出的值只在被该游戏支持时留下。认不出的偏好一律不填——宁可少配
 * （走默认开局），不可误配。只返回**非空**结果时调用方才附加 ``params.config``，
 * 以保持零偏好请求的参数形状与旧版一致。
 */
export function parseBattleOptions(text: string, game?: BattleGameOptions | null): Partial<BattleConfig> {
  const picked: Record<string, unknown> = {}
  if (!text) return {}
  const match = COUNT_RE.exec(text)
  if (match) {
    const raw = match[1]
    const count = /^\d+$/.test(raw) ? Number(raw) : CN_NUMERALS[raw]
    if (count !== undefined) picked.playerCount = count
  }
  for (const [key, value, pattern] of OPTION_RULES) {
    if (pattern.test(text)) picked[key as string] = value
  }
  const out: Partial<BattleConfig> = {}
  for (const key of Object.keys(picked) as (keyof BattleConfig)[]) {
    const value = picked[key as string]
    if (supports(key, value, game)) (out as unknown as Record<string, unknown>)[key as string] = value
  }
  return out
}
