// 开局配置测试（chat/battleConfig.ts）：默认值 / 合并校验 / 无 LLM 解析。
//
// 这张卡的下线换来的是「工具调用带偏好」：后端 play_game 的 config 参数
// （params.config）经 mergeBattleConfig 落到 BattleConfig；断连时
// parseBattleOptions 顶上。两条入口共用同一份默认值与同一套合法性判定，
// 因此这里同时锁定「默认值 = 旧配置卡初值」这条不变量。

import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  battleConfigFor,
  defaultBattleConfig,
  mergeBattleConfig,
  parseBattleOptions,
} from '../src/chat/battleConfig.ts'

const MOON_CHESS = {
  game_id: 'moon_chess',
  display_name: '月亮棋',
  description: '',
  kind: 'board',
  board_size: 3,
  seat_options: ['p_black', 'p_white'],
  seat_label: '颜色',
  seat_names: { p_black: '黑棋', p_white: '白棋' },
  player_counts: [2],
  difficulties: ['easy', 'normal', 'hard'],
  solver_options: [],
  family: 'grid',
  custom: false,
  variant_themes: null,
}

const UNDERCOVER = {
  ...MOON_CHESS,
  game_id: 'undercover',
  display_name: '谁是卧底',
  kind: 'uno' as const,
  seat_options: ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
  player_counts: [8, 4, 5, 6, 7, 9, 10, 11, 12],
  variant_themes: ['fruit', 'food', 'animal', 'object', 'place', 'plant'],
  family: 'social',
}

test('defaultBattleConfig：等于旧开局配置卡的初值（含 adaptive=true）', () => {
  const cfg = defaultBattleConfig(UNDERCOVER)
  assert.equal(cfg.playerPid, 'random')
  assert.equal(cfg.difficulty, 'normal')
  assert.equal(cfg.theme, 'fruit') // variant_themes[0]
  assert.equal(cfg.playerCount, 8) // player_counts[0]
  assert.equal(cfg.persona, 'gentle')
  assert.equal(cfg.hintLevel, 'off')
  assert.equal(cfg.pacing, 'standard')
  assert.equal(cfg.adaptive, true)
  assert.equal(cfg.teaching, false)
})

test('defaultBattleConfig：无主题/目录未知时也有可用默认（不崩）', () => {
  const cfg = defaultBattleConfig(MOON_CHESS)
  assert.equal(cfg.theme, undefined)
  assert.equal(cfg.playerCount, 2)
  assert.deepEqual(defaultBattleConfig(null), { ...cfg, playerCount: 2 })
})

test('mergeBattleConfig：白名单键生效，非法值一律丢弃（fail-soft）', () => {
  const base = defaultBattleConfig(UNDERCOVER)
  const merged = mergeBattleConfig(
    base,
    {
      playerCount: 10,
      difficulty: 'hard',
      theme: 'food',
      playerPid: 'p3',
      persona: 'banter',
      hintLevel: 'specific',
      pacing: 'fast',
      adaptive: false,
      teaching: true,
    },
    UNDERCOVER,
  )
  assert.deepEqual(merged, {
    playerPid: 'p3',
    difficulty: 'hard',
    theme: 'food',
    playerCount: 10,
    persona: 'banter',
    hintLevel: 'specific',
    pacing: 'fast',
    adaptive: false,
    teaching: true,
  })

  // 游戏不支持的取值 → 逐项丢弃，其余照旧；从不抛错。
  const partial = mergeBattleConfig(
    base,
    { playerCount: 3, difficulty: 'insane', theme: 'space', playerPid: 'p_black', persona: 'grumpy' },
    UNDERCOVER,
  )
  assert.equal(partial.playerCount, base.playerCount)
  assert.equal(partial.difficulty, base.difficulty)
  assert.equal(partial.theme, base.theme)
  assert.equal(partial.playerPid, base.playerPid)
  assert.equal(partial.persona, base.persona)
})

test('mergeBattleConfig：布尔键只认 boolean（字符串不认），垃圾输入不炸', () => {
  const base = defaultBattleConfig(MOON_CHESS)
  assert.equal(mergeBattleConfig(base, { adaptive: 'false' }, MOON_CHESS).adaptive, true)
  assert.equal(mergeBattleConfig(base, { adaptive: false }, MOON_CHESS).adaptive, false)
  assert.equal(mergeBattleConfig(base, null, MOON_CHESS), base)
  assert.equal(mergeBattleConfig(base, 'nonsense', MOON_CHESS), base)
  assert.equal(mergeBattleConfig(base, { unknownKey: 1 }, MOON_CHESS).adaptive, true)
})

test('battleConfigFor：默认值 + 后端 config 合并（dispatch 开局入口）', () => {
  const cfg = battleConfigFor(UNDERCOVER, { playerCount: 4, difficulty: 'easy', teaching: true })
  assert.equal(cfg.playerCount, 4)
  assert.equal(cfg.difficulty, 'easy')
  assert.equal(cfg.teaching, true)
  assert.equal(cfg.persona, 'gentle') // 未指定 → 默认
  assert.equal(cfg.adaptive, true)
})

test('parseBattleOptions：人数（阿拉伯/中文数词）+ 难度 + 教学', () => {
  assert.deepEqual(parseBattleOptions('四个人、困难、教学对局，玩谁是卧底', UNDERCOVER), {
    playerCount: 4,
    difficulty: 'hard',
    teaching: true,
  })
  assert.deepEqual(parseBattleOptions('4人局简单模式', UNDERCOVER), {
    playerCount: 4,
    difficulty: 'easy',
  })
  assert.deepEqual(parseBattleOptions('八人局水果主题，要具体建议', UNDERCOVER), {
    playerCount: 8,
    theme: 'fruit',
    hintLevel: 'specific',
  })
})

test('parseBattleOptions：不支持的取值不填（宁可少配），认不出则不填', () => {
  // 谁是卧底不支持 3 人 → 人数丢；moon_chess 只支持 2 人、无主题 → 两项都丢。
  assert.deepEqual(parseBattleOptions('三人局、困难', UNDERCOVER), { difficulty: 'hard' })
  assert.deepEqual(parseBattleOptions('四人局、水果主题', MOON_CHESS), {})
  // 单纯“来一局”不产生 config（params 形状与旧版一致）。
  assert.deepEqual(parseBattleOptions('来一局', UNDERCOVER), {})
  assert.deepEqual(parseBattleOptions('', UNDERCOVER), {})
})

test('parseBattleOptions：自适应开关的否定式覆盖肯定式', () => {
  assert.equal(parseBattleOptions('自适应难度', MOON_CHESS).adaptive, true)
  assert.equal(parseBattleOptions('不要自适应', MOON_CHESS).adaptive, false)
})
