// 本地意图分类测试（chat/intents.ts 正则兜底，与后端 chat.py fallback_intent 对齐）。
//
// 场景：/api/chat 不可用（旧服务端 / 断连）时，前端靠关键词把一句话路由到
// 平台动作。这里锁定每个正则分支的返回值，防止兜底分类漂移。

import assert from 'node:assert/strict'
import { test } from 'node:test'

import { classifyLocal } from '../src/chat/intents.ts'

const GAMES = [
  { game_id: 'moon_chess', display_name: '月亮棋' },
  { game_id: 'stochastic_gomoku', display_name: '随机五子棋' },
  { game_id: 'texas_holdem', display_name: '德州扑克' },
]

const NO_SESSION = { games: GAMES, activeGameId: null, activeDisplay: null }
const WITH_SESSION = { games: GAMES, activeGameId: 'sess-1', activeDisplay: '月亮棋' }

test('play：点名游戏 → intent play + game_id', () => {
  const r = classifyLocal('我想玩月亮棋', NO_SESSION)
  assert.equal(r.intent, 'play')
  assert.equal(r.params.game_id, 'moon_chess')
  assert.equal(r.mood, 'happy')
})

test('play：没点名 → clarify + chips（游戏目录前 8 个显示名）', () => {
  const r = classifyLocal('来一局', NO_SESSION)
  assert.equal(r.intent, 'clarify')
  const chips = r.params.chips as string[]
  assert.ok(Array.isArray(chips))
  assert.ok(chips.includes('月亮棋'))
})

test('play：点名但无该游戏关键词 → 不误报 play', () => {
  const r = classifyLocal('帮我看看', NO_SESSION)
  assert.notEqual(r.intent, 'play')
})

test('hint：需要对局上下文；无对局时回落 chat', () => {
  assert.equal(classifyLocal('这步怎么走', WITH_SESSION).intent, 'hint')
  assert.notEqual(classifyLocal('这步怎么走', NO_SESSION).intent, 'hint')
})

test('resume：有活跃对局 → resume + game_id', () => {
  const r = classifyLocal('继续上一局', WITH_SESSION)
  assert.equal(r.intent, 'resume')
  assert.equal(r.params.game_id, 'sess-1')
})

test('restart：有活跃对局 → restart + game_id', () => {
  const r = classifyLocal('再来一局', WITH_SESSION)
  assert.equal(r.intent, 'restart')
  assert.equal(r.params.game_id, 'sess-1')
})

test('功能面板意图：history / review / create / settings / platform / benchmark / learning', () => {
  assert.equal(classifyLocal('看看我的战绩', NO_SESSION).intent, 'history')
  assert.equal(classifyLocal('复盘一下上一局', NO_SESSION).intent, 'review')
  assert.equal(classifyLocal('创建一个新游戏', NO_SESSION).intent, 'create')
  assert.equal(classifyLocal('打开设置', NO_SESSION).intent, 'settings')
  assert.equal(classifyLocal('打开平台界面', NO_SESSION).intent, 'platform')
  assert.equal(classifyLocal('看评测中心', NO_SESSION).intent, 'benchmark')
  assert.equal(classifyLocal('在线学习状态', NO_SESSION).intent, 'learning')
})

test('help：问到能力 → help 且带帮助长文案', () => {
  const r = classifyLocal('你能做什么', NO_SESSION)
  assert.equal(r.intent, 'help')
  assert.ok(r.text.includes('玩月亮棋'))
})

test('默认：识别不出的闲聊 → chat（不抛异常）', () => {
  const r = classifyLocal('你好呀', NO_SESSION)
  assert.equal(r.intent, 'chat')
  assert.ok(r.text.length > 0)
})

test('priority：点名游戏优先于功能面板关键词', () => {
  const r = classifyLocal('玩德州扑克能看胜率吗', NO_SESSION)
  assert.equal(r.intent, 'play')
  assert.equal(r.params.game_id, 'texas_holdem')
})