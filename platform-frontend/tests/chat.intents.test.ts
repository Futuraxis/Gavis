// 本地意图分类测试（chat/intents.ts 正则兜底，与后端 chat.py fallback_intent 对齐）。
//
// 场景：/api/chat 不可用（旧服务端 / 断连）时，前端靠关键词把一句话路由到
// 平台动作。这里锁定每个正则分支的返回值，防止兜底分类漂移。

import assert from 'node:assert/strict'
import { test } from 'node:test'

import { classifyLocal } from '../src/chat/intents.ts'

const GAMES = [
  { game_id: 'moon_chess', display_name: '月亮棋', description: '3×3 经典月亮棋：三子连珠即胜，棋盘满时最旧的棋子被挤出。' },
  { game_id: 'stochastic_gomoku', display_name: '随机五子棋', description: '9×9 五子棋变体：每次落子后棋子有 50% 概率被随机抹去。' },
  { game_id: 'texas_holdem', display_name: '德州扑克', description: '双人德州扑克：翻前/翻牌/转牌/河牌四轮下注，AI 使用混合求解器。' },
]

// UNO 短名匹配的问题形态：display_name 带括注/空格，靠 aliases + 大小写
// 不敏感命中（与后端 game_knowledge.GAME_ALIASES 对齐）。
const UNO_GAMES = [
  { game_id: 'uno', display_name: 'UNO（经典）', description: '四人经典 UNO：108 张牌，同色或同符号接牌，先清空手牌者胜。', aliases: ['UNO', '优诺'] },
  { game_id: 'uno_seven_zero', display_name: 'UNO 7-0（换手/移交）', description: 'UNO 7-0 变体：打出 7 可与任一玩家换手。', aliases: ['UNO 7-0', 'UNO7-0', 'UNO 70', '换手'] },
]

const NO_SESSION = { games: GAMES, activeGameId: null, activeDisplay: null }
const WITH_SESSION = { games: GAMES, activeGameId: 'sess-1', activeDisplay: '月亮棋' }
const NO_SESSION_UNO = { games: UNO_GAMES, activeGameId: null, activeDisplay: null }

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

test('help 主题：具体功能提问 → 主题文档（与后端 platform_knowledge 对齐）', () => {
  const r = classifyLocal('怎么改难度', NO_SESSION)
  assert.equal(r.intent, 'help')
  assert.equal(r.params.topic, 'settings')
  assert.ok(r.text.includes('难度'))
  const r2 = classifyLocal('教学对局是什么', NO_SESSION)
  assert.equal(r2.intent, 'help')
  assert.equal(r2.params.topic, 'teaching')
  assert.ok(r2.text.includes('教练'))
  const r3 = classifyLocal('视觉识别怎么用', NO_SESSION)
  assert.equal(r3.intent, 'help')
  assert.equal(r3.params.topic, 'vision')
  assert.ok(r3.text.includes('视觉识别'))
  const r4 = classifyLocal('LLM 模型在哪配置', NO_SESSION)
  assert.equal(r4.intent, 'help')
  assert.equal(r4.params.topic, 'llm')
  assert.ok(r4.text.includes('密钥'))
  // 泛泛“你能做什么”不命中主题 → 维持原总览文案
  const r5 = classifyLocal('你能做什么', NO_SESSION)
  assert.equal(r5.intent, 'help')
  assert.ok(!r5.params.topic)
  assert.ok(r5.text.includes('玩月亮棋'))
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

test('what-is：问规则 → chat + 权威简介 + “来一局” chips（audit §5-2）', () => {
  const r = classifyLocal('月亮棋是什么', NO_SESSION)
  assert.equal(r.intent, 'chat')
  assert.equal(r.params.game_id, 'moon_chess')
  assert.ok(r.text.includes('3×3')) // 来自游戏目录 description（确定性数据）
  assert.deepEqual(r.params.chips, ['玩月亮棋'])
})

test('what-is：先于 play —— “怎么下”含开局动词但语义是问规则', () => {
  const r = classifyLocal('月亮棋怎么下', NO_SESSION)
  assert.equal(r.intent, 'chat')
  assert.ok(r.text.includes('3×3'))
})

test('what-is：不点名已注册游戏 → 维持原兜底（不猜、不编）', () => {
  const r = classifyLocal('围棋是什么', NO_SESSION)
  assert.equal(r.intent, 'chat')
  assert.ok(!r.params.game_id)
  assert.ok(!r.text.includes('3×3'))
})

test('alias：UNO 短名 + 大小写不敏感命中（audit §5-5）', () => {
  const r = classifyLocal('UNO的规则', NO_SESSION_UNO)
  assert.equal(r.intent, 'chat')
  assert.equal(r.params.game_id, 'uno')
  assert.ok(r.text.includes('108'))
  const r2 = classifyLocal('玩uno', NO_SESSION_UNO)
  assert.equal(r2.intent, 'play')
  assert.equal(r2.params.game_id, 'uno')
})

test('alias：最长匹配胜出 —— “UNO 7-0” 优先于裸 “UNO”', () => {
  const r = classifyLocal('UNO 7-0怎么玩', NO_SESSION_UNO)
  assert.equal(r.params.game_id, 'uno_seven_zero')
})
