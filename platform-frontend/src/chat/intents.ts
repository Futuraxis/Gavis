// Chat-first 本地意图分类（正则兜底）— 与后端 chat.py 的 fallback_intent 对齐。
// 用途：`/api/chat` 不可用（旧服务端 / 断连）时，前端仍能靠关键词把一句话
// 路由到平台动作；也用于快速指令 chips 的本地预判。意图契约见 types.ts。

import type { ChatIntent, ChatTurnResult } from '../types'

export interface LocalContext {
  games: { game_id: string; display_name: string }[]
  activeGameId: string | null
  activeDisplay: string | null
}

const PLAY_RE = /(?:玩|来一局|来一把|下|打|开局|对战|加入|开一局)/
const HINT_RE = /(?:提示|怎么走|这步为什么|帮我想|下一步)/
const RESTART_RE = /(?:再来一局|重来|重新|重开|换一局)/
const RESUME_RE = /(?:继续|接着|恢复|回到) *(?:上一局|对战|对局|游戏)/
const HISTORY_RE = /(?:战绩|历史|记录|胜率|输赢|数据)/
const REVIEW_RE = /(?:复盘|回放|重看)/
const CREATE_RE = /(?:创建|新建|自定义|设计一?个新?游戏)/
const SETTINGS_RE = /(?:设置|性格|声音|主题|偏好|选项)/
const PLATFORM_RE = /(?:平台界面|完整界面|平台模式|打开平台|回去|回平台)/
const BENCHMARK_RE = /(?:评测|benchmark|模拟对局|求解器对比)/
const LEARNING_RE = /(?:在线学习|学习状态|自动学习)/
const HELP_RE = /(?:帮助|能做什么|怎么用|你有什么功能|你会什么)/

const HELP_TEXT = [
  '你可以直接用大白话跟我说话，例如：',
  '· “玩月亮棋” / “来一局德州扑克” —— 开对局',
  '· “继续上一局” —— 恢复进行中的对局',
  '· 对局中：“这步怎么走” / “提示我”',
  '· “看战绩” / “复盘上一局”',
  '· “创建一个新游戏” —— 用自然语言写规则',
  '· “打开平台界面” —— 切回完整界面',
  '· “设置” / “评测中心” / “在线学习” —— 各功能面板',
].join('\n')

function findGame(text: string, games: LocalContext['games']): { game_id: string; display_name: string } | null {
  let best: { game_id: string; display_name: string } | null = null
  let bestLen = 0
  for (const g of games) {
    for (const key of ['display_name', 'game_id'] as const) {
      const name = g[key]
      if (name && text.includes(name) && name.length > bestLen) {
        best = g
        bestLen = name.length
      }
    }
  }
  return best
}

export function classifyLocal(text: string, ctx: LocalContext): ChatTurnResult {
  const game = findGame(text, ctx.games)
  const hasSession = Boolean(ctx.activeGameId)
  const chips = ctx.games.slice(0, 8).map((g) => g.display_name)

  if (game && PLAY_RE.test(text)) {
    return { intent: 'play', text: `好，来一局${game.display_name}！对局正在创建…`, mood: 'happy', params: { game_id: game.game_id } }
  }
  if (HINT_RE.test(text) && hasSession) {
    return { intent: 'hint', text: '这一步的思路是…', mood: 'thinking', params: {} }
  }
  if (PLATFORM_RE.test(text)) {
    return { intent: 'platform', text: '已为你打开完整平台界面 👇', mood: 'neutral', params: {} }
  }
  if (CREATE_RE.test(text)) {
    return { intent: 'create', text: '创建游戏面板已为你展开 👇', mood: 'neutral', params: {} }
  }
  if (REVIEW_RE.test(text)) {
    return { intent: 'review', text: '复盘已为你展开 👇', mood: 'neutral', params: {} }
  }
  if (HISTORY_RE.test(text)) {
    return { intent: 'history', text: '这是你最近的战绩 👇', mood: 'neutral', params: {} }
  }
  if (SETTINGS_RE.test(text)) {
    return { intent: 'settings', text: '设置面板已为你展开 👇', mood: 'neutral', params: {} }
  }
  if (BENCHMARK_RE.test(text)) {
    return { intent: 'benchmark', text: '评测中心已为你展开 👇', mood: 'neutral', params: {} }
  }
  if (LEARNING_RE.test(text)) {
    return { intent: 'learning', text: '在线学习状态已为你展开 👇', mood: 'neutral', params: {} }
  }
  if (hasSession && RESTART_RE.test(text)) {
    return { intent: 'restart', text: '好，重新开一局！', mood: 'happy', params: { game_id: ctx.activeGameId } }
  }
  if (RESUME_RE.test(text) && hasSession) {
    return { intent: 'resume', text: `继续对局「${ctx.activeDisplay ?? ctx.activeGameId}」！`, mood: 'happy', params: { game_id: ctx.activeGameId } }
  }
  if (HELP_RE.test(text)) {
    return { intent: 'help', text: HELP_TEXT, mood: 'neutral', params: {} }
  }
  if (PLAY_RE.test(text)) {
    return { intent: 'clarify', text: '想玩哪一款？', mood: 'neutral', params: { chips } }
  }
  return {
    intent: 'chat',
    text: '我在的，你可以试试：“玩月亮棋”“看战绩”“这步怎么走”…',
    mood: 'neutral',
    params: {},
  }
}

export type { ChatIntent }