// Chat-first 本地意图分类（正则兜底）— 与后端 chat.py 的 fallback_intent 对齐。
// 用途：`/api/chat` 不可用（旧服务端 / 断连）时，前端仍能靠关键词把一句话
// 路由到平台动作；也用于快速指令 chips 的本地预判。意图契约见 types.ts。

import type { ChatIntent, ChatTurnResult } from '../types'

export interface LocalContext {
  games: { game_id: string; display_name: string; description?: string; aliases?: string[] }[]
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
// 「X 是什么/怎么玩」类知识问句 — 与后端 chat.py 的 _WHAT_IS_RE 对齐；
// description 是确定性数据，断连时也能零幻觉作答。
const WHAT_IS_RE = /(?:是什么|什么叫|什么游戏|怎么玩|怎么下|怎么打|规则|玩法|介绍一?下|简介)/

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

// 平台功能帮助主题 —— 离线简版（与后端 platform_knowledge.py 对齐，保持
// 同一组主题 key）。keywords 子串匹配、最长命中胜出；命中返回主题文案，
// 让“具体功能怎么用/在哪”类提问在断连时也能得到权威说明而不是泛泛总览。
interface HelpTopic {
  key: string
  keywords: string[]
  text: string
}

const HELP_TOPICS: HelpTopic[] = [
  {
    key: 'overview',
    keywords: ['有哪些功能', '功能介绍', '有什么功能', '功能列表', '怎么开始用', '怎么使用平台', '平台功能'],
    text: [
      'Gavis 平台总览：',
      '· 对话即操作 —— 说“玩月亮棋”开局、“下第2行第3列”落子、“继续上一局”恢复；',
      '· 面板 —— 创建游戏/设置/评测中心/在线学习/教学对局/LLM 配置/视觉识别；',
      '· 完整界面 = 大厅 + 对局 + 战绩 + 复盘 + 创建 + 设置 + 评测 + 在线学习 + LLM 配置。',
    ].join('\n'),
  },
  {
    key: 'play',
    keywords: ['怎么开局', '怎么开始游戏', '怎么开一局', '怎么开始一局', '开始新游戏', '开新对局', '新开一局', '怎么玩'],
    text: '说“玩月亮棋”“来一局德州扑克”或在大厅点游戏即可开局。平台有棋盘（月亮棋/随机五子棋）、德州扑克、麻将六变种（默认4人）、UNO 六变体（2-10人）、谁是卧底（4-12人）、狼人杀（9人社交推理）与自定义游戏；没指明游戏时助手会追问。',
  },
  {
    key: 'resume',
    keywords: ['怎么继续', '继续对局', '恢复对局', '接着玩', '接着下', '接着打', '如何继续'],
    text: '说“继续上一局”“接着玩”恢复进行中的对局；没有进行中对局时助手会说明并建议先开一局。',
  },
  {
    key: 'move',
    keywords: ['怎么落子', '怎么下棋', '怎么出牌', '怎么打牌', '怎么发言', '怎么操作', '操作方式', '怎么走子'],
    text: '对局中直接说动作（“下第2行第3列”“跟注”“打这张牌”“我是平民”）或点击棋盘/牌面；含糊或不合法的动作助手会给出当前合法清单；麻将/UNO/发言桌游建议直接点击操作。',
  },
  {
    key: 'hint',
    keywords: ['怎么要提示', '怎么提示', '要提示', '提示功能', '这步怎么走', '下一步怎么走', '如何提示', '怎么走这步', '教我走'],
    text: '对局中说“这步怎么走”“提示我”要提示，分三级：方向（direction）、具体（specific）、演示（demo）；提示基于玩家自己可见的局面计算，不泄露 AI 信息。',
  },
  {
    key: 'history',
    keywords: ['看战绩', '查战绩', '战绩在哪', '怎么查战绩', '对局记录', '历史记录', '怎么查记录'],
    text: '说“看战绩”查看最近对局（游戏/难度/胜负/手数）；完整表格在顶部「战绩」页。',
  },
  {
    key: 'review',
    keywords: ['怎么复盘', '如何复盘', '复盘功能', '复盘在哪', '回顾一下', '怎么回顾', '复盘一下'],
    text: '说“复盘上一局”拉完整走子时间线+关键节点（转折点/胜着/昏招）+改进建议；完整逐手回放在顶部「复盘」页。',
  },
  {
    key: 'create',
    keywords: ['怎么创建', '如何创建', '创建游戏', '新建游戏', '自定义游戏', '写规则', '规则翻译', '如何自定义'],
    text: '说“创建一个新游戏”或进顶部「创建游戏」：自然语言写规则→翻译→校验→规则族识别→直接可对弈；也可用模板给基础游戏改变体；识别不了的规则会明确提示而不是静默失败。',
  },
  {
    key: 'settings',
    keywords: ['怎么改难度', '如何改难度', '难度设置', '改变难度', '调难度', '难度', '声音', '主题设置', '怎么调设置'],
    text: '「设置」页可调 AI 难度（简单/正常/困难；麻将当前为固定启发式强度三档暂无差异）、自适应难度、声音/主题、教练开关；LLM 端点/模型/密钥在侧边栏「LLM 配置」。',
  },
  {
    key: 'platform',
    keywords: ['平台界面', '完整界面', '回平台', '平台首页', '回到大厅'],
    text: '说“打开平台界面”切回完整平台：大厅/对局/战绩/复盘/创建/设置/评测中心/在线学习/LLM 配置/我的画像。',
  },
  {
    key: 'benchmark',
    keywords: ['评测中心', '求解器对比', '对比求解器', '模拟对局', '评测功能', 'benchmark'],
    text: '「评测中心」发起 AI vs AI 短赛（双方交替先手消除先手优势），按注册表人数对局对比各求解器（MCTS/CFR/PPO/PSRO/Hybrid/MAAC/QMix/HAPPO 等）；页面可发起任务、看状态与结果。',
  },
  {
    key: 'learning',
    keywords: ['在线学习', '自动学习', '学习状态', '学习中心', '学习功能', '如何学习'],
    text: '「在线学习」收集真实对局中人类决策（按信息集）→ 候选经验对手模型 → 门禁短赛（固定种子、换边、20局）→ 不回归才发布；德州扑克默认启用，页面可手动 apply，服务端可开后台自动发布。',
  },
  {
    key: 'teaching',
    keywords: ['教学对局', '教练模式', '教学模式', '教练功能', '教练', '怎么教学'],
    text: '开局开「教练」开关：教练能看到你自己的牌并推理，每步走完对照“参考动作”（求解器在你座位算的真实走法）点评；三条红线——教练看得不比你多、教练脑与对手脑分离、参考动作不污染在线学习。',
  },
  {
    key: 'llm',
    keywords: ['llm', 'llm配置', '模型配置', '密钥', '大模型', 'ai配置', '怎么配置模型'],
    text: '侧边栏「LLM 配置」填端点/模型/密钥（OpenAI 兼容，默认本地 Ollama qwen3:8b）；保存即对聊天/翻译/社交 AI 生效；密钥只写不回显，清空恢复环境变量；等价环境变量 LLM_BASE_URL / LLM_MODEL / LLM_API_KEY。',
  },
  {
    key: 'vision',
    keywords: ['视觉识别', '拍照识别', '图片识别', '截图识别', '摄像头识别', '识别功能'],
    text: '视觉识别是独立应用：截图/拍照 → AI 识别棋盘或手牌 → 接求解器给出可执行动作；启动 python -m layer4_interface.frontend.vision.server（默认 8766 端口），走 DashScope qwen-vision，P2 计划并入平台。',
  },
]

function matchHelpTopic(text: string): string | null {
  const lowered = text.toLowerCase()
  let best: string | null = null
  let bestLen = 0
  for (const t of HELP_TOPICS) {
    for (const kw of t.keywords) {
      if (lowered.includes(kw) && kw.length > bestLen) {
        best = t.key
        bestLen = kw.length
      }
    }
  }
  return best
}

function findGame(text: string, games: LocalContext['games']): LocalContext['games'][number] | null {
  // 与后端 _find_game 对齐：display_name / game_id / 别名子串匹配，
  // 大小写不敏感（"uno" 命中 "UNO"），最长匹配胜出（"UNO 7-0" 优先
  // 于裸 "UNO"）——display_name 带括注（「UNO（经典）」）时短名也能命中。
  const lowered = text.toLowerCase()
  let best: LocalContext['games'][number] | null = null
  let bestLen = 0
  for (const g of games) {
    const names = [g.display_name, g.game_id, ...(g.aliases ?? [])]
    for (const name of names) {
      if (!name) continue
      const lname = name.toLowerCase()
      if (lowered.includes(lname) && lname.length > bestLen) {
        best = g
        bestLen = lname.length
      }
    }
  }
  return best
}

export function classifyLocal(text: string, ctx: LocalContext): ChatTurnResult {
  const game = findGame(text, ctx.games)
  const hasSession = Boolean(ctx.activeGameId)
  const chips = ctx.games.slice(0, 8).map((g) => g.display_name)

  // 「X 是什么/怎么玩/规则」→ 确定性知识回答（游戏目录 description，
  // 零幻觉）。必须先于 play —— “怎么下/怎么打”也含开局动词，语义却是
  // 问规则；未点名任何已注册游戏则维持原兜底（不猜、不编）。
  if (game && WHAT_IS_RE.test(text)) {
    const desc = game.description ?? ''
    // description 多以名字开头（“3×3 经典月亮棋：…”），避免复读式拼接
    const body = desc
      ? desc.includes(game.display_name)
        ? desc
        : `${game.display_name}：${desc}`
      : `${game.display_name}是平台支持的一款游戏。`
    return {
      intent: 'chat',
      text: `${body}\n想试一试的话，说“玩${game.display_name}”即可开局。`,
      mood: 'thinking',
      params: { game_id: game.game_id, chips: [`玩${game.display_name}`] },
    }
  }

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
  // 具体功能提问（未命中上面任一动作意图）→ 主题文档确定性回答（离线简版，
  // 与后端 platform_knowledge.match_platform_topic 对齐）。
  const helpTopic = matchHelpTopic(text)
  if (helpTopic) {
    const doc = HELP_TOPICS.find((t) => t.key === helpTopic)
    return {
      intent: 'help',
      text: doc ? doc.text : HELP_TEXT,
      mood: 'neutral',
      params: { topic: helpTopic },
    }
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
