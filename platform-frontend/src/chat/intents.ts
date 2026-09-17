// Chat-first 本地意图分类（正则兜底）— 与后端 chat.py 的 fallback_intent 对齐。
// 用途：`/api/chat` 不可用（旧服务端 / 断连）时，前端仍能靠关键词把一句话
// 路由到平台动作；也用于快速指令 chips 的本地预判。意图契约见 types.ts。

import type { ChatIntent, ChatTurnResult } from '../types'
// 显式 .ts 扩展：本模块会被 node --experimental-strip-types 直接跑（前端离线
// 单测），extensionless specifier 在 Node ESM 下解析不到（Vite 侧无影响）。
import { parseBattleOptions } from './battleConfig.ts'

export interface LocalContext {
  games: {
    game_id: string
    display_name: string
    description?: string
    aliases?: string[]
    // 可配置面（可选：断连兜底时目录可能不全）。缺省则不做事后过滤，
    // 交给后端校验——离线路径也要能听懂“三人局、困难、教学对局”。
    player_counts?: number[]
    difficulties?: string[]
    seat_options?: string[]
    variant_themes?: string[] | null
  }[]
  activeGameId: string | null
  activeDisplay: string | null
}

const PLAY_RE = /(?:玩|来一局|来一把|下|打|开局|对战|加入|开一局)/
const HINT_RE = /(?:提示|怎么走|这步为什么|帮我想|下一步)/
const RESTART_RE = /(?:再来一局|重来|重新|重开|换一局)/
const RESUME_RE = /(?:继续|接着|恢复|回到) *(?:上一局|对战|对局|游戏)/
const HISTORY_RE = /(?:战绩|历史|记录|胜率|输赢|数据)/
const REVIEW_RE = /(?:复盘|回放|重看)/
// 「做一个…游戏 / 写个…游戏」与「创建」等价（口述规则的常见说法；
// 与后端 chat.py 的 _CREATE_RE 同表）。
const CREATE_RE = /(?:创建|新建|自定义|设计一?个新?游戏|(?:做|写|弄|生成|搞)一?(?:个|款|套).{0,12}游戏)/
const SETTINGS_RE = /(?:设置|性格|声音|主题|偏好|选项)/
const PLATFORM_RE = /(?:平台界面|完整界面|平台模式|打开平台|回去|回平台)/
// 「换风格 / 改设置」——与后端 chat.py 的 `_preference_result` 同表同口径：
// 明确说出取值 → settings + params.applied（useChatRuntime 写档案并回执，
// **不跳页**）；只说要换、没说成哪种 → clarify + 选项 chips。绝不替用户猜，
// 也绝不把人从对话里静默甩到设置页。
const OPEN_SETTINGS_RE = /(?:打开|进入|去|切到|回到|看看)\s*(?:一下)?\s*(?:设置|偏好)页?/
const PREFERENCE_CHANGE_RE = /(?:换|改|调|变|设置|来)(?:成|一个|个|一下|点)?/
const PERSONA_WORDS = ['性格', '人设', '风格', '语气', '口气', '人格', '说话方式']
const THEME_WORDS = ['主题', '外观', '配色']
const PERSONA_ASK_RE = /(?:温柔|贴心|陪玩|吐槽|幽默|搞笑|高冷|严肃|竞技|认真|老师).{0,2}(?:点|一点|一些|些)/
// 人格取值识别（“认真/教学”先判，避免“认真温柔”这类叠加句落到 gentle）。
const PERSONA_VALUE_RULES: [string, RegExp][] = [
  ['teacher', /(?:认真|教学|老师|讲道理)/],
  ['gentle', /(?:温柔|贴心|陪玩)/],
  ['banter', /(?:吐槽|幽默|搞笑)/],
  ['cold', /(?:高冷|严肃|竞技)/],
]
const THEME_VALUE_RULES: [string, RegExp][] = [
  ['dark', /(?:深色|暗色|夜间|黑夜|黑色主题|黑主题)/],
  ['light', /(?:浅色|亮色|日间|白色主题|白主题)/],
]
// 选项 chips（点一下 = 当作一句话发回来 → 命中上面的取值规则）。
const PERSONA_STYLE_CHIPS = ['换成温柔陪伴', '换成认真教学', '换成轻松吐槽', '换成高冷竞技']
const THEME_CHIPS = ['换成深色主题', '换成浅色主题']
const OPEN_SETTINGS_CHIP = '打开设置页'
const PREFERENCE_CLARIFY_TEXT = '想换成哪种？挑一个我立刻改；也可以直接打开设置页自己调。'
// profile 字段 → 中文名 / 取值显示名（回执文案与后端 `_settings_applied_result` 同形）。
const SETTING_FIELD_LABELS: Record<string, string> = {
  default_persona: '助手性格',
  hint_level: '提示档',
  default_difficulty: 'AI 难度',
  theme: '界面主题',
}
const SETTING_VALUE_LABELS: Record<string, Record<string, string>> = {
  default_persona: { gentle: '温柔陪伴', teacher: '认真教学', banter: '轻松吐槽', cold: '高冷竞技' },
  hint_level: { off: '关闭', direction: '方向提示', specific: '具体建议', demo: '演示' },
  default_difficulty: { easy: '简单', normal: '普通', hard: '困难', adaptive: '自适应' },
  theme: { light: '浅色', dark: '深色' },
}
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
  '· “创建一个新游戏” —— 直接说规则，我帮你生成（也可进「创建游戏」页）',
  '· “换个风格” / “换成高冷竞技” —— 换助手性格（立刻生效）',
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
    text: '对话里直接描述规则（“做一个 8×8 四子连珠的游戏，叫四子棋”）即可建好，成功后可立刻说“玩四子棋”；也可进顶部「创建游戏」页用表单（含模板变体、LLM 翻译开关与自定义游戏管理）。走法仍是：翻译→校验→规则族识别→直接可对弈；识别不了的规则会明确提示而不是静默失败。',
  },
  {
    key: 'settings',
    keywords: ['怎么改难度', '如何改难度', '难度设置', '改变难度', '调难度', '难度', '声音', '主题设置', '怎么调设置'],
    text: '助手性格可以直接在对话里换：说“换成温柔陪伴/认真教学/轻松吐槽/高冷竞技”（或“换个风格”让平台给选项）立刻生效并写进你的档案。说“打开设置”或进顶部「设置」页：可调 AI 难度（简单/正常/困难；麻将当前为固定启发式强度三档暂无差异）、自适应难度、声音/主题、教练开关；LLM 端点/模型/密钥在侧边栏「LLM 配置」。',
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

/** 已校验的偏好变更 → settings 意图（runtime 写档案 + 回执，不跳页）. */
function appliedResult(applied: Record<string, string>): ChatTurnResult {
  const detail = Object.entries(applied)
    .map(([field, value]) => {
      const label = SETTING_FIELD_LABELS[field] ?? field
      const shown = SETTING_VALUE_LABELS[field]?.[value] ?? value
      return `${label}改成「${shown}」`
    })
    .join('、')
  const lines = [`好，${detail} ✓`]
  if (applied.default_persona) lines.push('以后我说话就按这个风格来。')
  return {
    intent: 'settings',
    text: lines.join('\n'),
    mood: 'happy',
    params: { applied, chips: [OPEN_SETTINGS_CHIP] },
  }
}

/**
 * 「换风格 / 改设置」类表达 → 真改偏好 / 给选项；与偏好无关返回 null。
 *
 * 修复的 UX 事故：问「你能换个风格吗」时旧逻辑命中 `SETTINGS_RE` 的
 * `性格`/`主题` 关键词 → 直接切到平台设置页，问题没回答、设置一个没改。
 * 现在先说清"换成哪种"，用户点一下 chips 才真正落变更。
 */
function preferenceResult(text: string): ChatTurnResult | null {
  if (OPEN_SETTINGS_RE.test(text)) {
    return { intent: 'settings', text: '设置面板已为你展开 👇', mood: 'neutral', params: { open_page: true } }
  }
  const change = PREFERENCE_CHANGE_RE.test(text)
  const personaValue = PERSONA_VALUE_RULES.find(([, re]) => re.test(text))?.[0] ?? ''
  const themeValue = THEME_VALUE_RULES.find(([, re]) => re.test(text))?.[0] ?? ''
  const wantsPersona =
    PERSONA_WORDS.some((w) => text.includes(w)) || PERSONA_ASK_RE.test(text) || (change && personaValue !== '')
  const wantsTheme = THEME_WORDS.some((w) => text.includes(w)) || (change && themeValue !== '')
  if (!wantsPersona && !wantsTheme) return null
  const applied: Record<string, string> = {}
  if (wantsPersona && personaValue) applied.default_persona = personaValue
  if (wantsTheme && themeValue) applied.theme = themeValue
  if (Object.keys(applied).length > 0) return appliedResult(applied)
  const chips = wantsPersona ? [...PERSONA_STYLE_CHIPS] : [...THEME_CHIPS]
  chips.push(OPEN_SETTINGS_CHIP)
  return { intent: 'clarify', text: PREFERENCE_CLARIFY_TEXT, mood: 'thinking', params: { chips } }
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
    // 开局偏好：与后端 fallback_intent 同一张表（battleConfig.parseBattleOptions）
    // —— 断连时“三人局、困难、教学对局”同样落到 params.config。
    const config = parseBattleOptions(text, game)
    return {
      intent: 'play',
      text: `好，来一局${game.display_name}！对局正在创建…`,
      mood: 'happy',
      params: {
        game_id: game.game_id,
        ...(Object.keys(config).length > 0 ? { config } : {}),
      },
    }
  }
  if (HINT_RE.test(text) && hasSession) {
    return { intent: 'hint', text: '这一步的思路是…', mood: 'thinking', params: {} }
  }
  if (PLATFORM_RE.test(text)) {
    return { intent: 'platform', text: '已为你打开完整平台界面 👇', mood: 'neutral', params: {} }
  }
  if (CREATE_RE.test(text)) {
    // 无 LLM 兜底不臆造规则：切到平台创建游戏页（对话里的创建由 create_game 工具完成）。
    return { intent: 'create', text: '已为你打开创建游戏页 👇', mood: 'neutral', params: {} }
  }
  if (REVIEW_RE.test(text)) {
    return { intent: 'review', text: '复盘已为你展开 👇', mood: 'neutral', params: {} }
  }
  if (HISTORY_RE.test(text)) {
    return { intent: 'history', text: '这是你最近的战绩 👇', mood: 'neutral', params: {} }
  }
  const preference = preferenceResult(text)
  if (preference) return preference
  if (SETTINGS_RE.test(text)) {
    return { intent: 'settings', text: '设置面板已为你展开 👇', mood: 'neutral', params: { open_page: true } }
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
