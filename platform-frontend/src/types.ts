// API 数据类型 — 与 layer4_interface/frontend/platform/ 的 JSON 契约对应

export interface GameInfo {
  game_id: string
  display_name: string
  description: string
  kind: 'board' | 'poker' | 'mahjong' | 'uno'
  board_size: number | null
  seat_options: string[]
  seat_label: string
  /** 座位 pid → 中文称呼（后端按族下发：麻将=庄家/下家…、社交/UNO=N号玩家；前端零推导）。 */
  seat_names: Record<string, string>
  player_counts: number[]
  difficulties: string[]
  /** variant 主题选项（仅 undercover 等多 variant 游戏携带；null/缺省 → 无主题选择 UI）。 */
  variant_themes?: string[] | null
  solver_options: string[]
  family: string // 族: grid / poker / mahjong / social（渲染以 family 为准）
  custom: boolean // 是否来自自定义游戏注册表
  created_at?: string // 自定义游戏创建时间（自定义条目携带）
  aliases?: string[] // 短名/别名（内置游戏携带；断连兜底 classifyLocal 的短名匹配用）
}

// ── 自定义游戏 (A2 后端契约 / A3 前端) ──────────────────────────

export interface ValidationInfo {
  valid: boolean
  errors: string[]
  warnings: string[]
}

export interface CustomCreateResult {
  ok: boolean
  game_id: string
  game: GameInfo
  confidence: number
  family: string
  diff_summary?: string
  validation: ValidationInfo
}

// ── 快照 (snapshot) ────────────────────────────────────────────

/**
 * 局势评估（后端 session.snapshot() 统一注入，人类视角；agent 关闭时为 null）。
 * score ∈ [-1, 1]（正 = 人类占优）；summary 为一句中文局势描述，
 * mechanical_text 为可悬浮查看的机械化依据。
 */
export interface EvaluationInfo {
  score: number
  summary: string
  mechanical_text?: string
}

export interface BoardSnapshot {
  game_id: string
  family?: string // 族（后端 session 统一注入，快照自描述；social 快照为必填）
  player_pid: string
  difficulty: string
  evaluation?: EvaluationInfo | null // 局势评估（agent 关闭时为 null）
  board: (string | null)[]
  board_size?: number // 自定义 grid 游戏: N×N 边长（既有 moon/gomoku 快照不含）
  win_length?: number // 自定义 grid 游戏: 连珠长度（仅展示/信息用途）
  round_age?: Record<string, number> // moon chess: cell index → piece age
  turn: string | null
  winner: string | null
  over: boolean
  last_ai_move: number | null
  last_vanish?: number | null // gomoku: vanished cell, if any
  last_vanish_color?: string | null // gomoku: color of the vanished piece
  round?: number
  pending_cell?: number | null // 前端乐观落子标记: 人落子即时反馈（仅渲染层临时叠加）
  invalid_cell?: number | null // 前端非法落子标记: 被拒绝的格子就地闪烁提示（仅渲染层临时叠加）
}

export interface PokerSnapshot {
  game_id: string
  family?: string // 族（后端 session 统一注入，快照自描述）
  player_pid: string
  ai_pid: string
  difficulty: string
  evaluation?: EvaluationInfo | null // 局势评估（agent 关闭时为 null）
  over: boolean
  winner: string | null
  turn: string | null
  phase: string | null
  street: number
  street_name: string
  pot: number
  community: string[]
  my_hole: string[]
  ai_hole: string[]
  revealed: boolean
  my_stack: number
  ai_stack: number
  my_committed: number
  ai_committed: number
  my_folded: boolean
  ai_folded: boolean
  last_actor: string | null
  last_action: string | null
  last_ai_action: string | null
  call_to: number
  my_hand_name: string | null
  ai_hand_name: string | null
  payoff: number | null
  legal: { choice: string; amount: number | null }[]
  raise_amounts: number[]
}

export interface MahjongMeld {
  type: string
  tiles: string[]
  from?: string | null
}

export interface MahjongSnapshot {
  game_id: string
  family?: string // 族（后端 session 统一注入，快照自描述）
  player_pid: string
  ai_pid: string
  difficulty: string
  evaluation?: EvaluationInfo | null // 局势评估（agent 关闭时为 null）
  over: boolean
  winner: string | null
  turn: string | null
  phase: string | null
  my_hand: string[]
  ai_hand: string[]
  // 终局所有座位手牌（含全部 AI 座位）；非终局为空对象。
  final_hands?: Record<string, string[]>
  hand_counts: Record<string, number>
  melds: Record<string, MahjongMeld[]>
  discards: Record<string, string[]>
  wall_remaining: number
  last_discard: string | null
  last_discarder?: string | null // 谁打的（claim 提示用；旧后端快照可能缺省）
  last_drawn: string | null
  last_action: string | null
  done: string[]
  winners: string[]
  payoffs: number[]
  claim: { queue: string[]; passed: number; actor: string | null } | null
  legal: { type: string; tile?: string; tiles?: string[] }[]
  last_ai_action: string | null
}

// ── 社交推理族 (B3 契约) ─────────────────────────────────────────

export interface SocialDiscourseEntry {
  speaker: string
  text: string
  round?: number
  intent?: string
  /** 自爆等事件条目（非发言）：event=self_destruct + target + guess。 */
  event?: string
  target?: string
  guess?: string
}

export interface SocialSnapshot {
  family: 'social'
  game_id: string
  player_pid: string
  difficulty: string
  evaluation?: EvaluationInfo | null // 局势评估（agent 关闭时为 null）
  over: boolean
  winner: string | null
  turn: string | null
  phase: string | null
  round?: number | null // 当前轮次（who is the undercover 的轮数；狼人杀等可能无）
  my_role: string | null
  my_word?: string | null // 卧底局：自己的词卡（狼人杀等无词卡玩法为 null）
  alive: string[]
  discourse: SocialDiscourseEntry[]
  /** 投票记录（v5.2 公开 vote_log 投影）：voter → target，含轮次。 */
  votes?: { voter: string; target: string; round?: number }[]
  /** 出局名单（公开 deaths_arr）：按出局顺序的玩家 id。 */
  deaths?: string[]
  /** 当前轮被投出者 id（平票无人出局时为 null）。 */
  eliminated?: string | null
  last_action: string | null
  winners: string[]
  legal: { type: string; text?: string; target?: string; guess?: string }[]
  ai_mode: 'ollama' | 'random'
  /** 终局全场身份揭晓（含存活者）；undercover 另带 word，werewolf 只有 role。 */
  final_roles?: { pid: string; role: string | null; word?: string | null }[]
}

export type Snapshot = (BoardSnapshot | PokerSnapshot | MahjongSnapshot | SocialSnapshot | UnoSnapshot) & SnapshotExtras

/**
 * 会话级注入键（每条快照统一带上，不属于某族快照本体契约）：
 *   - ``chat`` — 后端 pending_chat 增量（陪伴 Agent / 教练待投递消息）。
 *   - ``teaching`` — 教学对局标记（教练能看到玩家自己的牌并推理）。
 */
export interface SnapshotExtras {
  chat?: SnapshotChatEntry[]
  teaching?: boolean
  /** 自适应难度已生效：本局 AI 强度由玩家近 10 局胜率自动升降（非显式档位）。 */
  adaptive?: boolean
  /** 本局实际 AI 搜索强度（预算；自适应开启时随胜率浮动）。 */
  ai_strength?: number
}

/** 后端 session.snapshot() 的 chat 增量条目（agent 消息，按 step 排序）。 */
export interface SnapshotChatEntry {
  scenario: string
  text: string
  mood: Mood
  step: number
  /** 陪伴/教练消息的思维链（统一客户端 reasoning 透传；可选）。 */
  reasoning?: string
  /**
   * 说话人标签（座位 pid + 显示名 / persona 显示名）。多座位群聊（P2）按
   * 此渲染不同头像与名字；二人对手模式与啦啦队都带 persona 显示名。
   * 旧后端 / 旧快照可能缺省 → 前端按 persona 默认名兜底渲染。
   */
  speaker?: string
}

// ── UNO 族 ────────────────────────────────────────────────────────

/** UNO 动作（与后端 _uno_snapshot 的 legal 条目一一对应）。 */
export interface UnoLegalAction {
  type: string
  card?: string
  color?: string
  target?: string
}

export interface UnoSnapshot {
  family?: string // 族（后端 session 统一注入）
  game_id: string
  player_pid: string
  ai_pid: string
  difficulty: string
  evaluation?: EvaluationInfo | null // 局势评估（agent 关闭时为 null）
  over: boolean
  winner: string | null
  turn: string | null
  phase: string | null
  direction: number // 1 顺时针 / -1 逆时针
  top_color: string | null // 台面顶牌颜色（wild 后为所选色）
  top_symbol: string | null // 台面顶牌符号（数字 / skip / reverse / draw2 / wild / wild4）
  my_hand: string[]
  ai_hand: string[] // 仅终局展示（隐藏信息红线）
  // 终局所有座位手牌（含全部 AI 座位）；非终局为空对象。
  final_hands?: Record<string, string[]>
  hand_counts: Record<string, number> // 他人只暴露张数
  discard_top: string | null
  discard_recent: string[]
  deck_count: number
  pending_draw: number
  penalty_target: string | null
  // UNO env 未存动作类型，只暴露最后行动者（与德州快照 last_actor 对齐）；
  // 动作内容见 last_ai_action。曾误命名为 last_action（值实为 lastActor）。
  last_actor: string | null
  last_ai_action: string | null
  legal: UnoLegalAction[]
  payoff: number | null
}

// ── 历史与回放 ─────────────────────────────────────────────────

export interface MatchMeta {
  match_id: string
  game_id: string
  player_pid: string
  ai_pid: string
  difficulty: string
  winner: string | null
  /** 玩家视角胜负（后端 layer4_interface/result 解析；阵营胜者正确归边——旧记录缺省 → 前端回退 pid 比较）。 */
  won?: boolean | null
  over: boolean
  moves: number
  started_at: string
  finished_at: string
  persona?: PersonaKey | null
  hinted?: boolean
  /** 本局实际 AI 搜索强度（预算；自适应局随胜率浮动，旧记录缺省）。 */
  ai_strength?: number | null
  teaching?: boolean | null
  /** 自适应难度局标记（旧记录缺省 undefined）。 */
  adaptive?: boolean
  /** 座位 pid → 中文称呼（后端 /api/history 注入；旧记录缺省 → 前端兜底 pid）。 */
  seat_names?: Record<string, string>
}

export interface MoveEntry {
  step: number
  actor: 'human' | 'ai'
  action: string
  snapshot: Snapshot
}

export interface MatchLog {
  match_id: string
  game_id: string
  player_pid: string
  ai_pid: string
  difficulty: string
  seed: number
  started_at: string
  finished_at: string
  winner: string | null
  /** 玩家视角胜负（后端 layer4_interface/result 解析；阵营胜者正确归边——旧记录缺省）。 */
  won?: boolean | null
  over: boolean
  moves: MoveEntry[]
  persona?: PersonaKey | null
  hinted?: boolean
  ai_strength?: number | null
  adaptive?: boolean
  family?: string // 复盘快照渲染优先按 family 分发
  /** 座位 pid → 中文称呼（后端 /api/history/:id 注入；旧记录缺省 → 前端兜底 pid）。 */
  seat_names?: Record<string, string>
}

// ── 求解器评测 ─────────────────────────────────────────────────

export interface BenchmarkResults {
  iterations: number
  a_wins: number
  b_wins: number
  draws: number
  a_win_rate: number
  b_win_rate: number
  draw_rate: number
  avg_moves: number
  avg_seconds_per_move: number
  errors: number
  per_iteration: { moves: number; seconds: number; winner: string | null }[]
}

export interface BenchmarkJob {
  job_id: string
  game_id: string
  solver_a: string
  solver_b: string
  iterations: number
  status: 'pending' | 'running' | 'done' | 'error'
  progress: number
  error: string | null
  results: BenchmarkResults | null
  started_at: string | null
  ended_at: string | null
}

// ── 在线学习 ─────────────────────────────────────────────────

export interface LearningGate {
  episodes: number
  candidate_wins: number
  baseline_wins: number
  draws: number
  candidate_win_rate: number
  baseline_win_rate: number
  draw_rate: number
  budget: number
}

export interface LearningModel {
  version: number
  samples: number
  coverage: number
  published_at: string | null
  gate: LearningGate | null
  preview: unknown[]
}

export interface LearningStatus {
  game_id: string
  enabled: boolean
  matches: number
  decisions: number
  human_decisions: number
  ai_decisions: number
  model: LearningModel | null
  min_samples: number
  pending: boolean
}

export interface LearningApplyResult {
  game_id: string
  applied: boolean
  reason: 'ok' | 'insufficient' | 'unchanged' | 'rejected' | 'disabled' | 'error'
  version: number | null
  gate: LearningGate | null
  samples: number
  coverage: number
  error: string | null
}

// ── Agent 陪伴 (C2 契约) ───────────────────────────────────────

export type Mood = 'happy' | 'thinking' | 'sorry' | 'neutral'
export type PersonaKey = 'gentle' | 'teacher' | 'banter' | 'cold'
export type HintLevel = 'off' | 'direction' | 'specific' | 'demo'
export type Pacing = 'fast' | 'standard' | 'slow'

export interface AgentMessage {
  text: string
  mood: Mood
}

// ── Chat-first (对话即一切, agent 聊天模式) ──────────────────────
// 意图契约与后端 layer4_interface/frontend/platform/chat.py 对齐。

export type ChatIntent =
  | 'play'
  | 'resume'
  | 'move'
  | 'hint'
  | 'restart'
  | 'history'
  | 'review'
  | 'create'
  | 'settings'
  | 'platform'
  | 'benchmark'
  | 'learning'
  | 'help'
  | 'chat'
  | 'clarify'

export interface ChatTurnResult {
  intent: ChatIntent
  text: string
  mood: Mood
  params: Record<string, unknown>
}

export interface ChatMessage {
  id: string
  role: 'agent' | 'player'
  text: string
  mood?: Mood
  /**
   * 思维链（模型思考过程）。流式时随 reasoning 事件增量累积；
   * 存档/镜像原样保留；MessageBubble 以可折叠块展示。
   */
  reasoning?: string
  ts: number
  /** 后端意图（agent 消息携带；用于消息内联卡片渲染与 chips）。 */
  intent?: ChatIntent
  /** 意图执行参数（如 play.game_id、clarify.chips、history.matches…）。 */
  params?: Record<string, unknown>
  /** 发送/执行中占位标记。 */
  pending?: boolean
  /**
   * 说话人标签（来自快照 chat 增量的 speaker；多座位群聊按此区分头像/名字）。
   * 平台聊天助手消息（非对局陪伴）不带此字段 → 按 persona 默认名渲染。
   */
  speaker?: string
}

// ── 对话管理与存档 (conversations, 与后端 conversations.py 契约对齐) ──

/** 会话列表条目（/api/conversations 与 append/update 响应携带）。 */
export interface ConversationMeta {
  conv_id: string
  title: string
  archived: boolean
  created_at: string
  updated_at: string
  message_count: number
  preview: string
}

/** 完整会话记录（/api/conversations/<id> 携带，含消息流）。 */
export interface Conversation extends ConversationMeta {
  messages: ChatMessage[]
}

export interface ChatState {
  messages: ChatMessage[]
  muted: boolean
}

// ── 活跃会话 (D 节接线: /api/match/active) ─────────────────────

export interface ActiveSession {
  game_id: string // 会话 id → 用于 ?game=<id> 恢复
  game: string // 游戏注册 id（/battle/<game> 路由）
  display_name: string
  player_pid: string
  difficulty: string
  persona: PersonaKey | null
  hint_level: HintLevel
  teaching?: boolean
  /** 自适应难度已生效 + 实际 AI 强度（未开启/旧后端时缺省）。 */
  adaptive?: boolean
  ai_strength?: number
  step: number
  started_at: string
}

// ── 偏好档案 (C3 契约) ─────────────────────────────────────────

export interface Profile {
  nickname: string
  agent_call: string
  default_persona: PersonaKey
  default_difficulty: string
  hint_level: HintLevel
  pacing: Pacing
  adaptive: boolean
  difficulty_locked: boolean
  learning_enabled: boolean
  theme: 'light' | 'dark'
  recent: Record<string, { wins: number; plays: number }>
}

// ── 复盘 (C4 契约) ─────────────────────────────────────────────

export type KeyNodeKind = 'turning_point' | 'winning_move' | 'blunder'

export interface KeyNode {
  step: number
  kind: KeyNodeKind
  why: string
  /** 该手的动作内容（后端 moves[].action，如 cell_1_2 / raise 40） */
  what?: string
}

export interface ReviewReport {
  key_nodes: KeyNode[]
  improvement: string
  summary: string
}
