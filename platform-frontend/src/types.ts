// API 数据类型 — 与 layer4_interface/frontend/platform/ 的 JSON 契约对应

export interface GameInfo {
  game_id: string
  display_name: string
  description: string
  kind: 'board' | 'poker' | 'mahjong'
  board_size: number | null
  seat_options: string[]
  seat_label: string
  player_counts: number[]
  difficulties: string[]
  solver_options: string[]
}

// ── 快照 (snapshot) ────────────────────────────────────────────

export interface BoardSnapshot {
  game_id: string
  player_pid: string
  difficulty: string
  board: (string | null)[]
  round_age?: Record<string, number> // moon chess: cell index → piece age
  turn: string | null
  winner: string | null
  over: boolean
  last_ai_move: number | null
  last_vanish?: number | null // gomoku: vanished cell, if any
  last_vanish_color?: string | null // gomoku: color of the vanished piece
  round?: number
}

export interface PokerSnapshot {
  game_id: string
  player_pid: string
  ai_pid: string
  difficulty: string
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
  player_pid: string
  ai_pid: string
  difficulty: string
  over: boolean
  winner: string | null
  turn: string | null
  phase: string | null
  my_hand: string[]
  ai_hand: string[]
  hand_counts: Record<string, number>
  melds: Record<string, MahjongMeld[]>
  discards: Record<string, string[]>
  wall_remaining: number
  last_discard: string | null
  last_action: string | null
  done: string[]
  winners: string[]
  payoffs: number[]
  claim: { queue: string[]; passed: number; actor: string | null } | null
  legal: { type: string; tile?: string; tiles?: string[] }[]
  last_ai_action: string | null
}

export type Snapshot = BoardSnapshot | PokerSnapshot | MahjongSnapshot

// ── 历史与回放 ─────────────────────────────────────────────────

export interface MatchMeta {
  match_id: string
  game_id: string
  player_pid: string
  ai_pid: string
  difficulty: string
  winner: string | null
  over: boolean
  moves: number
  started_at: string
  finished_at: string
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
  over: boolean
  moves: MoveEntry[]
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
