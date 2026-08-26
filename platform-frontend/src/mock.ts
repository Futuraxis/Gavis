// 自测假数据 — 后端路由 (/agent/say、/profile、/review/:id) 接线前用于页面渲染兜底。
// 集成阶段接线后，各页面优先走真实 API，本文件仅作连接失败时的 fallback。

import type { GameInfo, MatchLog, Profile, ReviewReport } from './types'

export const DEFAULT_PROFILE: Profile = {
  nickname: '玩家',
  agent_call: '阿远',
  default_persona: 'gentle',
  default_difficulty: 'normal',
  hint_level: 'direction',
  pacing: 'standard',
  adaptive: true,
  difficulty_locked: false,
  learning_enabled: true,
  theme: 'light',
  recent: {
    moon_chess: { wins: 3, plays: 5 },
    texas_holdem: { wins: 1, plays: 2 },
    stochastic_gomoku: { wins: 0, plays: 1 },
  },
}

export const MOCK_GAMES: GameInfo[] = [
  {
    game_id: 'moon_chess',
    display_name: '月亮棋',
    description: '3×3 经典月亮棋：三子连珠即胜，棋盘满时最旧的棋子被挤出。',
    kind: 'board',
    board_size: 3,
    seat_options: ['p_black', 'p_white'],
    seat_label: '颜色',
    player_counts: [2],
    difficulties: ['easy', 'normal', 'hard'],
    solver_options: ['mcts', 'random'],
    family: 'grid',
    custom: false,
  },
  {
    game_id: 'stochastic_gomoku',
    display_name: '随机五子棋',
    description: '9×9 五子棋变体：每次落子后棋子有 50% 概率被随机抹去。',
    kind: 'board',
    board_size: 9,
    seat_options: ['p_black', 'p_white'],
    seat_label: '颜色',
    player_counts: [2],
    difficulties: ['easy', 'normal', 'hard'],
    solver_options: ['mcts', 'random'],
    family: 'grid',
    custom: false,
  },
  {
    game_id: 'texas_holdem',
    display_name: '德州扑克',
    description: '双人德州扑克：翻前/翻牌/转牌/河牌四轮下注。',
    kind: 'poker',
    board_size: null,
    seat_options: ['p_sb', 'p_bb'],
    seat_label: '座位',
    player_counts: [2],
    difficulties: ['easy', 'normal', 'hard'],
    solver_options: ['hybrid', 'random'],
    family: 'poker',
    custom: false,
  },
]

function board9(...cells: (string | null)[]): (string | null)[] {
  const b: (string | null)[] = Array(9).fill(null)
  cells.forEach((c, i) => {
    if (c) b[i] = c
  })
  return b
}

export const MOCK_MATCH_LOG: MatchLog = {
  match_id: 'mock-moon-001',
  game_id: 'moon_chess',
  player_pid: 'p_black',
  ai_pid: 'p_white',
  difficulty: 'normal',
  seed: 42,
  started_at: new Date().toISOString(),
  finished_at: new Date().toISOString(),
  winner: 'p_white',
  over: true,
  persona: 'gentle',
  hinted: true,
  ai_strength: 'normal',
  moves: [
    {
      step: 0,
      actor: 'human',
      action: 'cell_1_1',
      snapshot: {
        game_id: 'mock-moon-001',
        player_pid: 'p_black',
        difficulty: 'normal',
        board: board9(null, null, null, null, 'p_black', null, null, null, null),
        turn: 'p_white',
        winner: null,
        over: false,
        last_ai_move: null,
      },
    },
    {
      step: 1,
      actor: 'ai',
      action: 'cell_0_0',
      snapshot: {
        game_id: 'mock-moon-001',
        player_pid: 'p_black',
        difficulty: 'normal',
        board: board9('p_white', null, null, null, 'p_black', null, null, null, null),
        turn: 'p_black',
        winner: null,
        over: false,
        last_ai_move: 0,
      },
    },
    {
      step: 2,
      actor: 'human',
      action: 'cell_0_1',
      snapshot: {
        game_id: 'mock-moon-001',
        player_pid: 'p_black',
        difficulty: 'normal',
        board: board9('p_white', 'p_black', null, null, 'p_black', null, null, null, null),
        turn: 'p_white',
        winner: null,
        over: false,
        last_ai_move: 0,
      },
    },
    {
      step: 3,
      actor: 'ai',
      action: 'cell_2_2',
      snapshot: {
        game_id: 'mock-moon-001',
        player_pid: 'p_black',
        difficulty: 'normal',
        board: board9('p_white', 'p_black', null, null, 'p_black', null, null, null, 'p_white'),
        turn: null,
        winner: 'p_white',
        over: true,
        last_ai_move: 8,
      },
    },
  ],
}

export const MOCK_REVIEW: ReviewReport = {
  key_nodes: [
    { step: 2, kind: 'blunder', why: '己方评估值显著下降，随后落败的一手。' },
    { step: 3, kind: 'winning_move', why: '胜方最后一次落子，锁定三连。' },
    { step: 1, kind: 'turning_point', why: '效用跳变最大的一手。' },
  ],
  improvement: '第 3 手后优势转为劣势，注意守住角位与斜线。',
  summary: '共 4 步，AI 获胜。',
}
