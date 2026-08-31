// InlineBoard — 对话流里的常驻对局面板（钉在输入框上方）。
// 棋盘/牌面点击走快速路径（/match/move，不经 LLM）；文本动作走聊天。

import type { ReactNode } from 'react'
import type { BoardSnapshot, GameInfo, MahjongSnapshot, PokerSnapshot, Snapshot, SocialSnapshot } from '../../types'
import { FAMILY_BOARDS } from '../../components/boards/familyBoards'
import GenericGridBoard from '../../components/boards/GenericGridBoard'
import MahjongTable from '../../components/boards/MahjongTable'
import PokerTable from '../../components/boards/PokerTable'
import { resolveBoardFamily } from '../boardFamily'

interface Props {
  snapshot: Snapshot
  game: GameInfo | null
  busy: boolean
  onMove: (action: unknown) => void
  onRestart: () => void
}

/** 社交阵营胜者的中文标签（无匹配 → null，回退 pid 座位称呼）。 */
function factionLabel(winner: string): string | null {
  if (winner === 'undercover') return '卧底'
  if (winner === 'civilian') return '平民'
  if (winner === 'blank') return '白板'
  if (winner === 'wolf') return '狼人'
  if (winner === 'good') return '好人'
  return null
}

export default function InlineBoard({ snapshot, game, busy, onMove, onRestart }: Props) {
  // 渲染分发对齐 BattlePage: 优先按 family（快照自描述 / game.family，见
  // boardFamily.resolveBoardFamily）经 FAMILY_BOARDS 查表；**未知 family 绝不
  // 默认 grid** —— 非 grid 快照没有 board，误路由到 GenericGridBoard 会在
  // board.length 上直接崩掉整个对话页（如 mahjong_sichuan/changsha/taiwan 等
  // 曾不在后端 _BUILTIN_FAMILY 的麻将变体）。查不到再按 game.kind 兜底；
  // kind 也未知（游戏目录未加载等）时显示占位提示。
  const family = resolveBoardFamily(snapshot, game)
  const Board = FAMILY_BOARDS[family]
  const myTurn = !snapshot.over && snapshot.turn === snapshot.player_pid && !busy
  // 终局胜负（头部口径）：社交阵营胜者（winner=undercover 等）按终局身份表
  // 归边——否则卧底获胜被误标「AI 赢了」（实测 e7deb84b）；其余回退 pid 比较。
  const myFinalRole = (snapshot as SocialSnapshot).final_roles?.find((r) => r.pid === snapshot.player_pid)?.role ?? null
  const won =
    snapshot.winner != null &&
    (snapshot.winner === snapshot.player_pid || (myFinalRole != null && myFinalRole === snapshot.winner))

  let board: ReactNode
  if (Board) {
    board = <Board snapshot={snapshot} interactive={myTurn} onMove={onMove} />
  } else if (game?.kind === 'mahjong') {
    board = (
      <MahjongTable
        snapshot={snapshot as MahjongSnapshot}
        interactive={myTurn}
        onAction={(a) => onMove(a)}
      />
    )
  } else if (game?.kind === 'poker') {
    board = (
      <PokerTable
        snapshot={snapshot as PokerSnapshot}
        interactive={myTurn}
        onAction={(a) => onMove(a)}
      />
    )
  } else if (game?.kind === 'board') {
    board = (
      <GenericGridBoard
        snapshot={snapshot as BoardSnapshot}
        interactive={myTurn}
        onMove={(i) => onMove({ cell_index: i })}
      />
    )
  } else {
    board = (
      <div className="chat-inline-error">
        暂不支持渲染该对局{game ? `（family=${family}）` : '（游戏信息加载中…）'}。
      </div>
    )
  }

  return (
    <div className="chat-board">
      <div className="chat-board-head">
        <span className="chat-board-title">{game?.display_name ?? snapshot.game_id}</span>
        <span className="chat-board-meta">
          {snapshot.over
            ? won
              ? '🎉 你赢了'
              : snapshot.winner
                // 目录未加载 / 未命中时不能回退原始 pid（会泄漏 'p_white'），
                // 退到关系称呼 "AI"（本面板对手恒为 AI）。
                ? `${factionLabel(snapshot.winner) ?? game?.seat_names?.[snapshot.winner] ?? 'AI'} 赢了`
                : '平局'
            : busy
              ? 'AI 思考中…'
              : myTurn
                ? (snapshot as MahjongSnapshot).phase === 'claim'
                  ? // claim 是响应别人打出的牌，不是你的出牌回合。
                    `响应 ${game?.seat_names?.[(snapshot as MahjongSnapshot).last_discarder ?? ''] ?? '对方'} 的牌`
                  : '轮到你了'
                : 'AI 回合'}
        </span>
        <span className="chat-board-close" role="button" onClick={onRestart} title="重新开始">
          再来一局
        </span>
      </div>
      {board}
    </div>
  )
}