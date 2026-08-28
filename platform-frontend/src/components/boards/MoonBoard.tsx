import type { BoardSnapshot } from '../../types'

interface Props {
  snapshot: BoardSnapshot
  interactive: boolean
  onMove?: (cellIndex: number) => void
}

/** 3×3 月亮棋棋盘 — 棋子带年龄角标, AI 最后落子带高亮环。 */
export default function MoonBoard({ snapshot, interactive, onMove }: Props) {
  const { board, round_age = {}, last_ai_move, pending_cell, invalid_cell } = snapshot
  return (
    <div className="board-grid" style={{ gridTemplateColumns: 'repeat(3, 96px)' }}>
      {board.map((piece, i) => {
        const age = round_age[i]
        const empty = piece == null
        return (
          <div
            key={i}
            className={`cell ${interactive && empty ? 'interactive empty' : ''}${invalid_cell === i ? ' cell-invalid' : ''}`}
            onClick={() => {
              if (interactive && empty && onMove) onMove(i)
            }}
            style={{ width: 96, height: 96 }}
          >
            {last_ai_move === i && <div className="ai-move-ring" />}
            {piece != null && <div className={`piece ${piece}${pending_cell === i ? ' piece-pending' : ''}`} />}
            {age != null && <span className="age-badge">{age}</span>}
          </div>
        )
      })}
    </div>
  )
}
