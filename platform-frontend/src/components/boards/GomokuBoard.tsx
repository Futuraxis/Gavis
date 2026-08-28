import { useEffect, useRef, useState } from 'react'
import type { BoardSnapshot } from '../../types'

interface Props {
  snapshot: BoardSnapshot
  interactive: boolean
  stepKey: number
  onMove?: (cellIndex: number) => void
}

interface Ghost {
  cell: number
  color: string
  key: number
}

/** 9×9 随机五子棋棋盘 — 棋子被随机抹去时播放幽灵消散动画。 */
export default function GomokuBoard({ snapshot, interactive, stepKey, onMove }: Props) {
  const { board, last_vanish, last_vanish_color, pending_cell, invalid_cell } = snapshot
  const [ghost, setGhost] = useState<Ghost | null>(null)
  const prevVanish = useRef<number | null>(null)

  // 快照出现新的消失事件时, 用被抹去棋子的颜色渲染消散动画
  useEffect(() => {
    if (last_vanish != null && last_vanish !== prevVanish.current) {
      prevVanish.current = last_vanish
      if (last_vanish_color) {
        setGhost({ cell: last_vanish, color: last_vanish_color, key: stepKey })
      }
      const timer = setTimeout(() => setGhost(null), 950)
      return () => clearTimeout(timer)
    }
  }, [last_vanish, last_vanish_color, stepKey])

  return (
    <div className="board-grid" style={{ gridTemplateColumns: 'repeat(9, 44px)' }}>
      {board.map((piece, i) => {
        const empty = piece == null
        return (
          <div
            key={i}
            className={`cell ${interactive && empty ? 'interactive empty' : ''}${invalid_cell === i ? ' cell-invalid' : ''}`}
            onClick={() => {
              if (interactive && empty && onMove) onMove(i)
            }}
            style={{ width: 44, height: 44 }}
          >
            {piece != null && (
              <div
                className={`piece ${piece}${pending_cell === i ? ' piece-pending' : ''}`}
                style={{ width: 32, height: 32 }}
              />
            )}
            {ghost != null && ghost.cell === i && (
              <div className="ghost-wrap" key={ghost.key}>
                <div className={`ghost-piece ${ghost.color}`} />
                <div className="vanish-smoke">💨</div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
