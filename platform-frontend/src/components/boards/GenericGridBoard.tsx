import { useEffect, useMemo, useRef, useState } from 'react'
import type { BoardSnapshot } from '../../types'

interface Props {
  /** 自定义 grid 家族游戏快照（含 board_size）；既有 moon/gomoku 快照缺失时按板长推导。 */
  snapshot: BoardSnapshot
  interactive: boolean
  stepKey?: number
  onMove?: (cellIndex: number) => void
}

interface Ghost {
  cell: number
  color: string
  key: number
}

const SEAT_COLORS = ['p_black', 'p_white']

/** 通用 N×N 网格棋盘 — 自定义 grid 家族游戏渲染。
 *
 * 棋子 occupant 为 p_black/p_white 时直接使用原类；其他座位 id（如 p0/p1/自定义）按
 * 棋盘扫描序映射到 p_black/p_white。last_vanish 时播放幽灵消散动画（仿 GomokuBoard）。
 */
export default function GenericGridBoard({ snapshot, interactive, stepKey = 0, onMove }: Props) {
  // board 缺失时兜底为空棋盘（防误路由快照在 board.length 处崩掉整棵渲染树）。
  const { board = [], board_size, last_vanish, last_vanish_color, pending_cell, invalid_cell } = snapshot
  const size = board_size ?? Math.round(Math.sqrt(board.length))
  const [ghost, setGhost] = useState<Ghost | null>(null)
  const prevVanish = useRef<number | null>(null)

  // 非标准座位 id → p_black/p_white 按首次出现顺序稳定映射
  const pieceColor = useMemo(() => {
    const map = new Map<string, string>()
    let next = 0
    for (const p of board) {
      if (p != null && !map.has(p)) {
        map.set(p, SEAT_COLORS[next % SEAT_COLORS.length])
        next++
      }
    }
    return (occupant: string): string => {
      if (occupant === 'p_black' || occupant === 'p_white') return occupant
      return map.get(occupant) ?? 'p_black'
    }
  }, [board])

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
    <div className="board-grid" style={{ gridTemplateColumns: `repeat(${size}, 44px)` }}>
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
                className={`piece ${pieceColor(piece)}${pending_cell === i ? ' piece-pending' : ''}`}
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