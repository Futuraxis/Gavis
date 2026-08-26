import type { BoardSnapshot, MahjongSnapshot, PokerSnapshot, Snapshot, SocialSnapshot } from '../../types'
import GenericGridBoard from './GenericGridBoard'
import MahjongTable from './MahjongTable'
import PokerTable from './PokerTable'
import SocialChatTable from './SocialChatTable'

/** 族分发面板统一 props — onMove 接受任意动作结构（由各族组件自行构造）。 */
export interface FamilyBoardProps {
  snapshot: Snapshot
  interactive: boolean
  onMove?: (action: unknown) => void
  stepKey?: number
}

/** 族 → 渲染组件映射：渲染分发以 family 优先，查不到再回退 kind/game_id 分支。 */
export const FAMILY_BOARDS: Record<string, React.FC<FamilyBoardProps>> = {
  grid: ({ snapshot, interactive, onMove, stepKey }) => (
    <GenericGridBoard
      snapshot={snapshot as BoardSnapshot}
      interactive={interactive}
      stepKey={stepKey}
      onMove={(i) => onMove?.({ cell_index: i })}
    />
  ),
  poker: ({ snapshot, interactive, onMove }) => (
    <PokerTable
      snapshot={snapshot as PokerSnapshot}
      interactive={interactive}
      onAction={(a) => onMove?.(a)}
    />
  ),
  mahjong: ({ snapshot, interactive, onMove }) => (
    <MahjongTable
      snapshot={snapshot as MahjongSnapshot}
      interactive={interactive}
      onAction={(a) => onMove?.(a)}
    />
  ),
  social: ({ snapshot, interactive, onMove }) => (
    <SocialChatTable
      snapshot={snapshot as SocialSnapshot}
      interactive={interactive}
      onMove={(a) => onMove?.(a)}
    />
  ),
}
// social entry appended by B3