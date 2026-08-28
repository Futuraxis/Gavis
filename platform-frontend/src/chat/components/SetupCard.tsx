// SetupCard — 开局配置卡（内嵌于聊天消息；直接复用 BattleSetup 表单）。

import type { GameInfo } from '../../types'
import type { BattleConfig } from '../../components/BattleSetup'
import BattleSetup from '../../components/BattleSetup'

interface Props {
  game: GameInfo
  busy: boolean
  onStart: (config: BattleConfig) => void
}

export default function SetupCard({ game, busy, onStart }: Props) {
  return (
    <div className="chat-card">
      <BattleSetup game={game} busy={busy} error={null} onStart={onStart} />
    </div>
  )
}