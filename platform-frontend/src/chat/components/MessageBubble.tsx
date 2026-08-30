// MessageBubble — 单条聊天消息；agent 消息按意图内联卡片（开局卡/创建卡/战绩/复盘/进度）。
// 对局面板不挂消息上——由 ChatPage 钉在输入框上方（InlineBoard），消息里只留文字与卡片。

import type { ChatMessage, GameInfo } from '../../types'
import type { BattleConfig } from '../../components/BattleSetup'
import SetupCard from './SetupCard'
import CreateCard from './CreateCard'
import StatsCard from './StatsCard'
import ReviewCard from './ReviewCard'
import ProgressCard from './ProgressCard'
import Chips from './Chips'
import type { StatsData, BenchmarkJob, LearningItem } from '../useChatRuntime'

interface Props {
  msg: ChatMessage
  games: GameInfo[]
  busy: boolean
  onStart: (gameId: string, config: BattleConfig) => void
  onCreated: (game: GameInfo) => void
  onChip: (chip: string) => void
}

export default function MessageBubble({ msg, games, busy, onStart, onCreated, onChip }: Props) {
  const params = msg.params ?? {}

  function inlineCard() {
    switch (msg.intent) {
      case 'play': {
        const game = games.find((g) => g.game_id === params.game_id)
        return game ? <SetupCard game={game} busy={busy} onStart={(c) => onStart(game.game_id, c)} /> : null
      }
      case 'create':
        return <CreateCard onCreated={onCreated} />
      case 'history':
        return (
          <StatsCard
            matches={(params.matches ?? []) as StatsData['matches']}
            wins={Number(params.wins ?? 0)}
            plays={Number(params.plays ?? 0)}
          />
        )
      case 'review':
        return params.report ? (
          <ReviewCard report={params.report as import('../../types').ReviewReport} matchId={String(params.match_id ?? '')} />
        ) : null
      case 'benchmark':
        return <ProgressCard mode="benchmark" jobs={(params.jobs ?? []) as BenchmarkJob[]} />
      case 'learning':
        return <ProgressCard mode="learning" learning={(params.learning ?? []) as LearningItem[]} />
      default:
        return null
    }
  }

  // clarify 追问选项与 chat 知识回答的“来一局”快捷 chips 复用同一组件。
  const chips =
    msg.intent === 'clarify' || msg.intent === 'chat' ? ((params.chips ?? []) as string[]) : []

  return (
    <div className={`chat-msg ${msg.role === 'player' ? 'chat-msg-player' : 'chat-msg-agent'}`}>
      <div className="chat-msg-bubble">
        <div className="chat-msg-bubble-text">{msg.text}</div>
        {inlineCard()}
        {chips.length > 0 && <Chips chips={chips} disabled={busy} onPick={onChip} />}
      </div>
    </div>
  )
}