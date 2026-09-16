// MessageBubble — 单条聊天消息；agent 消息按意图内联**只读**卡片（战绩/复盘/进度）。
// 对局面板不挂消息上——由 ChatPage 钉在输入框上方（InlineBoard），消息里只留文字与卡片。
// 「开局配置卡」「创建游戏表单卡」已下线：开局偏好由 play_game 工具参数带过来
// （后端 → params.config），建游戏由 create_game 工具在对话里直接执行，因此消息里
// 不再需要多步点选的表单（本组件只剩只读卡片 + 快捷 chips）。
// 思维链（reasoning）折叠块仅在调试模式打开时渲染（getStoredDebug）；
// 后端照常产出/透传 reasoning，默认隐藏，避免把模型思考过程暴露给玩家。

import type { ChatMessage } from '../../types'
import type { StatsData, BenchmarkJob, LearningItem } from '../useChatRuntime'
import { getStoredDebug } from '../../settings'
import StatsCard from './StatsCard'
import ReviewCard from './ReviewCard'
import ProgressCard from './ProgressCard'
import Chips from './Chips'

interface Props {
  msg: ChatMessage
  busy: boolean
  onChip: (chip: string) => void
}

export default function MessageBubble({ msg, busy, onChip }: Props) {
  const params = msg.params ?? {}

  function inlineCard() {
    switch (msg.intent) {
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
        {msg.reasoning && getStoredDebug() ? (
          <details className="chat-msg-reasoning">
            <summary>🧠 思维链</summary>
            <div className="chat-msg-reasoning-body">{msg.reasoning}</div>
          </details>
        ) : null}
        {msg.speaker ? <div className="chat-msg-speaker">{msg.speaker}</div> : null}
        <div className="chat-msg-bubble-text">{msg.text}</div>
        {inlineCard()}
        {chips.length > 0 && <Chips chips={chips} disabled={busy} onPick={onChip} />}
      </div>
    </div>
  )
}
