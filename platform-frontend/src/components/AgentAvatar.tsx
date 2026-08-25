import type { Mood } from '../types'

const MOOD_EMOJI: Record<Mood, string> = {
  happy: '😊',
  thinking: '🤔',
  sorry: '😔',
  neutral: '🙂',
}

interface Props {
  mood?: Mood
  thinking?: boolean
  size?: number
}

/** Agent 头像 — 月亮脸 + 右下角 mood 表情徽章；thinking 时徽章换成 spinner（复用全局 .spinner）。 */
export default function AgentAvatar({ mood = 'neutral', thinking = false, size = 72 }: Props) {
  return (
    <div className="agent-avatar" style={{ width: size, height: size, fontSize: size * 0.48 }}>
      <span className="agent-avatar-face">🌙</span>
      <span className="agent-avatar-mood">
        {thinking ? <span className="spinner agent-avatar-spinner" /> : MOOD_EMOJI[mood]}
      </span>
      {thinking && <span className="agent-avatar-thinking">思考中…</span>}
    </div>
  )
}
