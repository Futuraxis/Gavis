import { useEffect, useRef, useState } from 'react'
import type { ChatMessage, Mood } from '../types'

const MOOD_EMOJI: Record<Mood, string> = {
  happy: '😊',
  thinking: '🤔',
  sorry: '😔',
  neutral: '🙂',
}

const QUICK_PHRASES = ['再来一局', '这步为什么？']

interface Props {
  messages: ChatMessage[]
  muted?: boolean
  disabled?: boolean
  onSend?: (text: string) => void
  onQuick?: (phrase: string) => void
}

/** 对话面板 — Agent 与玩家气泡流 + 快捷短语 + 输入框。 */
export default function ChatPanel({ messages, muted = false, disabled = false, onSend, onQuick }: Props) {
  const [draft, setDraft] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages.length])

  function send(text: string) {
    const t = text.trim()
    if (!t || disabled) return
    onSend?.(t)
    setDraft('')
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <span>💬 对话</span>
        {muted && <span className="badge">已静音</span>}
      </div>
      <div className="chat-messages" ref={scrollRef}>
        {messages.length === 0 && <div className="chat-empty">Agent 会在这里陪你聊天</div>}
        {messages.map((m) => (
          <div key={m.id} className={`chat-bubble ${m.role}`}>
            {m.role === 'agent' && m.mood && <span className="chat-mood">{MOOD_EMOJI[m.mood]}</span>}
            <div className="chat-bubble-text">{m.text}</div>
          </div>
        ))}
      </div>
      <div className="chat-quick">
        {QUICK_PHRASES.map((p) => (
          <button key={p} className="btn" disabled={disabled} onClick={() => onQuick?.(p)}>
            {p}
          </button>
        ))}
      </div>
      <form
        className="chat-input"
        onSubmit={(e) => {
          e.preventDefault()
          send(draft)
        }}
      >
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="对 Agent 说…"
          disabled={disabled}
        />
        <button className="btn btn-primary" type="submit" disabled={disabled || !draft.trim()}>
          发送
        </button>
      </form>
    </div>
  )
}
