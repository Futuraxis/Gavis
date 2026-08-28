// ChatPage — 「对话即一切」主界面。消息流 + 常驻对局面板 + 输入区。
// 浅色极简：无侧栏、无按钮堆叠——一切从一句自然语言开始。
// 平台界面仍可访问：右上「打开平台界面」或口令「打开平台界面/平台模式」。

import { useEffect, useRef, useState } from 'react'
import type { ChatMessage, GameInfo } from '../types'
import type { BattleConfig } from '../components/BattleSetup'
import { useChatRuntime } from './useChatRuntime'
import { openPlatform } from './sessionStore'
import MessageBubble from './components/MessageBubble'
import InlineBoard from './components/InlineBoard'
import Chips from './components/Chips'

const WELCOME_TEXT =
  '你好，我是 Gavis。对局、战绩、创建游戏、评测——一句话就行。\n' +
  '试试：· “玩月亮棋” · “来一局德州扑克” · “继续上一局” · “看战绩” · “创建游戏” · “打开平台界面”'

const WELCOME_CHIPS = ['玩月亮棋', '来一局德州扑克', '看战绩', '创建游戏', '打开平台界面']

const GAME_CHIPS = ['这步怎么走？', '再来一局', '看战绩']

function welcomeMessage(): ChatMessage {
  return { id: 'welcome', role: 'agent', text: WELCOME_TEXT, mood: 'happy', ts: Date.now() }
}

export default function ChatPage() {
  const rt = useChatRuntime()
  const { messages, busy, error, games, activeSession, activeGameInfo } = rt
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  const lastAgent = [...messages].reverse().find((m) => m.role === 'agent')

  function chips(): string[] {
    if (lastAgent?.intent === 'clarify') {
      const c = (lastAgent.params?.chips ?? []) as string[]
      return c.length ? c : WELCOME_CHIPS
    }
    if (activeSession) return GAME_CHIPS
    if (messages.length <= 1) return WELCOME_CHIPS
    return []
  }

  function send(text: string) {
    if (!text.trim() || busy) return
    void rt.send(text)
  }

  function startInChat(gameId: string, config: BattleConfig) {
    void rt.startSession(gameId, config)
  }

  // 对局点击（棋盘快路径，不经 LLM）
  function onBoardMove(action: unknown) {
    void rt.moveAction(action)
  }

  return (
    <div className="chat-app">
      <header className="chat-header">
        <div className="chat-brand">
          <span className="chat-brand-dot" />
          Gavis
          <span className="chat-brand-sub">对话即一切</span>
        </div>
        <div className="chat-header-right">
          {busy && (<span className="chat-thinking">思考中…</span>)}
          {activeSession && !activeSession.over && (
            <button className="chat-header-btn" onClick={() => void rt.clearSession()}>
              结束对局
            </button>
          )}
          <button className="chat-header-btn" onClick={openPlatform}>
            打开平台界面
          </button>
        </div>
      </header>

      <div className="chat-scroll" ref={scrollRef}>
        {error && <div className="chat-error-banner">{error}</div>}
        {messages.map((m) => (
          <MessageBubble
            key={m.id}
            msg={m}
            games={games}
            busy={busy}
            onStart={startInChat}
            onCreated={(g: GameInfo) => rt.notifyCreated(g)}
            onChip={send}
          />
        ))}
        {messages.length === 0 && <MessageBubble msg={welcomeMessage()} games={games} busy={busy} onStart={startInChat} onCreated={() => undefined} onChip={send} />}
      </div>

      {activeSession && (
        <div className="chat-board-region">
          <InlineBoard
            snapshot={activeSession}
            game={activeGameInfo}
            busy={busy}
            onMove={onBoardMove}
            onRestart={() => void rt.send('再来一局')}
          />
        </div>
      )}

      <div className="chat-input-region">
        {chips().length > 0 && <Chips chips={chips()} disabled={busy} onPick={send} />}
        <div className="chat-input-row">
          <input
            className="chat-input"
            value={input}
            placeholder="说点什么…（对局中也可直接点棋盘落子）"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                const t = input
                setInput('')
                send(t)
              }
            }}
            disabled={busy}
          />
          <button className="chat-send" disabled={busy || !input.trim()} onClick={() => {
            const t = input
            setInput('')
            send(t)
          }}>
            发送
          </button>
        </div>
      </div>
    </div>
  )
}