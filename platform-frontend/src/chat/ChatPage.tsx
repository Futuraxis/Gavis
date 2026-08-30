// ChatPage — 「对话即一切」主界面。
// 纯对话：单栏居中极简流；对局中：左右分栏——棋盘主区（左）+ 对话侧栏（右），
// 窄屏（≤900px）自动回落为上下排布（棋盘在上、对话输入在下）。
// 对局中也可「收起界面」专心对话：棋盘整个让位给对话流（对局不结束、后台
// 继续，顶部提示条 + 顶栏按钮随时一键回到棋盘）——关不关由用户自己选。
//
// 对话管理与存档：顶栏「对话」打开抽屉——历史会话列表（切换/重命名/归档/
// 删除/导出 Markdown），「＋ 新对话」随时另起一段；存档在后端
// data/conversations/，刷新/重开页面自动恢复当前会话。

import { useEffect, useRef, useState } from 'react'
import type { ChatMessage, ConversationMeta, GameInfo } from '../types'
import type { BattleConfig } from '../components/BattleSetup'
import { getConversation } from '../api/client'
import { useChatRuntime } from './useChatRuntime'
import { loadChatStore, openPlatform, saveChatStore } from './sessionStore'
import MessageBubble from './components/MessageBubble'
import InlineBoard from './components/InlineBoard'
import Chips from './components/Chips'

// 开场白（WELCOME_TEXT/welcomeMessage）在 useChatRuntime 里作为第一条消息
// 常驻对话流——不再用「无消息时的兜底渲染」，首次输入后依然可见。

const WELCOME_CHIPS = ['玩月亮棋', '来一局德州扑克', '看战绩', '创建游戏', '打开平台界面']

const GAME_CHIPS = ['这步怎么走？', '再来一局', '看战绩']

/** 会话条目的时间标签：今天只显示时刻，更早的带月/日。 */
function convWhen(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const now = new Date()
  const sameDay = d.toDateString() === now.toDateString()
  const hm = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  return sameDay ? hm : `${d.getMonth() + 1}/${d.getDate()} ${hm}`
}

/** 把一段消息流导出为 Markdown 文件（本地下载，不经后端）。 */
function exportMarkdown(title: string, messages: ChatMessage[]): void {
  const lines = [`# Gavis 对话存档 · ${title}`, '']
  for (const m of messages) {
    const who = m.role === 'player' ? '你' : 'Gavis'
    const when = new Date(m.ts).toLocaleString()
    lines.push(`**${who}** · ${when}`, '', m.text, '')
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `gavis-对话-${title || '未命名'}.md`
  a.click()
  URL.revokeObjectURL(url)
}

export default function ChatPage() {
  const rt = useChatRuntime()
  const { messages, busy, error, games, activeSession, activeGameInfo, conversationId, conversations } = rt
  const [input, setInput] = useState('')
  // 「不想玩了 / 想专心聊」：收起对局界面（对局不结束、后台继续，随时展开）。
  // 选择持久化——刷新/恢复对局后仍尊重用户上次的取舍。
  const [boardCollapsed, setBoardCollapsed] = useState(() => loadChatStore().boardCollapsed)
  const [convOpen, setConvOpen] = useState(false)
  const [showArchived, setShowArchived] = useState(false)
  const scrollRef = useRef<HTMLDivElement | null>(null)

  function setBoardCollapsedPersist(collapsed: boolean) {
    setBoardCollapsed(collapsed)
    saveChatStore({ boardCollapsed: collapsed })
  }

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
    setBoardCollapsedPersist(false) // 对话里开出新局 → 自动展开棋盘（收起是用户主动的选择，新局默认看得见）
    void rt.startSession(gameId, config)
  }

  // 对局点击（棋盘快路径，不经 LLM）
  function onBoardMove(action: unknown) {
    void rt.moveAction(action)
  }

  // ── 对话存档抽屉操作 ──────────────────────────────────────────

  const activeConvs = conversations.filter((c) => !c.archived)
  const archivedConvs = conversations.filter((c) => c.archived)

  function switchTo(conv: ConversationMeta) {
    setConvOpen(false)
    void rt.switchConversation(conv.conv_id)
  }

  function renameConv(conv: ConversationMeta) {
    const title = window.prompt('给这段对话起个名字：', conv.title || '')?.trim()
    if (title) void rt.renameConversation(conv.conv_id, title)
  }

  function archiveConv(conv: ConversationMeta) {
    void rt.setConversationArchived(conv.conv_id, !conv.archived)
  }

  function deleteConv(conv: ConversationMeta) {
    if (window.confirm(`删除对话「${conv.title || '未命名'}」？删除后无法恢复。`)) {
      void rt.removeConversation(conv.conv_id)
    }
  }

  async function exportConv(conv: ConversationMeta) {
    try {
      // 当前会话直接导出内存流（可能比后端存档新最多一个防抖周期）。
      const msgs =
        conv.conv_id === conversationId
          ? messages.filter((m) => m.id !== 'welcome')
          : (await getConversation(conv.conv_id)).messages
      exportMarkdown(conv.title || '未命名', msgs)
    } catch (err) {
      window.alert(`导出失败：${(err as Error).message}`)
    }
  }

  function newConversation() {
    setConvOpen(false)
    rt.startNewConversation()
  }

  function convItem(conv: ConversationMeta) {
    const isActive = conv.conv_id === conversationId
    return (
      <div
        key={conv.conv_id}
        className={`chat-conv-item${isActive ? ' active' : ''}${conv.archived ? ' archived' : ''}`}
        onClick={() => switchTo(conv)}
      >
        <div className="chat-conv-item-main">
          <div className="chat-conv-item-title">{conv.title || '新对话'}</div>
          <div className="chat-conv-item-meta">
            {convWhen(conv.updated_at)} · {conv.message_count} 条{conv.preview ? ` · ${conv.preview}` : ''}
          </div>
        </div>
        <div className="chat-conv-item-actions" onClick={(e) => e.stopPropagation()}>
          <button title="重命名" onClick={() => renameConv(conv)}>
            改名
          </button>
          <button title={conv.archived ? '取消归档' : '归档（移出列表但保留存档）'} onClick={() => archiveConv(conv)}>
            {conv.archived ? '恢复' : '归档'}
          </button>
          <button title="导出 Markdown 存档" onClick={() => void exportConv(conv)}>
            导出
          </button>
          <button title="删除（不可恢复）" className="danger" onClick={() => deleteConv(conv)}>
            删除
          </button>
        </div>
      </div>
    )
  }

  // 收起界面后「后台对局」的一句话状态（提示条用；与 InlineBoard 头部口径一致）。
  function bgMatchStatus(): string {
    if (!activeSession) return ''
    if (activeSession.over) {
      if (activeSession.winner === activeSession.player_pid) return '你赢了 🎉'
      return activeSession.winner ? '这一局输了' : '平局'
    }
    if (busy) return 'AI 思考中…'
    // claim 是响应别人打出的牌（碰/杠/过），不是出牌回合 —— 不显示「轮到你了」。
    if (activeSession.turn === activeSession.player_pid) {
      const phase = (activeSession as { phase?: string | null }).phase
      return phase === 'claim' ? '响应对方打出的牌' : '轮到你了'
    }
    return 'AI 回合'
  }

  // 消息流 + 输入区：两种布局（纯对话 / 对局分栏）复用同一份 JSX。
  const messageStream = (
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
    </div>
  )

  const inputRegion = (
    <div className="chat-input-region">
      {chips().length > 0 && <Chips chips={chips()} disabled={busy} onPick={send} />}
      <div className="chat-input-row">
        <input
          className="chat-inputbox"
          value={input}
          placeholder={
            activeSession && !boardCollapsed
              ? '说点什么…（对局中也可直接点棋盘落子）'
              : activeSession
                ? '专心对话…（对局还在后台，想下棋就展开界面）'
                : '说点什么…'
          }
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
  )

  return (
    <div className="chat-app">
      <header className="chat-topbar">
        <div className="chat-brand">
          <span className="chat-brand-dot" />
          Gavis
          <span className="chat-brand-sub">对话即一切</span>
        </div>
        <div className="chat-header-right">
          {busy && (<span className="chat-thinking">思考中…</span>)}
          {activeSession && (
            <button
              className="chat-header-btn"
              onClick={() => setBoardCollapsedPersist(!boardCollapsed)}
              title={boardCollapsed ? '展开棋盘，回到对局' : '收起棋盘专心对话（对局不结束，后台继续）'}
            >
              {boardCollapsed ? '展开界面' : '收起界面'}
            </button>
          )}
          {activeSession && !activeSession.over && (
            <button className="chat-header-btn" onClick={() => void rt.clearSession()}>
              结束对局
            </button>
          )}
          <button className="chat-header-btn" onClick={() => setConvOpen(true)}>
            对话
          </button>
          <button className="chat-header-btn" onClick={openPlatform}>
            打开平台界面
          </button>
        </div>
      </header>

      {activeSession && !boardCollapsed ? (
        // 对局布局：棋盘主区（左，占满剩余宽度）+ 对话侧栏（右，常驻消息流与输入）。
        <div className="chat-match">
          <main className="chat-match-board">
            <InlineBoard
              snapshot={activeSession}
              game={activeGameInfo}
              busy={busy}
              onMove={onBoardMove}
              onRestart={() => void rt.send('再来一局')}
            />
          </main>
          <aside className="chat-match-dock">
            {messageStream}
            {inputRegion}
          </aside>
        </div>
      ) : (
        // 纯对话布局：消息流 + 输入区纵向铺满。收起界面后走同一分支——棋盘不
        // 渲染、对局留后台；顶部提示条常驻（游戏名 + 实时状态），点一下即回棋盘。
        <>
          {activeSession && (
            <button
              className={`chat-bgmatch${activeSession.over ? ' done' : ''}`}
              onClick={() => setBoardCollapsedPersist(false)}
              title="展开棋盘"
            >
              <span className="chat-bgmatch-dot" />
              <span className="chat-bgmatch-game">{activeGameInfo?.display_name ?? activeSession.game_id}</span>
              <span className="chat-bgmatch-sep">·</span>
              <span className="chat-bgmatch-status">{bgMatchStatus()}</span>
              <span className="chat-bgmatch-action">展开界面</span>
            </button>
          )}
          {messageStream}
          {inputRegion}
        </>
      )}

      {/* 对话存档抽屉：点击遮罩关闭；列表/切换/重命名/归档/删除/导出。 */}
      {convOpen && (
        <div className="chat-conv-overlay" onClick={() => setConvOpen(false)}>
          <aside className="chat-conv-drawer" onClick={(e) => e.stopPropagation()}>
            <header className="chat-conv-header">
              <span className="chat-conv-title">对话记录</span>
              <div className="chat-conv-header-actions">
                <button className="chat-conv-new" onClick={newConversation}>
                  ＋ 新对话
                </button>
                <button className="chat-conv-close" onClick={() => setConvOpen(false)} title="关闭">
                  ✕
                </button>
              </div>
            </header>
            <div className="chat-conv-list">
              {activeConvs.length === 0 && archivedConvs.length === 0 && (
                <div className="chat-conv-empty">
                  还没有存档的对话——聊过一段后自动存档，刷新也不会丢。
                </div>
              )}
              {activeConvs.map(convItem)}
              {archivedConvs.length > 0 && (
                <>
                  <button className="chat-conv-archived-toggle" onClick={() => setShowArchived((v) => !v)}>
                    {showArchived ? '▾' : '▸'} 已归档（{archivedConvs.length}）
                  </button>
                  {showArchived && archivedConvs.map(convItem)}
                </>
              )}
            </div>
          </aside>
        </div>
      )}
    </div>
  )
}
