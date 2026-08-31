import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { agentSay, apiGet, apiPost, matchHint } from '../api/client'
import { getStoredMuted, setStoredMuted } from '../settings'
import type {
  BoardSnapshot,
  ChatMessage,
  GameInfo,
  MahjongSnapshot,
  Mood,
  PersonaKey,
  PokerSnapshot,
  Snapshot,
} from '../types'
import AgentAvatar from '../components/AgentAvatar'
import BattleSetup, { type BattleConfig } from '../components/BattleSetup'
import ChatPanel from '../components/ChatPanel'
import ResultOverlay from '../components/ResultOverlay'
import VanishToast from '../components/VanishToast'
import { snapshotChatToMessages } from '../chat/snapshotChat'
import GomokuBoard from '../components/boards/GomokuBoard'
import MahjongTable from '../components/boards/MahjongTable'
import MoonBoard from '../components/boards/MoonBoard'
import PokerTable from '../components/boards/PokerTable'
import { FAMILY_BOARDS } from '../components/boards/familyBoards'

const DIFFICULTY_SHORT: Record<string, string> = { easy: '简单', normal: '普通', hard: '困难' }
const PERSONA_NAMES: Record<PersonaKey, string> = {
  gentle: '温柔陪伴', teacher: '认真教学', banter: '轻松吐槽', cold: '高冷竞技',
}
const PERSONA_GREETINGS: Record<PersonaKey, string> = {
  gentle: '你好呀，我们开始吧～放轻松，玩得开心最重要。',
  teacher: '我们开始吧。有不懂的随时问我，我会讲解每一步。',
  banter: '来啦？这局可别手下留情，我已经准备好整活了。',
  cold: '轮到你了。',
}

function uid(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

/** 从 grid 落子动作中取出落点; 非棋盘动作返回 null（用于乐观即时反馈）。 */
function gridCellOf(action: unknown): number | null {
  if (typeof action !== 'object' || action === null) return null
  const v = (action as Record<string, unknown>).cell_index
  if (typeof v === 'number') return Number.isInteger(v) ? v : null
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v)
    return Number.isInteger(n) ? n : null
  }
  return null
}

export default function BattlePage() {
  const { gameId = '' } = useParams()
  const [searchParams, setSearchParams] = useSearchParams()
  const [games, setGames] = useState<GameInfo[]>([])
  const [session, setSession] = useState<Snapshot | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stepKey, setStepKey] = useState(0)
  const [persona, setPersona] = useState<PersonaKey>('gentle')
  const [chat, setChat] = useState<ChatMessage[]>([])
  // 乐观落子: 人下棋后立即本地摆放棋子, 服务端权威快照返回后清除
  const [pendingCell, setPendingCell] = useState<number | null>(null)
  // 非法落子就地提示: 服务端拒绝后在对应格子上短暂闪烁, 随后自动清除
  const [invalidCell, setInvalidCell] = useState<number | null>(null)
  const invalidTimer = useRef<number | null>(null)
  const [muted, setMuted] = useState<boolean>(() => getStoredMuted())
  const navigate = useNavigate()

  const game = games.find((g) => g.game_id === gameId)
  const activeId = searchParams.get('game')
  const lastAgentMood: Mood | undefined = [...chat].reverse().find((m) => m.role === 'agent')?.mood

  useEffect(() => {
    apiGet<{ games: GameInfo[] }>('/games')
      .then((data) => setGames(data.games))
      .catch((err: Error) => setError(err.message))
  }, [])

  // 刷新页面时按 URL 中的会话 id 恢复对局
  useEffect(() => {
    if (!activeId || session) return
    apiPost<{ session: Snapshot }>('/match/state', { game_id: activeId })
      .then((data) => {
        setSession(data.session)
        drainChat(data.session)
      })
      .catch(() => {
        setError('对局已失效（服务器可能已重启），请重新开始')
        setSearchParams({})
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId, session, setSearchParams])

  function pushAgent(text: string, mood: Mood) {
    setChat((prev) => [...prev, { id: uid(), role: 'agent', text, mood, ts: Date.now() }])
  }

  function pushPlayer(text: string) {
    setChat((prev) => [...prev, { id: uid(), role: 'player', text, ts: Date.now() }])
  }

  /** 把后端快照里待投递的陪伴/教练消息（chat 增量）落进对话流。 */
  function drainChat(snap: Snapshot) {
    const msgs = snapshotChatToMessages(snap)
    if (msgs.length > 0) setChat((prev) => [...prev, ...msgs])
  }

  function greet(p: PersonaKey) {
    pushAgent(PERSONA_GREETINGS[p], 'happy')
  }

  async function start(config: BattleConfig) {
    setBusy(true)
    setError(null)
    try {
      const data = await apiPost<{ session: Snapshot }>('/match/start', {
        game_id: gameId,
        player_pid: config.playerPid,
        difficulty: config.difficulty,
        theme: config.theme,
        player_count: config.playerCount,
        persona: config.persona,
        hint_level: config.hintLevel,
        pacing: config.pacing,
        adaptive: config.adaptive,
        teaching: config.teaching,
      })
      setSession(data.session)
      setPendingCell(null)
      setInvalidCell(null)
      if (invalidTimer.current != null) window.clearTimeout(invalidTimer.current)
      setPersona(config.persona)
      setChat([])
      setSearchParams({ game: data.session.game_id })
      greet(config.persona)
      // 教练开场（teach_greet / teach_turn）与陪伴消息都在快照 chat 增量里。
      drainChat(data.session)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function move(action: unknown) {
    if (!session || session.over || busy) return
    setBusy(true)
    setError(null)
    if (invalidTimer.current != null) window.clearTimeout(invalidTimer.current)
    setInvalidCell(null)
    // 棋类即时反馈: 人落子先本地乐观摆放自己的棋子, 服务端权威快照返回后再覆盖。
    // 服务端 /match/move 在一趟请求里同时执行人 + AI 两步, 若不乐观渲染,
    // 人点击后要等 AI 思考完才一次性看到两个子 —— 这正是本修复要消除的体验。
    const cell = gridCellOf(action)
    if (cell != null && 'board' in session) setPendingCell(cell)
    try {
      const data = await apiPost<{ session: Snapshot }>('/match/move', {
        game_id: session.game_id,
        action: action,
      })
      setSession(data.session)
      drainChat(data.session) // 教练讲评（teach_move）与导读（teach_turn）
      setPendingCell(null)
      setStepKey((k) => k + 1)
    } catch (err) {
      setError((err as Error).message)
      setPendingCell(null)
      // 非法落子就地提示: 在被拒绝的格子上闪烁, 短暂后自动消失
      if (cell != null) {
        setInvalidCell(cell)
        if (invalidTimer.current != null) window.clearTimeout(invalidTimer.current)
        invalidTimer.current = window.setTimeout(() => setInvalidCell(null), 1400)
      }
    } finally {
      setBusy(false)
    }
  }

  function restart() {
    setSession(null)
    setPendingCell(null)
    setInvalidCell(null)
    if (invalidTimer.current != null) window.clearTimeout(invalidTimer.current)
    setSearchParams({})
    setError(null)
    setChat([])
  }

  function toggleMute() {
    const next = !muted
    setMuted(next)
    setStoredMuted(next)
  }

  async function handleSend(text: string) {
    pushPlayer(text)
    if (!session) return
    try {
      const data = await agentSay(session.game_id, 'chat', { message: text })
      pushAgent(data.text, data.mood)
    } catch {
      pushAgent('我在的，你继续说。', 'neutral')
    }
  }

  async function handleQuick(phrase: string) {
    if (phrase === '再来一局') {
      restart()
      return
    }
    if (phrase === '这步为什么？') {
      if (!session) return
      pushPlayer(phrase)
      try {
        const data = await matchHint(session.game_id, 'direction')
        pushAgent(data.text, data.mood)
      } catch {
        pushAgent('这一步…我建议先看看空位更多的方向。', 'thinking')
      }
    }
  }

  if (!game) {
    return <div className="page-sub">{error ?? `未知游戏: ${gameId}`}</div>
  }

  if (!session) {
    return <BattleSetup game={game} busy={busy} error={error} onStart={start} />
  }

  const interactive = !session.over && session.turn === session.player_pid && !busy

  // 渲染分发: 优先按 family（快照无 family 时按 game.family）经 FAMILY_BOARDS 查表，
  // 查不到回退现有 kind 分支; 其余状态机不动。
  const family = (session as { family?: string }).family ?? game.family ?? ''
  const FamilyBoard = FAMILY_BOARDS[family]

  // 棋类乐观/就地提示视图: 落子后把 pendingCell 的棋子临时叠加到快照上,
  // 被拒绝时在 invalidCell 格子就地闪烁; 服务端权威快照返回后全部清空。
  const boardView: Snapshot =
    (pendingCell != null || invalidCell != null) && 'board' in session
      ? {
          ...session,
          board:
            pendingCell != null
              ? session.board.map((p, i) => (i === pendingCell ? session.player_pid : p))
              : session.board,
          pending_cell: pendingCell,
          invalid_cell: invalidCell,
        }
      : session

  const board = FamilyBoard ? (
    <FamilyBoard snapshot={boardView} interactive={interactive} stepKey={stepKey} onMove={move} />
  ) : game.kind === 'mahjong' ? (
    <MahjongTable
      snapshot={session as MahjongSnapshot}
      interactive={interactive}
      onAction={(action) => move(action)}
    />
  ) : game.kind === 'poker' ? (
    <PokerTable
      snapshot={session as PokerSnapshot}
      interactive={interactive}
      onAction={(action) => move(action)}
    />
  ) : game.board_size === 3 ? (
    <MoonBoard snapshot={boardView as BoardSnapshot} interactive={interactive} onMove={(i) => move({ cell_index: i })} />
  ) : (
    <GomokuBoard
      snapshot={boardView as BoardSnapshot}
      interactive={interactive}
      stepKey={stepKey}
      onMove={(i) => move({ cell_index: i })}
    />
  )

  return (
    <div>
      <div className="battle-header">
        <h1 className="page-title">{game.display_name} · 人机对战</h1>
        <span
          className="badge accent"
          title={session.adaptive ? `自适应难度：AI 强度按你近 10 局胜率自动升降（本局实际强度 ${session.ai_strength ?? '—'}）` : undefined}
        >
          {session.adaptive
            ? `自适应 ⚙ 强度 ${session.ai_strength ?? '—'}`
            : DIFFICULTY_SHORT[session.difficulty] ?? session.difficulty}
        </span>
        <span className="badge">{game.seat_names[session.player_pid] ?? session.player_pid}</span>
        {session.teaching && (
          <span className="badge" title="教练能看到你的牌并推理；看不到 AI/对手的牌">
            📖 教学对局
          </span>
        )}
        {persona && <span className="badge">{PERSONA_NAMES[persona]}</span>}
        <span style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button className="btn" onClick={toggleMute}>
            {muted ? '🔇 已静音' : '🔊 对话'}
          </button>
          <button className="btn" onClick={restart}>
            重新开始
          </button>
        </span>
      </div>
      {error && <div className="error-banner">{error}</div>}

      <div className="battle-body">
        <div className="battle-main">
          {!session.over && (
            <div style={{ marginBottom: 12, color: 'var(--muted)', display: 'flex', gap: 16, alignItems: 'center' }}>
              {busy ? (
                <span>
                  <span className="spinner" /> AI 思考中…
                </span>
              ) : (
                <span>
                  {session.turn === session.player_pid
                    ? (session as MahjongSnapshot).phase === 'claim'
                      ? // claim 是响应别人打出的牌，不是你的出牌回合（四人轮转里
                        // 三家 AI 先被问碰/杠/胡，轮到你就是响应这张牌）。
                        `响应 ${game.seat_names[(session as MahjongSnapshot).last_discarder ?? ''] ?? '对方'} 的牌`
                      : '轮到你了'
                    : 'AI 回合'}
                </span>
              )}
              {'round' in session && session.round != null && <span>第 {session.round} 轮</span>}
              {session.evaluation && (
                <span
                  title={session.evaluation.mechanical_text ?? session.evaluation.summary}
                  style={{ cursor: 'help' }}
                >
                  📊 局势：{session.evaluation.summary}
                </span>
              )}
              {game.kind === 'board' && game.board_size === 9 && <VanishToast snapshot={session} />}
            </div>
          )}
          {board}
          {session.over && (
            <ResultOverlay
              snapshot={session}
              game={game}
              onReplay={() => navigate(`/review/${session.game_id}`)}
              onRestart={restart}
            />
          )}
        </div>

        {!muted && (
          <div className="battle-side">
            <div className="battle-agent">
              <AgentAvatar mood={lastAgentMood ?? 'neutral'} thinking={busy} size={84} />
              <div className="battle-agent-name">Gavis{persona ? ` · ${PERSONA_NAMES[persona]}` : ''}</div>
            </div>
            <ChatPanel messages={chat} disabled={!session} onSend={handleSend} onQuick={handleQuick} />
          </div>
        )}
      </div>
    </div>
  )
}
