import { useEffect, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { apiGet, apiPost } from '../api/client'
import type { BoardSnapshot, GameInfo, MahjongSnapshot, PokerSnapshot, Snapshot } from '../types'
import BattleSetup from '../components/BattleSetup'
import ResultOverlay from '../components/ResultOverlay'
import VanishToast from '../components/VanishToast'
import GomokuBoard from '../components/boards/GomokuBoard'
import MahjongTable from '../components/boards/MahjongTable'
import MoonBoard from '../components/boards/MoonBoard'
import PokerTable from '../components/boards/PokerTable'

const SEAT_SHORT: Record<string, string> = {
  p_black: '黑棋', p_white: '白棋', p_sb: '小盲位', p_bb: '大盲位',
  p0: '庄家', p1: '下家', p2: '对家', p3: '上家',
}
const DIFFICULTY_SHORT: Record<string, string> = { easy: '简单', normal: '普通', hard: '困难' }

export default function BattlePage() {
  const { gameId = '' } = useParams()
  const [searchParams, setSearchParams] = useSearchParams()
  const [games, setGames] = useState<GameInfo[]>([])
  const [session, setSession] = useState<Snapshot | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stepKey, setStepKey] = useState(0)
  const navigate = useNavigate()

  const game = games.find((g) => g.game_id === gameId)
  const activeId = searchParams.get('game')

  useEffect(() => {
    apiGet<{ games: GameInfo[] }>('/games')
      .then((data) => setGames(data.games))
      .catch((err: Error) => setError(err.message))
  }, [])

  // 刷新页面时按 URL 中的会话 id 恢复对局
  useEffect(() => {
    if (!activeId || session) return
    apiPost<{ session: Snapshot }>('/match/state', { game_id: activeId })
      .then((data) => setSession(data.session))
      .catch(() => {
        setError('对局已失效（服务器可能已重启），请重新开始')
        setSearchParams({})
      })
  }, [activeId, session, setSearchParams])

  async function start(playerPid: string, difficulty: string, playerCount: number) {
    setBusy(true)
    setError(null)
    try {
      const data = await apiPost<{ session: Snapshot }>('/match/start', {
        game_id: gameId,
        player_pid: playerPid,
        difficulty: difficulty,
        player_count: playerCount,
      })
      setSession(data.session)
      setSearchParams({ game: data.session.game_id })
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
    try {
      const data = await apiPost<{ session: Snapshot }>('/match/move', {
        game_id: session.game_id,
        action: action,
      })
      setSession(data.session)
      setStepKey((k) => k + 1)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  function restart() {
    setSession(null)
    setSearchParams({})
    setError(null)
  }

  if (!game) {
    return <div className="page-sub">{error ?? `未知游戏: ${gameId}`}</div>
  }

  if (!session) {
    return <BattleSetup game={game} busy={busy} error={error} onStart={start} />
  }

  const interactive = !session.over && session.turn === session.player_pid && !busy

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 18 }}>
        <h1 className="page-title">{game.display_name} · 人机对战</h1>
        <span className="badge accent">{DIFFICULTY_SHORT[session.difficulty] ?? session.difficulty}</span>
        <span className="badge">{SEAT_SHORT[session.player_pid] ?? session.player_pid}</span>
        <button className="btn" style={{ marginLeft: 'auto' }} onClick={restart}>
          重新开始
        </button>
      </div>
      {error && <div className="error-banner">{error}</div>}
      {!session.over && (
        <div style={{ marginBottom: 12, color: 'var(--muted)', display: 'flex', gap: 16, alignItems: 'center' }}>
          {busy ? (
            <span>
              <span className="spinner" /> AI 思考中…
            </span>
          ) : (
            <span>{session.turn === session.player_pid ? '轮到你了' : 'AI 回合'}</span>
          )}
          {'round' in session && session.round != null && <span>第 {session.round} 轮</span>}
          {game.kind === 'board' && game.board_size === 9 && <VanishToast snapshot={session} />}
        </div>
      )}
      {game.kind === 'mahjong' ? (
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
        <MoonBoard snapshot={session as BoardSnapshot} interactive={interactive} onMove={(i) => move({ cell_index: i })} />
      ) : (
        <GomokuBoard
          snapshot={session as BoardSnapshot}
          interactive={interactive}
          stepKey={stepKey}
          onMove={(i) => move({ cell_index: i })}
        />
      )}
      {session.over && (
        <ResultOverlay
          snapshot={session}
          game={game}
          onReplay={() => navigate(`/replay/${session.game_id}`)}
          onRestart={restart}
        />
      )}
    </div>
  )
}
