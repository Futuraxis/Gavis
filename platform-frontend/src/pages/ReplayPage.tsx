import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { apiGet } from '../api/client'
import type { BoardSnapshot, MahjongSnapshot, MatchLog, PokerSnapshot } from '../types'
import GomokuBoard from '../components/boards/GomokuBoard'
import MahjongTable from '../components/boards/MahjongTable'
import MoonBoard from '../components/boards/MoonBoard'
import PokerTable from '../components/boards/PokerTable'

const GAME_LABELS: Record<string, string> = {
  moon_chess: '月亮棋',
  stochastic_gomoku: '随机五子棋',
  texas_holdem: '德州扑克',
  mahjong_guangdong: '广东麻将',
  mahjong_hongzhong: '红中麻将',
  mahjong_blood: '血战到底',
}

const SEAT_LABELS: Record<string, string> = {
  p_black: '黑棋',
  p_white: '白棋',
  p_sb: '小盲位',
  p_bb: '大盲位',
  p0: '庄家',
  p1: '下家',
  p2: '对家',
  p3: '上家',
}

export default function ReplayPage() {
  const { matchId = '' } = useParams()
  const [match, setMatch] = useState<MatchLog | null>(null)
  const [idx, setIdx] = useState(0)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    apiGet<{ match: MatchLog }>(`/history/${matchId}`)
      .then((data) => {
        setMatch(data.match)
        setIdx(0)
      })
      .catch((err: Error) => setError(err.message))
  }, [matchId])

  if (error) {
    return <div className="error-banner">{error}</div>
  }
  if (!match) {
    return <div className="page-sub">加载中…</div>
  }

  const entries = match.moves
  const entry = entries[idx]
  const snapshot = entry?.snapshot
  const lastIdx = entries.length - 1
  const won = match.winner === match.player_pid
  const title = match.winner == null ? '🤝 平局' : won ? '🎉 胜利' : '😢 失败'

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4 }}>
        <h1 className="page-title">回放 · {GAME_LABELS[match.game_id] ?? match.game_id}</h1>
        <span className={`badge ${match.winner == null ? '' : won ? 'win' : 'lose'}`}>{title}</span>
      </div>
      <p className="page-sub">
        你执 {SEAT_LABELS[match.player_pid] ?? match.player_pid} · 难度 {match.difficulty} · 共 {entries.length} 步
      </p>

      {snapshot &&
        (match.game_id.startsWith('mahjong_') ? (
          <MahjongTable snapshot={snapshot as MahjongSnapshot} interactive={false} />
        ) : match.game_id === 'texas_holdem' ? (
          <PokerTable snapshot={snapshot as PokerSnapshot} interactive={false} />
        ) : match.game_id === 'moon_chess' ? (
          <MoonBoard snapshot={snapshot as BoardSnapshot} interactive={false} />
        ) : (
          <GomokuBoard snapshot={snapshot as BoardSnapshot} interactive={false} stepKey={idx} />
        ))}

      <div className="replay-timeline panel">
        <div className="step-caption">
          {entry
            ? `第 ${entry.step + 1} 步 · ${entry.actor === 'ai' ? '🤖 AI' : '🧑 你'} · ${entry.action}`
            : '对局开始'}
          {entry && 'last_vanish' in entry.snapshot && entry.snapshot.last_vanish != null && (
            <span style={{ marginLeft: 10 }}>💨 棋子被抹去了</span>
          )}
        </div>
        <input
          className="replay-slider"
          type="range"
          min={0}
          max={lastIdx}
          value={idx}
          onChange={(e) => setIdx(Number(e.target.value))}
        />
        <div style={{ display: 'flex', gap: 10, justifyContent: 'center' }}>
          <button className="btn" disabled={idx <= 0} onClick={() => setIdx((i) => Math.max(0, i - 1))}>
            ⏮ 上一步
          </button>
          <button
            className="btn btn-primary"
            disabled={idx >= lastIdx}
            onClick={() => setIdx((i) => Math.min(lastIdx, i + 1))}
          >
            下一步 ⏭
          </button>
        </div>
      </div>
    </div>
  )
}
