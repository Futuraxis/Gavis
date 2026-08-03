import { useState } from 'react'
import type { GameInfo } from '../types'

interface Props {
  game: GameInfo
  busy: boolean
  error: string | null
  onStart: (playerPid: string, difficulty: string) => void
}

const SEAT_LABELS: Record<string, string> = {
  p_black: '黑棋 ⚫',
  p_white: '白棋 ⚪',
  p_sb: '小盲位 (先手)',
  p_bb: '大盲位 (后手)',
  random: '随机 🎲',
}

const DIFFICULTY_LABELS: Record<string, string> = {
  easy: '简单 😊',
  normal: '普通 🤔',
  hard: '困难 😈',
}

export default function BattleSetup({ game, busy, error, onStart }: Props) {
  const [playerPid, setPlayerPid] = useState('random')
  const [difficulty, setDifficulty] = useState('normal')

  return (
    <div className="panel" style={{ maxWidth: 520 }}>
      <h1 className="page-title">{game.display_name}</h1>
      <p className="page-sub">{game.description}</p>
      {error && <div className="error-banner">{error}</div>}
      <div className="form-row">
        <label>{game.seat_label}:</label>
        <select value={playerPid} onChange={(e) => setPlayerPid(e.target.value)}>
          <option value="random">{SEAT_LABELS.random}</option>
          {game.seat_options.map((pid) => (
            <option key={pid} value={pid}>
              {SEAT_LABELS[pid] ?? pid}
            </option>
          ))}
        </select>
      </div>
      <div className="form-row">
        <label>难度:</label>
        <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
          {game.difficulties.map((d) => (
            <option key={d} value={d}>
              {DIFFICULTY_LABELS[d] ?? d}
            </option>
          ))}
        </select>
      </div>
      <button className="btn btn-primary" disabled={busy} onClick={() => onStart(playerPid, difficulty)}>
        {busy ? '加载中…' : '开始对局'}
      </button>
    </div>
  )
}
