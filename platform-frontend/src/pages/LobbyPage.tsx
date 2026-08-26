import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiGet } from '../api/client'
import type { GameInfo } from '../types'

const FAMILY_LABELS: Record<string, string> = { grid: '网格', poker: '扑克', mahjong: '麻将', social: '社交' }

export default function LobbyPage() {
  const [games, setGames] = useState<GameInfo[]>([])
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    apiGet<{ games: GameInfo[] }>('/games')
      .then((data) => setGames(data.games))
      .catch((err: Error) => setError(err.message))
  }, [])

  return (
    <div>
      <h1 className="page-title">游戏大厅</h1>
      <p className="page-sub">选择游戏，开始人机对战</p>
      {error && <div className="error-banner">{error}</div>}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
        {games.map((game) => (
          <div key={game.game_id} className="card" onClick={() => navigate(`/battle/${game.game_id}`)}>
            <h3 style={{ marginBottom: 8 }}>
              {game.display_name}
              <span className="badge accent" style={{ marginLeft: 10 }}>
                {game.kind === 'poker' ? '🃏 扑克' : `🎯 ${game.board_size}×${game.board_size}`}
              </span>
              {game.custom && (
                <span className="badge accent" style={{ marginLeft: 6 }}>
                  🛠 自定义·{FAMILY_LABELS[game.family] ?? game.family}
                </span>
              )}
            </h3>
            <p style={{ color: 'var(--muted)', fontSize: 14, minHeight: 44 }}>{game.description}</p>
            <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
              <span className="badge">{game.difficulties.join(' / ')}</span>
              <span className="badge">{game.seat_label}: {game.seat_options.join(' / ')}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
