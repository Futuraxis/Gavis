import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiGet, deleteCustomGame } from '../api/client'
import type { GameInfo } from '../types'

const FAMILY_LABELS: Record<string, string> = { grid: '网格', poker: '扑克', mahjong: '麻将', social: '社交', uno: 'UNO' }

/** 大厅卡片徽标：家族优先（社交类统一 🎭），再按 kind 分发——非棋盘类没有 board_size，
 * 旧逻辑会渲染「🎯 null×null」。 */
function kindBadge(game: GameInfo): string {
  if (game.family === 'social') return '🎭 社交'
  switch (game.kind) {
    case 'poker':
      return '🃏 扑克'
    case 'mahjong':
      return '🀄 麻将'
    case 'uno':
      return '🎴 UNO'
    default:
      return game.board_size != null ? `🎯 ${game.board_size}×${game.board_size}` : '🎲 对战'
  }
}

export default function LobbyPage() {
  const [games, setGames] = useState<GameInfo[]>([])
  const [error, setError] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    void loadGames()
  }, [])

  async function loadGames() {
    try {
      const data = await apiGet<{ games: GameInfo[] }>('/games')
      setGames(data.games)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  async function removeCustom(game: GameInfo) {
    if (deletingId !== null) return
    const confirmText = `确定删除自定义游戏「${game.display_name}」（id: ${game.game_id}）吗？`
    if (!window.confirm(`${confirmText}\n删除后不可恢复，其保存的数据会被一并移除。`)) return
    setDeletingId(game.game_id)
    setError(null)
    try {
      await deleteCustomGame(game.game_id)
      setGames((prev) => prev.filter((g) => g.game_id !== game.game_id))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div>
      <h1 className="page-title">游戏大厅</h1>
      <p className="page-sub">选择游戏，开始人机对战（自定义游戏卡片可点击 🗑 删除）</p>
      {error && <div className="error-banner">{error}</div>}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
        {games.map((game) => (
          <div key={game.game_id} className="card" onClick={() => navigate(`/battle/${game.game_id}`)}>
            <h3 style={{ marginBottom: 8 }}>
              {game.display_name}
              <span className="badge accent" style={{ marginLeft: 10 }}>
                {kindBadge(game)}
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
            {game.custom && (
              <div style={{ marginTop: 12, display: 'flex', justifyContent: 'flex-end' }}>
                <button
                  className="btn btn-danger lobby-delete-btn"
                  title="删除自定义游戏（不会影响内置游戏）"
                  disabled={deletingId !== null}
                  onClick={(e) => {
                    e.stopPropagation()
                    void removeCustom(game)
                  }}
                >
                  {deletingId === game.game_id ? '删除中…' : '🗑 删除'}
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
      {games.length === 0 && !error && (
        <p style={{ color: 'var(--muted)', marginTop: 16 }}>暂无可用游戏，请先在「创建游戏」页生成一个。</p>
      )}
    </div>
  )
}