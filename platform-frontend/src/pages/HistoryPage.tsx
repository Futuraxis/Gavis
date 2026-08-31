import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiGet } from '../api/client'
import type { MatchMeta } from '../types'

const GAME_LABELS: Record<string, string> = {
  moon_chess: '月亮棋',
  stochastic_gomoku: '随机五子棋',
  texas_holdem: '德州扑克',
}

const SEAT_LABELS: Record<string, string> = {
  p_black: '黑棋',
  p_white: '白棋',
  p_sb: '小盲位',
  p_bb: '大盲位',
}

const DIFFICULTY_LABELS: Record<string, string> = { easy: '简单', normal: '普通', hard: '困难' }

export default function HistoryPage() {
  const [matches, setMatches] = useState<MatchMeta[]>([])
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    apiGet<{ matches: MatchMeta[] }>('/history')
      .then((data) => setMatches(data.matches))
      .catch((err: Error) => setError(err.message))
  }, [])

  return (
    <div>
      <h1 className="page-title">对局记录</h1>
      <p className="page-sub">点击任意对局查看逐步回放</p>
      {error && <div className="error-banner">{error}</div>}
      {matches.length === 0 && !error && (
        <div className="panel" style={{ color: 'var(--muted)', textAlign: 'center' }}>
          还没有对局记录 — 去对战中心玩一局吧 ⚔️
        </div>
      )}
      {matches.length > 0 && (
        <div className="panel">
          <table className="data">
            <thead>
              <tr>
                <th>时间</th>
                <th>游戏</th>
                <th>你的座位</th>
                <th>难度</th>
                <th>步数</th>
                <th>结果</th>
              </tr>
            </thead>
            <tbody>
              {matches.map((m) => {
                // 后端已把阵营胜者解析为玩家视角 won（社交游戏 winner=undercover 等；
                // 旧记录缺省 → 回退 pid 比较）。
                const won = m.won ?? (m.winner != null && m.winner === m.player_pid)
                const badge = m.winner == null ? '' : won ? 'win' : 'lose'
                const label = m.winner == null ? '平局' : won ? '胜利 🎉' : '失败'
                return (
                  <tr key={m.match_id} className="clickable" onClick={() => navigate(`/review/${m.match_id}`)}>
                    <td>{new Date(m.started_at).toLocaleString('zh-CN')}</td>
                    <td>{GAME_LABELS[m.game_id] ?? m.game_id}</td>
                    <td>{SEAT_LABELS[m.player_pid] ?? m.player_pid}</td>
                    <td>
                      {m.adaptive
                        ? `自适应 ⚙ 强度 ${m.ai_strength ?? '—'}`
                        : DIFFICULTY_LABELS[m.difficulty] ?? m.difficulty}
                    </td>
                    <td>{m.moves}</td>
                    <td>
                      <span className={`badge ${badge}`}>{label}</span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
