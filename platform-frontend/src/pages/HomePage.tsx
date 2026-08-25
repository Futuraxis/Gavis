import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiGet, getProfile } from '../api/client'
import { DEFAULT_PROFILE, MOCK_GAMES } from '../mock'
import type { ActiveSession, GameInfo, PersonaKey, Profile } from '../types'
import AgentAvatar from '../components/AgentAvatar'

const PERSONA_NAMES: Record<PersonaKey, string> = {
  gentle: '温柔陪伴',
  teacher: '认真教学',
  banter: '轻松吐槽',
  cold: '高冷竞技',
}

export default function HomePage() {
  const [profile, setProfile] = useState<Profile>(DEFAULT_PROFILE)
  const [games, setGames] = useState<GameInfo[]>([])
  const [active, setActive] = useState<ActiveSession[]>([])

  useEffect(() => {
    getProfile()
      .then(setProfile)
      .catch(() => setProfile(DEFAULT_PROFILE))
    apiGet<{ games: GameInfo[] }>('/games')
      .then((d) => setGames(d.games))
      .catch(() => setGames(MOCK_GAMES))
    apiGet<{ sessions: ActiveSession[] }>('/match/active')
      .then((d) => setActive(d.sessions))
      .catch(() => setActive([]))
  }, [])

  const call = profile.agent_call || profile.nickname || '朋友'
  const recentGames = games.filter((g) => profile.recent[g.game_id] != null)
  const lastGameId = recentGames[0]?.game_id ?? 'moon_chess'
  // 有未结束会话 → 真·继续上一局（恢复 ?game=<id>）；否则按最近游戏新开
  const resume = active[0]
  const resumeLink = resume ? `/battle/${resume.game}?game=${resume.game_id}` : `/battle/${lastGameId}`

  return (
    <div>
      <div className="home-hero panel">
        <AgentAvatar mood="happy" size={88} />
        <div className="home-hero-text">
          <h1 className="page-title">你好，{call} 👋</h1>
          <p className="page-sub" style={{ marginBottom: 0 }}>
            我是 Gavis，今天想玩点什么？{PERSONA_NAMES[profile.default_persona]}模式已就绪。
          </p>
        </div>
      </div>

      <div className="home-grid">
        <Link className="card" to={resumeLink}>
          <h3>▶ 继续上一局</h3>
          <p style={{ color: 'var(--muted)', fontSize: 14 }}>
            {resume
              ? `未结束 · ${resume.display_name} · 第 ${resume.step + 1} 手`
              : '从上次的游戏接着玩'}
          </p>
        </Link>
        <Link className="card" to="/lobby">
          <h3>🏛️ 游戏大厅</h3>
          <p style={{ color: 'var(--muted)', fontSize: 14 }}>浏览全部游戏与开局配置</p>
        </Link>
        <Link className="card" to="/profile">
          <h3>👤 个人中心</h3>
          <p style={{ color: 'var(--muted)', fontSize: 14 }}>档案、战绩与对局记录</p>
        </Link>
      </div>

      {active.length > 0 && (
        <>
          <h2 className="home-section-title">未结束的对局（可继续）</h2>
          <div className="home-grid">
            {active.map((s) => (
              <Link key={s.game_id} className="card" to={`/battle/${s.game}?game=${s.game_id}`}>
                <h3>{s.display_name}</h3>
                <p style={{ color: 'var(--muted)', fontSize: 14 }}>
                  第 {s.step + 1} 手 · {s.difficulty} ·{' '}
                  {s.persona ? PERSONA_NAMES[s.persona] : '无表达'}
                </p>
              </Link>
            ))}
          </div>
        </>
      )}

      {recentGames.length > 0 && (
        <>
          <h2 className="home-section-title">最近玩过的游戏</h2>
          <div className="home-grid">
            {recentGames.map((g) => {
              const r = profile.recent[g.game_id]
              return (
                <Link key={g.game_id} className="card" to={`/battle/${g.game_id}`}>
                  <h3>{g.display_name}</h3>
                  <p style={{ color: 'var(--muted)', fontSize: 14 }}>
                    {r.wins} 胜 / {r.plays} 局
                  </p>
                </Link>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}