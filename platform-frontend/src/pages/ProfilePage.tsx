import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getProfile } from '../api/client'
import { DEFAULT_PROFILE } from '../mock'
import { recentOf } from '../profile'
import type { Profile } from '../types'

const GAME_LABELS: Record<string, string> = {
  moon_chess: '月亮棋',
  stochastic_gomoku: '随机五子棋',
  texas_holdem: '德州扑克',
  mahjong_guangdong: '广东麻将',
  mahjong_hongzhong: '红中麻将',
  mahjong_blood: '血流成河',
  mahjong_sichuan: '四川麻将（血战到底）',
  mahjong_changsha: '长沙麻将（258将）',
  mahjong_taiwan: '台湾麻将（16张）',
}

export default function ProfilePage() {
  const [profile, setProfile] = useState<Profile>(DEFAULT_PROFILE)

  useEffect(() => {
    getProfile()
      .then(setProfile)
      .catch(() => setProfile(DEFAULT_PROFILE))
  }, [])

  // recent 可能缺失（旧档案/新契约）— recentOf 兜底为空表，避免整页崩溃
  const recentEntries = Object.entries(recentOf(profile))

  return (
    <div>
      <h1 className="page-title">个人中心</h1>
      <p className="page-sub">你的档案与战绩概览</p>

      <div className="panel" style={{ marginBottom: 18 }}>
        <div className="form-row">
          <label>昵称:</label>
          <span>{profile.nickname || '未设置'}</span>
        </div>
        <div className="form-row">
          <label>Agent 称呼:</label>
          <span>{profile.agent_call || '未设置'}</span>
        </div>
        <div style={{ marginTop: 12 }}>
          <Link className="btn" to="/settings">
            前往 Agent 设置
          </Link>
        </div>
      </div>

      <div className="panel">
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 14 }}>
          <h3 style={{ margin: 0 }}>战绩概览</h3>
          <Link className="btn" style={{ marginLeft: 'auto' }} to="/history">
            查看全部对局记录 →
          </Link>
        </div>
        {recentEntries.length === 0 ? (
          <p style={{ color: 'var(--muted)' }}>还没有战绩 — 去对战中心玩一局吧。</p>
        ) : (
          <table className="data">
            <thead>
              <tr>
                <th>游戏</th>
                <th>局数</th>
                <th>胜局</th>
                <th>胜率</th>
              </tr>
            </thead>
            <tbody>
              {recentEntries.map(([gameId, r]) => (
                <tr key={gameId}>
                  <td>{GAME_LABELS[gameId] ?? gameId}</td>
                  <td>{r.plays}</td>
                  <td>{r.wins}</td>
                  <td>{r.plays > 0 ? Math.round((r.wins / r.plays) * 100) : 0}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
