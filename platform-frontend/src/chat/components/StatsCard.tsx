// StatsCard — 战绩卡（内嵌于聊天消息；数据来自 /api/history）。

import type { MatchMeta } from '../../types'

interface Props {
  matches: MatchMeta[]
  wins: number
  plays: number
}

const SEATS: Record<string, string> = {
  p_black: '黑棋', p_white: '白棋', p_sb: '小盲位', p_bb: '大盲位',
  p0: '庄家', p1: '下家', p2: '对家', p3: '上家',
}

function resultOf(m: MatchMeta): string {
  if (m.winner === m.player_pid) return '胜'
  if (m.winner === m.ai_pid) return '负'
  return '平'
}

export default function StatsCard({ matches, wins, plays }: Props) {
  if (!plays) {
    return (
      <div className="chat-card chat-card-muted">
        还没有对局记录。说一声“玩月亮棋”或“来一局德州扑克”开始第一局吧。
      </div>
    )
  }
  return (
    <div className="chat-card">
      <div className="chat-card-title">最近 {plays} 局 · {wins} 胜</div>
      <ul className="chat-stats-list">
        {matches.map((m) => (
          <li key={m.match_id} className="chat-stats-row">
            <span className="chat-stats-game">{m.game_id}</span>
            <span className={`chat-stats-result ${resultOf(m) === '胜' ? 'win' : resultOf(m) === '负' ? 'lose' : ''}`}>
              {resultOf(m)}
            </span>
            <span className="chat-stats-seat">{SEATS[m.player_pid] ?? m.player_pid}</span>
            <span className="chat-stats-date">{m.started_at?.slice(0, 10) ?? ''}</span>
          </li>
        ))}
      </ul>
      <div className="chat-card-hint">想复盘某局？直接说一声“复盘上一局”。</div>
    </div>
  )
}