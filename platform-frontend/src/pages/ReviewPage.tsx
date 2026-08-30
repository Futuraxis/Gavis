import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { apiGet, getReview } from '../api/client'
import { MOCK_MATCH_LOG, MOCK_REVIEW } from '../mock'
import type { BoardSnapshot, KeyNode, MahjongSnapshot, MatchLog, PokerSnapshot, ReviewReport } from '../types'
import GomokuBoard from '../components/boards/GomokuBoard'
import MahjongTable from '../components/boards/MahjongTable'
import MoonBoard from '../components/boards/MoonBoard'
import PokerTable from '../components/boards/PokerTable'
import { FAMILY_BOARDS } from '../components/boards/familyBoards'

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

const KIND_LABELS: Record<KeyNode['kind'], string> = {
  turning_point: '转折点',
  winning_move: '胜负手',
  blunder: '失误',
}

export default function ReviewPage() {
  const { matchId = '' } = useParams()
  const [match, setMatch] = useState<MatchLog | null>(null)
  const [review, setReview] = useState<ReviewReport | null>(null)
  const [idx, setIdx] = useState(0)
  const [mockMode, setMockMode] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    apiGet<{ match: MatchLog }>(`/history/${matchId}`)
      .then((d) => {
        setMatch(d.match)
        setIdx(0)
      })
      .catch(() => {
        setMockMode(true)
        setMatch(MOCK_MATCH_LOG)
        setIdx(0)
      })
    getReview(matchId)
      .then(setReview)
      .catch(() => setReview(MOCK_REVIEW))
  }, [matchId])

  const keyNodesByStep = useMemo(() => {
    const map = new Map<number, KeyNode>()
    for (const n of review?.key_nodes ?? []) map.set(n.step, n)
    return map
  }, [review])

  if (!match) {
    return <div className="page-sub">加载中…</div>
  }

  const entries = match.moves
  const entry = entries[idx]
  const snapshot = entry?.snapshot
  const lastIdx = entries.length - 1
  const won = match.winner === match.player_pid
  const title = match.winner == null ? '🤝 平局' : won ? '🎉 胜利' : '😢 失败'
  const currentKey = entry ? keyNodesByStep.get(entry.step) : undefined
  // 复盘快照渲染: 优先按 match.family 经 FAMILY_BOARDS（带 stepKey 的组件传 idx），
  // 缺失时回退现有 game_id 判断; 其余不动。
  const FamilyBoard = match.family != null ? FAMILY_BOARDS[match.family] ?? null : null

  const downloadReport = () => {
    if (!review) return
    const lines: string[] = [
      `Gavis 复盘报告 — ${GAME_LABELS[match.game_id] ?? match.game_id}`,
      `时间: ${new Date(match.started_at).toLocaleString('zh-CN')}`,
      `结果: ${match.winner == null ? '平局' : won ? '胜利' : '失败'}`,
      `总步数: ${entries.length}`,
      '',
      `【摘要】${review.summary}`,
      '',
      '【关键节点】',
      ...review.key_nodes.map((n) => `- 第 ${n.step + 1} 步 [${KIND_LABELS[n.kind] ?? n.kind}] ${n.why}`),
      '',
      `【改进建议】${review.improvement}`,
    ]
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `gavis-review-${match.match_id}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div>
      {mockMode && (
        <div className="success-banner">演示数据 — 后端 /history 或 /review 尚未接线。</div>
      )}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4 }}>
        <h1 className="page-title">复盘 · {GAME_LABELS[match.game_id] ?? match.game_id}</h1>
        <span className={`badge ${match.winner == null ? '' : won ? 'win' : 'lose'}`}>{title}</span>
      </div>
      <p className="page-sub">
        你执 {SEAT_LABELS[match.player_pid] ?? match.player_pid} · 难度 {match.difficulty} · 共 {entries.length} 步
      </p>

      <div className="review-layout">
        <aside className="panel review-timeline">
          <h4 style={{ marginBottom: 10 }}>步骤时间线</h4>
          {entries.length === 0 && <p style={{ color: 'var(--muted)' }}>无记录</p>}
          {entries.map((e) => {
            const key = keyNodesByStep.get(e.step)
            return (
              <div
                key={e.step}
                className={`review-step ${idx === e.step ? 'active' : ''} ${key ? key.kind : ''}`}
                onClick={() => setIdx(e.step)}
              >
                <span className="review-step-num">#{e.step + 1}</span>
                <span className="review-step-actor">{e.actor === 'ai' ? '🤖' : '🧑'}</span>
                <span className="review-step-action">{e.action}</span>
                {key && <span className="review-step-tag">{KIND_LABELS[key.kind]}</span>}
              </div>
            )
          })}
        </aside>

        <div className="review-board panel">
          {snapshot &&
            (FamilyBoard ? (
              <FamilyBoard snapshot={snapshot} interactive={false} stepKey={idx} />
            ) : match.game_id.startsWith('mahjong_') ? (
              <MahjongTable snapshot={snapshot as MahjongSnapshot} interactive={false} />
            ) : match.game_id === 'texas_holdem' ? (
              <PokerTable snapshot={snapshot as PokerSnapshot} interactive={false} />
            ) : match.game_id === 'moon_chess' ? (
              <MoonBoard snapshot={snapshot as BoardSnapshot} interactive={false} />
            ) : (
              <GomokuBoard snapshot={snapshot as BoardSnapshot} interactive={false} stepKey={idx} />
            ))}
          <div className="step-caption" style={{ marginTop: 12 }}>
            {entry
              ? `第 ${entry.step + 1} 步 · ${entry.actor === 'ai' ? '🤖 AI' : '🧑 你'} · ${entry.action}`
              : '对局开始'}
          </div>
          <div className="review-controls">
            <button className="btn" disabled={idx <= 0} onClick={() => setIdx((i) => Math.max(0, i - 1))}>
              ⏮ 上一步
            </button>
            <input
              className="replay-slider"
              type="range"
              min={0}
              max={lastIdx}
              value={idx}
              onChange={(e) => setIdx(Number(e.target.value))}
            />
            <button
              className="btn btn-primary"
              disabled={idx >= lastIdx}
              onClick={() => setIdx((i) => Math.min(lastIdx, i + 1))}
            >
              下一步 ⏭
            </button>
          </div>
        </div>

        <aside className="panel review-comments">
          <h4 style={{ marginBottom: 10 }}>Agent 评语</h4>
          {currentKey ? (
            <div className={`review-comment ${currentKey.kind}`}>
              <div className="review-comment-tag">{KIND_LABELS[currentKey.kind]}</div>
              <p>{currentKey.why}</p>
            </div>
          ) : (
            <p style={{ color: 'var(--muted)' }}>这一步没有特别标注。</p>
          )}
          {review && (
            <div className="review-summary">
              <div>
                <strong>摘要</strong>
                <p>{review.summary}</p>
              </div>
              <div>
                <strong>改进建议</strong>
                <p>{review.improvement}</p>
              </div>
            </div>
          )}
        </aside>
      </div>

      <div className="review-footer">
        <button className="btn" onClick={downloadReport} disabled={!review}>
          ⬇ 导出报告
        </button>
        <button className="btn btn-primary" onClick={() => navigate(`/battle/${match.game_id}`)}>
          🔁 再来一局
        </button>
      </div>
    </div>
  )
}
