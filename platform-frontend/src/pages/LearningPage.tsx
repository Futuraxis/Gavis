import { useEffect, useState } from 'react'
import { applyLearning, getLearningStatus, setLearningConfig } from '../api/client'
import { usePolling } from '../hooks/usePolling'
import type { LearningStatus } from '../types'

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

const REASON_LABELS: Record<string, string> = {
  ok: '✅ 已发布',
  insufficient: '⏳ 样本不足',
  unchanged: '➖ 模型未变化',
  rejected: '⛔ 门禁未通过（保留旧版）',
  disabled: '🚫 已停用',
  error: '⚠️ 失败',
}

function ModelBlock({ status }: { status: LearningStatus }) {
  const model = status.model
  if (!model) {
    return (
      <>
        <td />
        <td />
        <td />
        <td />
        <td>
          <span className="badge accent">
            {status.pending ? '已有足够样本，等待 apply' : '尚无已发布模型'}
          </span>
        </td>
      </>
    )
  }
  const gate = model.gate
  return (
    <>
      <td>v{model.version}</td>
      <td>{model.samples}</td>
      <td>{model.coverage}</td>
      <td>{model.published_at ? new Date(model.published_at).toLocaleString('zh-CN') : '-'}</td>
      <td>
        {gate
          ? `候选 ${Math.round(gate.candidate_win_rate * 100)}% vs 基准 ${Math.round(gate.baseline_win_rate * 100)}%`
          : '-'}
      </td>
    </>
  )
}

export default function LearningPage() {
  const [status, setStatus] = useState<LearningStatus[]>([])
  const [results, setResults] = useState<Record<string, string> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function refresh() {
    getLearningStatus()
      .then((data) => setStatus(data.learning))
      .catch((err: Error) => setError(err.message))
  }

  usePolling(refresh, 5000, status.some((s) => s.pending))

  async function runApply(gameId?: string) {
    setBusy(true)
    setError(null)
    try {
      const data = await applyLearning(gameId)
      const map: Record<string, string> = {}
      if ('result' in data) {
        const r = data.result
        map[r.game_id] = REASON_LABELS[r.reason] ?? r.reason
      } else {
        for (const r of data.results) {
          map[r.game_id] = REASON_LABELS[r.reason] ?? r.reason
        }
      }
      setResults(map)
      refresh()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function toggle(gameId: string, enabled: boolean) {
    try {
      await setLearningConfig(gameId, enabled)
      refresh()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  return (
    <div>
      <div className="page-header">
        <h1>🧠 在线学习</h1>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn" onClick={() => runApply()} disabled={busy}>
            {busy ? '评估中…' : '立即学习（全部）'}
          </button>
          <button className="btn ghost" onClick={refresh}>
            刷新
          </button>
        </div>
      </div>
      <p className="muted">
        在线学习收集真实对局中人类的决策（按信息集聚合），把候选对手模型与当前模型做短赛门禁（固定随机种子、双方换边），
        不回归才发布。德州扑克默认启用，发布后新开的对局 AI 会使用学习到的对手模型。
      </p>
      {error && <div className="error-banner">{error}</div>}
      {results && (
        <div className="panel" style={{ marginBottom: 12 }}>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {Object.entries(results).map(([gameId, label]) => (
              <span key={gameId} className="badge">
                {GAME_LABELS[gameId] ?? gameId}: {label}
              </span>
            ))}
          </div>
        </div>
      )}
      <div className="panel">
        <table className="data">
          <thead>
            <tr>
              <th>游戏</th>
              <th>启用</th>
              <th>对局数</th>
              <th>决策数（人/AI）</th>
              <th>版本</th>
              <th>样本</th>
              <th>覆盖信息集</th>
              <th>发布时间</th>
              <th>门禁</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {status.map((s) => (
              <tr key={s.game_id}>
                <td>{GAME_LABELS[s.game_id] ?? s.game_id}</td>
                <td>
                  <button
                    className={`btn ${s.enabled ? '' : 'ghost'}`}
                    onClick={() => toggle(s.game_id, !s.enabled)}
                  >
                    {s.enabled ? '开' : '关'}
                  </button>
                </td>
                <td>{s.matches}</td>
                <td>
                  {s.decisions}（人 {s.human_decisions} / AI {s.ai_decisions}）
                </td>
                <ModelBlock status={s} />
                <td>
                  <button className="btn ghost" onClick={() => runApply(s.game_id)} disabled={busy}>
                    学习
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {status.length === 0 && <p className="muted">暂无学习数据 — 先在人机对战中完成几局。</p>}
      </div>
    </div>
  )
}