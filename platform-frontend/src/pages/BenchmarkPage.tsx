import { useEffect, useState } from 'react'
import { apiGet, apiPost } from '../api/client'
import { usePolling } from '../hooks/usePolling'
import type { BenchmarkJob, GameInfo } from '../types'

const SOLVER_LABELS: Record<string, string> = {
  mcts: 'MCTS',
  cfr: 'CFR',
  hybrid: 'Hybrid',
  random: '随机',
}

const GAME_LABELS: Record<string, string> = {
  moon_chess: '月亮棋',
  stochastic_gomoku: '随机五子棋',
  texas_holdem: '德州扑克',
}

const DEFAULT_ITERATIONS: Record<string, number> = {
  moon_chess: 30,
  stochastic_gomoku: 20,
  texas_holdem: 10,
}

const STATUS_LABELS: Record<string, string> = {
  pending: '排队中',
  running: '运行中',
  done: '已完成',
  error: '失败',
}

function JobCard({ job }: { job: BenchmarkJob }) {
  const progress = Math.round((job.progress / job.iterations) * 100)
  return (
    <div className="panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <strong>
          {GAME_LABELS[job.game_id] ?? job.game_id} · {SOLVER_LABELS[job.solver_a] ?? job.solver_a} vs{' '}
          {SOLVER_LABELS[job.solver_b] ?? job.solver_b}
        </strong>
        <span className={`badge ${job.status === 'done' ? 'win' : job.status === 'error' ? 'lose' : 'accent'}`}>
          {STATUS_LABELS[job.status] ?? job.status}
        </span>
        <span className="badge">{job.iterations} 迭代</span>
        {job.started_at && <span className="badge">{new Date(job.started_at).toLocaleTimeString('zh-CN')}</span>}
      </div>
      {job.status === 'pending' || job.status === 'running' ? (
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
      ) : null}
      {job.error && <div className="error-banner" style={{ marginTop: 10 }}>{job.error}</div>}
      {job.status === 'done' && job.results && (
        <table className="data" style={{ marginTop: 12 }}>
          <tbody>
            <tr>
              <th>{SOLVER_LABELS[job.solver_a] ?? job.solver_a} 胜</th>
              <td>
                {job.results.a_wins} 局 ({Math.round(job.results.a_win_rate * 100)}%)
              </td>
              <th>{SOLVER_LABELS[job.solver_b] ?? job.solver_b} 胜</th>
              <td>
                {job.results.b_wins} 局 ({Math.round(job.results.b_win_rate * 100)}%)
              </td>
            </tr>
            <tr>
              <th>平局</th>
              <td>
                {job.results.draws} 局 ({Math.round(job.results.draw_rate * 100)}%)
              </td>
              <th>平均步数 / 每步耗时</th>
              <td>
                {job.results.avg_moves} 步 / {job.results.avg_seconds_per_move}s
              </td>
            </tr>
            <tr>
              <th>异常局数</th>
              <td colSpan={3}>{job.results.errors}</td>
            </tr>
          </tbody>
        </table>
      )}
    </div>
  )
}

export default function BenchmarkPage() {
  const [games, setGames] = useState<GameInfo[]>([])
  const [gameId, setGameId] = useState('moon_chess')
  const [solverA, setSolverA] = useState('mcts')
  const [solverB, setSolverB] = useState('random')
  const [iterations, setIterations] = useState(DEFAULT_ITERATIONS.moon_chess)
  const [jobs, setJobs] = useState<BenchmarkJob[]>([])
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const game = games.find((g) => g.game_id === gameId)
  const options = game?.solver_options ?? []

  useEffect(() => {
    apiGet<{ games: GameInfo[] }>('/games')
      .then((data) => setGames(data.games))
      .catch((err: Error) => setError(err.message))
    refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function refresh() {
    apiGet<{ jobs: BenchmarkJob[] }>('/benchmark')
      .then((data) => setJobs(data.jobs))
      .catch((err: Error) => setError(err.message))
  }

  const hasActiveJob = jobs.some((j) => j.status === 'pending' || j.status === 'running')
  usePolling(refresh, 1500, hasActiveJob)

  function selectGame(id: string) {
    setGameId(id)
    const fresh = games.find((g) => g.game_id === id)
    const opts = fresh?.solver_options ?? []
    setSolverA(opts[0] ?? 'mcts')
    setSolverB(opts[opts.length - 1] ?? 'random')
    setIterations(DEFAULT_ITERATIONS[id] ?? 30)
  }

  async function run() {
    setSubmitting(true)
    setError(null)
    try {
      const data = await apiPost<{ job: BenchmarkJob }>('/benchmark/start', {
        game_id: gameId,
        solver_a: solverA,
        solver_b: solverB,
        iterations: iterations,
      })
      setJobs((prev) => [data.job, ...prev])
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div>
      <h1 className="page-title">求解器评测</h1>
      <p className="page-sub">AI vs AI 对局对比 — 双方交替先手，消除先手优势</p>
      {error && <div className="error-banner">{error}</div>}

      <div className="panel" style={{ marginBottom: 22 }}>
        <div className="form-row">
          <label>游戏:</label>
          <select value={gameId} onChange={(e) => selectGame(e.target.value)}>
            {games.map((g) => (
              <option key={g.game_id} value={g.game_id}>
                {g.display_name}
              </option>
            ))}
          </select>
        </div>
        <div className="form-row">
          <label>求解器 A:</label>
          <select value={solverA} onChange={(e) => setSolverA(e.target.value)}>
            {options.map((s) => (
              <option key={s} value={s}>
                {SOLVER_LABELS[s] ?? s}
              </option>
            ))}
          </select>
          <span style={{ color: 'var(--muted)' }}>VS</span>
          <label>求解器 B:</label>
          <select value={solverB} onChange={(e) => setSolverB(e.target.value)}>
            {options.map((s) => (
              <option key={s} value={s}>
                {SOLVER_LABELS[s] ?? s}
              </option>
            ))}
          </select>
        </div>
        <div className="form-row">
          <label>迭代次数:</label>
          <input
            type="number"
            min={1}
            max={200}
            value={iterations}
            onChange={(e) => setIterations(Math.max(1, Number(e.target.value)))}
            style={{ width: 90 }}
          />
          <button className="btn btn-primary" disabled={submitting || solverA === solverB} onClick={run}>
            {submitting ? '启动中…' : '开始评测'}
          </button>
          {solverA === solverB && <span style={{ color: 'var(--lose)', fontSize: 13 }}>两个求解器不能相同</span>}
        </div>
      </div>

      {jobs.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {jobs.map((job) => (
            <JobCard key={job.job_id} job={job} />
          ))}
        </div>
      )}
    </div>
  )
}
