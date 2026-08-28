// ProgressCard — 进度卡（评测中心 / 在线学习，内嵌于聊天消息）。
// 数据经聊天意图的 params 传入（benchmark.jobs / learning.learning）。

import type { BenchmarkJob, LearningItem } from '../useChatRuntime'

interface Props {
  mode: 'benchmark' | 'learning'
  jobs?: BenchmarkJob[]
  learning?: LearningItem[]
}

export default function ProgressCard({ mode, jobs, learning }: Props) {
  if (mode === 'benchmark') {
    const list = jobs ?? []
    if (!list.length) {
      return <div className="chat-card chat-card-muted">还没有评测任务。完整面板在 平台界面 → 求解器评测。</div>
    }
    return (
      <div className="chat-card">
        <div className="chat-card-title">评测中心</div>
        <ul className="chat-progress-list">
          {list.map((j) => (
            <li key={j.job_id} className="chat-progress-row">
              <span className="chat-progress-game">{j.game_id}</span>
              <span className="chat-progress-vs">
                {j.solver_a} vs {j.solver_b}
              </span>
              <span className={`chat-progress-status ${j.status}`}>{j.status}</span>
              <span className="chat-progress-meta">
                {j.progress}/{j.iterations}
              </span>
            </li>
          ))}
        </ul>
      </div>
    )
  }
  const list = learning ?? []
  if (!list.length) {
    return <div className="chat-card chat-card-muted">在线学习还没有数据。对局几局后这里会出现学习状态。</div>
  }
  return (
    <div className="chat-card">
      <div className="chat-card-title">在线学习</div>
      <ul className="chat-progress-list">
        {list.map((item) => (
          <li key={item.game_id} className="chat-progress-row">
            <span className="chat-progress-game">{item.game_id}</span>
            <span className="chat-progress-meta">{item.matches} 局 · {item.decisions} 决策</span>
            <span className={`chat-progress-status ${item.enabled ? 'enabled' : 'disabled'}`}>
              {item.enabled ? '开启' : '关闭'}
            </span>
            {item.model ? <span className="chat-progress-meta">v{item.model.version}</span> : null}
          </li>
        ))}
      </ul>
    </div>
  )
}