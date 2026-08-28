// ReviewCard — 复盘卡（内嵌于聊天消息；数据来自 /api/review/<match_id>）。

import type { ReviewReport } from '../../types'

interface Props {
  report: ReviewReport
  matchId: string
}

const KIND_LABEL: Record<string, string> = {
  turning_point: '转折点',
  winning_move: '胜着',
  blunder: '昏招',
}

export default function ReviewCard({ report, matchId }: Props) {
  return (
    <div className="chat-card">
      <div className="chat-card-title">复盘 · {matchId}</div>
      <div className="chat-review-summary">{report.summary}</div>
      {report.key_nodes.length > 0 && (
        <ul className="chat-review-nodes">
          {report.key_nodes.map((n, i) => (
            <li key={i} className="chat-review-node">
              <span className="chat-review-step">第 {n.step} 手</span>
              <span className="chat-review-kind">{KIND_LABEL[n.kind] ?? n.kind}</span>
              <span>{n.why}</span>
            </li>
          ))}
        </ul>
      )}
      <div className="chat-card-hint">{report.improvement}</div>
    </div>
  )
}