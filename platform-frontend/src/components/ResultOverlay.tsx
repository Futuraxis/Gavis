import type { GameInfo, PokerSnapshot, Snapshot } from '../types'

interface Props {
  snapshot: Snapshot
  game: GameInfo
  onReplay: () => void
  onRestart: () => void
}

export default function ResultOverlay({ snapshot, game, onReplay, onRestart }: Props) {
  const { winner, player_pid } = snapshot
  const won = winner === player_pid
  const title = winner == null ? '🤝 平局' : won ? '🎉 你赢了！' : '😢 AI 获胜'
  const cls = winner == null ? 'draw' : won ? 'win' : 'lose'

  return (
    <div className="result-overlay panel">
      <div className={`result-title ${cls}`}>{title}</div>
      {game.kind === 'poker' && (
        <div style={{ color: 'var(--muted)' }}>
          <div>
            你的牌型: {(snapshot as PokerSnapshot).my_hand_name ?? '—'}　AI 牌型:{' '}
            {(snapshot as PokerSnapshot).ai_hand_name ?? '—'}
          </div>
          <div>
            结算: {(snapshot as PokerSnapshot).payoff != null && (snapshot as PokerSnapshot).payoff! >= 0
              ? `+${(snapshot as PokerSnapshot).payoff}`
              : (snapshot as PokerSnapshot).payoff}{' '}
            筹码
          </div>
        </div>
      )}
      <div style={{ display: 'flex', gap: 10 }}>
        <button className="btn btn-primary" onClick={onReplay}>
          查看回放
        </button>
        <button className="btn" onClick={onRestart}>
          再来一局
        </button>
      </div>
    </div>
  )
}
