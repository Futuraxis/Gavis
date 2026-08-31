import type { GameInfo, MahjongSnapshot, PokerSnapshot, Snapshot, SocialSnapshot } from '../types'
import { factionLabel, humanWon } from '../matchResult'

interface Props {
  snapshot: Snapshot
  game: GameInfo
  onReplay: () => void
  onRestart: () => void
}

export default function ResultOverlay({ snapshot, game, onReplay, onRestart }: Props) {
  const { winner, player_pid } = snapshot

  // 血战等多胡局没有单一 winner：按 winners 列表展示（旧逻辑会误报「平局」）。
  const mahjong = game.kind === 'mahjong' ? (snapshot as MahjongSnapshot) : null
  const winners = mahjong?.winners ?? []
  // 社交游戏（谁是卧底/狼人杀）的 winner 是**阵营名**而非 pid：终局身份表
  // final_roles 已公开，统一走 matchResult.humanWon 阵营比对（否则卧底获胜会
  // 误报「AI 获胜」——实测对局 e7deb84b）。
  const finalRoles = (snapshot as SocialSnapshot).final_roles ?? []
  let title: string
  let cls: string
  if (winner == null && winners.length > 0) {
    const won = winners.includes(player_pid)
    title = won ? '🎉 你胡了！' : `🏆 本局胡家：${winners.join('、')}`
    cls = won ? 'win' : 'lose'
  } else {
    const won = humanWon(winner, player_pid, finalRoles)
    title =
      winner == null ? '🤝 平局' : won ? '🎉 你赢了！' : `😢 ${factionLabel(winner) ?? 'AI'} 获胜`
    cls = winner == null ? 'draw' : won ? 'win' : 'lose'
  }

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
      {mahjong && mahjong.payoffs.length > 0 && (
        <div style={{ color: 'var(--muted)' }}>
          <div>
            结算:{' '}
            {mahjong.payoffs
              .map((p, i) => `p${i}${i === 0 ? '(庄)' : ''} ${p >= 0 ? '+' : ''}${p}`)
              .join('　')}
            （血战局按胡牌顺序累计）
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
