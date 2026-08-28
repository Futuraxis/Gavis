import { useEffect, useState } from 'react'
import type { PokerSnapshot } from '../../types'

interface Props {
  snapshot: PokerSnapshot
  interactive: boolean
  onAction?: (action: { choice: string; amount?: number | null }) => void
}

const SUIT_GLYPHS: Record<string, string> = { S: '♠', H: '♥', D: '♦', C: '♣' }
const SUIT_CHARS = ['s', 'h', 'd', 'c', 'S', 'H', 'D', 'C']
const RANK_NAMES: Record<string, string> = { T: '10' }
const SEAT_LABELS: Record<string, string> = { p_sb: '小盲位', p_bb: '大盲位' }

/** 解析规则引擎的牌串（花⾊在前的小写形式如 `hA`/`d9`，兼容花⾊在后的 `Ah` 形式）。 */
function parseCard(card: string): { suit: string; rank: string } | null {
  if (!card) return null
  if (SUIT_CHARS.includes(card[0])) return { suit: card[0].toUpperCase(), rank: card.slice(1) }
  const last = card[card.length - 1]
  if (SUIT_CHARS.includes(last)) return { suit: last.toUpperCase(), rank: card.slice(0, -1) }
  return null
}

function CardView({ card }: { card: string }) {
  const parsed = parseCard(card)
  if (parsed === null) {
    return (
      <div className="poker-card">
        <span style={{ fontSize: 13 }}>{card}</span>
      </div>
    )
  }
  const red = parsed.suit === 'H' || parsed.suit === 'D'
  return (
    <div className={`poker-card ${red ? 'red' : ''}`}>
      <span style={{ fontSize: 18 }}>{SUIT_GLYPHS[parsed.suit] ?? parsed.suit}</span>
      <span style={{ fontSize: 13 }}>{RANK_NAMES[parsed.rank] ?? parsed.rank}</span>
    </div>
  )
}

function ActionPanel({ snapshot, interactive, onAction }: Props) {
  const [amount, setAmount] = useState<number>(snapshot.raise_amounts[0] ?? 0)
  const canCall = snapshot.legal.some((a) => a.choice === 'call')
  const canRaise = snapshot.raise_amounts.length > 0
  const callLabel = snapshot.call_to === 0 ? 'Check 过牌' : 'Call 跟注'

  // 换街后合法加注额变化时同步选择器
  useEffect(() => {
    if (!snapshot.raise_amounts.includes(amount)) {
      setAmount(snapshot.raise_amounts[0] ?? 0)
    }
  }, [snapshot.raise_amounts, amount])

  return (
    <div className="action-panel">
      <button className="btn btn-danger" disabled={!interactive} onClick={() => onAction?.({ choice: 'fold' })}>
        Fold 弃牌
      </button>
      {canCall && (
        <button className="btn" disabled={!interactive} onClick={() => onAction?.({ choice: 'call' })}>
          {callLabel}
        </button>
      )}
      {canRaise && (
        <>
          <select value={amount} onChange={(e) => setAmount(Number(e.target.value))} disabled={!interactive}>
            {snapshot.raise_amounts.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
          <button
            className="btn btn-primary"
            disabled={!interactive}
            onClick={() => onAction?.({ choice: 'raise', amount })}
          >
            Raise 加注
          </button>
        </>
      )}
    </div>
  )
}

/** 双人德州扑克桌面 — 筹码区 / 底池 / 公共牌 / 手牌 / 操作面板。 */
export default function PokerTable({ snapshot, interactive, onAction }: Props) {
  const { ai_pid, player_pid } = snapshot
  return (
    <div className="poker-table-wrap">
      <div className="poker-table">
        <div className="player-zone">
          <div className="zone-label">
            AI · {SEAT_LABELS[ai_pid] ?? ai_pid}
          </div>
          <div className="zone-cards">
            {(snapshot.revealed ? snapshot.ai_hole : ['', '']).map((card, i) =>
              card ? <CardView key={i} card={card} /> : <div key={i} className="poker-card poker-card-back" />,
            )}
          </div>
          {snapshot.ai_folded && <span className="folded-tag">已弃牌</span>}
          <div className="stack-info">
            <div>筹码 {snapshot.ai_stack}</div>
            <div>已下注 {snapshot.ai_committed}</div>
          </div>
        </div>

        <div className="table-center">
          <div className="pot">🪙 底池 {snapshot.pot}</div>
          <div className="street">{snapshot.street_name}</div>
          <div className="community">{snapshot.community.map((c, i) => <CardView key={i} card={c} />)}</div>
          {snapshot.over && (
            <div className="hand-names">
              <span>
                你: {snapshot.my_hand_name ?? '—'}
              </span>
              <span>
                AI: {snapshot.ai_hand_name ?? '—'}
              </span>
            </div>
          )}
        </div>

        <div className="player-zone">
          <div className="zone-label">
            你 · {SEAT_LABELS[player_pid] ?? player_pid}
          </div>
          <div className="zone-cards">{snapshot.my_hole.map((c, i) => <CardView key={i} card={c} />)}</div>
          {snapshot.my_folded && <span className="folded-tag">已弃牌</span>}
          <div className="stack-info">
            <div>筹码 {snapshot.my_stack}</div>
            <div>已下注 {snapshot.my_committed}</div>
          </div>
        </div>
      </div>
      {!snapshot.over && <ActionPanel snapshot={snapshot} interactive={interactive} onAction={onAction} />}
    </div>
  )
}
