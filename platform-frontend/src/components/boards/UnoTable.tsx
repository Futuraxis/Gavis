import { useState } from 'react'
import type { UnoLegalAction, UnoSnapshot } from '../../types'

interface Props {
  snapshot: UnoSnapshot
  interactive: boolean
  onAction?: (action: Record<string, unknown>) => void
}

/** 卡牌 id 形如 ``r7a``（色 r/b/g/y + 符号 + 副本号）、``wild`` / ``wild4``。 */
const COLOR_NAMES: Record<string, string> = { r: '红', b: '蓝', g: '绿', y: '黄' }
const COLOR_HEX: Record<string, string> = { r: '#c0392b', b: '#2471a3', g: '#1e8449', y: '#b7950b' }
const SYMBOL_NAMES: Record<string, string> = { s: '禁止', r: '反转', d: '+2' }
const SYMBOL_GLYPHS: Record<string, string> = { s: '⊘', r: '⇄', d: '+2' }
const TOP_SYMBOL_NAMES: Record<string, string> = {
  skip: '禁止',
  reverse: '反转',
  draw2: '+2',
  wild: '万能',
  wild4: '+4',
}
/** 无 card 参数的动作 → 通用按钮（有 card 的动作经手牌选择后出现）。 */
const ACTION_NAMES: Record<string, string> = {
  draw: '摸牌',
  pass: '过',
  play_drawn: '打出刚摸的牌',
  jump_pass: '放弃抢牌',
  take_penalty: '吃下罚牌',
}

interface CardInfo {
  color: string | null
  glyph: string
  label: string
}

function cardInfo(id: string): CardInfo {
  if (id === 'wild') return { color: null, glyph: '🌟', label: '万能' }
  if (id === 'wild4') return { color: null, glyph: '+4', label: '+4 万能' }
  const c = id[0]
  if (c in COLOR_NAMES) {
    const sym = id[1]
    const glyph = SYMBOL_GLYPHS[sym] ?? sym
    return { color: c, glyph, label: `${COLOR_NAMES[c]}${SYMBOL_NAMES[sym] ?? sym}` }
  }
  return { color: null, glyph: id, label: id }
}

function UnoCard({
  id,
  selected,
  playable,
  onClick,
}: {
  id: string
  selected?: boolean
  playable?: boolean
  onClick?: () => void
}) {
  const info = cardInfo(id)
  const bg = info.color ? COLOR_HEX[info.color] : '#4a4a4a'
  return (
    <div
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      className="poker-card"
      style={{
        minWidth: 52,
        height: 74,
        background: bg,
        color: '#fff',
        cursor: onClick ? 'pointer' : 'default',
        outline: selected ? '3px solid #ffd166' : playable ? '2px dashed rgba(255,255,255,0.55)' : 'none',
        outlineOffset: 1,
        opacity: playable === false ? 0.55 : 1,
        flexDirection: 'column',
        justifyContent: 'center',
        gap: 2,
      }}
    >
      <span style={{ fontSize: 20, fontWeight: 700 }}>{info.glyph}</span>
      <span style={{ fontSize: 11 }}>{info.label}</span>
    </div>
  )
}

function CardBack() {
  return <div className="poker-card poker-card-back" style={{ minWidth: 30, height: 44 }} />
}

/** 由合法动作构造 payload（仅含动作实际携带的参数键，undefined 不下发）。 */
function payloadOf(l: UnoLegalAction): Record<string, unknown> {
  const p: Record<string, unknown> = { type: l.type }
  if (l.card !== undefined) p.card = l.card
  if (l.color !== undefined) p.color = l.color
  if (l.target !== undefined) p.target = l.target
  return p
}

function actionLabel(l: UnoLegalAction): string {
  const info = l.card ? cardInfo(l.card) : null
  if (l.type === 'play') return `打出 ${info?.label ?? l.card}`
  if (l.type === 'play_wild') return `${info?.label ?? '万能'} → ${COLOR_NAMES[l.color ?? ''] ?? l.color ?? ''}`
  if (l.type === 'play7') return `出 7 与 ${l.target} 换手`
  if (l.type === 'play_drawn') return `打出刚摸的 ${info?.label ?? l.card}`
  if (l.type === 'jump_play') return `抢出 ${info?.label ?? l.card}`
  if (l.type === 'stack2') return `叠加 ${info?.label ?? l.card}（+2）`
  if (l.type === 'stack4') return `叠加 ${info?.label ?? l.card}（+4）`
  return ACTION_NAMES[l.type] ?? l.type
}

/** UNO 桌面 — 三 AI 座位 / 台面顶牌 / 手牌（可点选） / 动作条。 */
export default function UnoTable({ snapshot, interactive, onAction }: Props) {
  const { player_pid, ai_pid, over } = snapshot
  const [selected, setSelected] = useState<string | null>(null)
  const myTurn = snapshot.turn === player_pid
  const canAct = interactive && myTurn && !over

  const seats = Object.keys(snapshot.hand_counts)
  const others = seats.filter((s) => s !== player_pid)
  const legal = snapshot.legal
  // 可出的手牌（出现在任一合法动作的 card 参数里）
  const playableCards = new Set(legal.filter((l) => l.card !== undefined).map((l) => l.card as string))
  const selectedActions = selected !== null ? legal.filter((l) => l.card === selected) : []
  const freeActions = legal.filter((l) => l.card === undefined)

  const topColorName = snapshot.top_color ? (COLOR_NAMES[snapshot.top_color] ?? snapshot.top_color) : ''
  const topSymbolLabel =
    snapshot.top_symbol !== null ? (TOP_SYMBOL_NAMES[snapshot.top_symbol] ?? snapshot.top_symbol) : ''

  const seatLabel = (pid: string) => (pid === ai_pid ? `AI · ${pid}` : `AI ${pid}`)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* AI 座位行 */}
      <div style={{ display: 'flex', gap: 10, justifyContent: 'center', flexWrap: 'wrap' }}>
        {others.map((pid) => (
          <div
            key={pid}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 4,
              padding: '8px 12px',
              borderRadius: 10,
              background: snapshot.turn === pid && !over ? 'rgba(46, 204, 113, 0.12)' : 'rgba(0,0,0,0.04)',
              minWidth: 110,
            }}
          >
            <div className="zone-label">
              {seatLabel(pid)}
              {snapshot.turn === pid && !over && <span className="turn-tag">行动中</span>}
              {pid === snapshot.penalty_target && snapshot.pending_draw > 0 && (
                <span className="folded-tag">吃罚牌中</span>
              )}
            </div>
            <div style={{ display: 'flex', gap: 2 }}>
              {Array.from({ length: Math.min(snapshot.hand_counts[pid] ?? 0, 8) }, (_, i) => (
                <CardBack key={i} />
              ))}
            </div>
            <div style={{ fontSize: 12, opacity: 0.75 }}>{snapshot.hand_counts[pid] ?? 0} 张</div>
            {over && snapshot.ai_hand.length > 0 && pid === ai_pid && (
              <div style={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                {snapshot.ai_hand.map((c, i) => (
                  <UnoCard key={i} id={c} />
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* 台面中央：牌堆 / 顶牌 / 方向 */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 22,
          padding: '12px 16px',
          borderRadius: 12,
          background: 'rgba(0,0,0,0.05)',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ textAlign: 'center', fontSize: 13 }}>
          <div style={{ fontSize: 22 }}>🂠</div>
          牌堆 {snapshot.deck_count}
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 4 }}>台面顶牌</div>
          {snapshot.top_color || snapshot.top_symbol ? (
            <div
              className="poker-card"
              style={{
                minWidth: 58,
                height: 78,
                background: snapshot.top_color ? COLOR_HEX[snapshot.top_color] : '#4a4a4a',
                color: '#fff',
                flexDirection: 'column',
                justifyContent: 'center',
                gap: 2,
              }}
            >
              <span style={{ fontSize: 20, fontWeight: 700 }}>{topSymbolLabel}</span>
              <span style={{ fontSize: 11 }}>{topColorName || '万能'}</span>
            </div>
          ) : (
            <div style={{ opacity: 0.6 }}>—</div>
          )}
        </div>
        <div style={{ fontSize: 13, textAlign: 'center' }}>
          <div style={{ fontSize: 22 }}>{snapshot.direction === 1 ? '↻' : '↺'}</div>
          {snapshot.direction === 1 ? '顺时针' : '逆时针'}
        </div>
        {snapshot.pending_draw > 0 && (
          <div style={{ color: '#c0392b', fontSize: 13 }}>
            ⚠️ 罚牌累计 +{snapshot.pending_draw}
            {snapshot.penalty_target ? `（${snapshot.penalty_target} 吃）` : ''}
          </div>
        )}
        {over && (
          <div className="hand-names" style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <span>{snapshot.winner === player_pid ? '🎉 你赢了！' : `胜者：${snapshot.winner ?? '—'}`}</span>
            {snapshot.payoff !== null && <span>收益 {snapshot.payoff}</span>}
          </div>
        )}
      </div>

      {/* 我的手牌 */}
      <div>
        <div className="zone-label" style={{ marginBottom: 6 }}>
          你 · {player_pid}
          {myTurn && !over && <span className="turn-tag">你的回合</span>}
          <span style={{ marginLeft: 10, opacity: 0.7 }}>剩 {snapshot.my_hand.length} 张</span>
        </div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {snapshot.my_hand.map((c, i) => (
            <UnoCard
              key={`${c}-${i}`}
              id={c}
              selected={selected === c}
              playable={canAct && playableCards.has(c)}
              onClick={
                canAct && playableCards.has(c) ? () => setSelected(selected === c ? null : c) : undefined
              }
            />
          ))}
          {snapshot.my_hand.length === 0 && !over && <span style={{ opacity: 0.6 }}>（无手牌）</span>}
        </div>
      </div>

      {/* 动作条 */}
      {!over && canAct && (
        <div className="action-panel" style={{ flexWrap: 'wrap', gap: 8 }}>
          {selectedActions.map((l, i) => (
            <button
              key={`${l.type}-${l.card ?? ''}-${l.color ?? ''}-${l.target ?? ''}-${i}`}
              className="btn btn-primary"
              onClick={() => {
                onAction?.(payloadOf(l))
                setSelected(null)
              }}
            >
              {actionLabel(l)}
            </button>
          ))}
          {freeActions.map((l, i) => (
            <button
              key={`${l.type}-${i}`}
              className="btn"
              onClick={() => {
                onAction?.(payloadOf(l))
                setSelected(null)
              }}
            >
              {actionLabel(l)}
            </button>
          ))}
          {selectedActions.length === 0 && freeActions.length === 0 && (
            <span className="hint" style={{ opacity: 0.7, fontSize: 13 }}>
              点击上方高亮手牌选择要出的牌
            </span>
          )}
        </div>
      )}
      {!canAct && !over && (
        <div style={{ textAlign: 'center', opacity: 0.65, fontSize: 13 }}>等待 AI 行动…</div>
      )}
    </div>
  )
}
