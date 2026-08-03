import { useState } from 'react'
import type { MahjongMeld, MahjongSnapshot } from '../../types'

interface Props {
  snapshot: MahjongSnapshot
  interactive: boolean
  onAction?: (action: Record<string, unknown>) => void
}

const SUIT_NAMES: Record<string, string> = { m: '万', p: '筒', s: '条', z: '字' }
const HONOR_NAMES: Record<string, string> = {
  z1: '东', z2: '南', z3: '西', z4: '北', z5: '中', z6: '发', z7: '白',
}
const SEAT_LABELS: Record<string, string> = {
  p0: '庄家', p1: '下家', p2: '对家', p3: '上家',
}
const MELD_NAMES: Record<string, string> = {
  chi: '吃', peng: '碰', gang: '杠', concealed_gang: '暗杠', added_gang: '加杠',
}

function tileLabel(tile: string): string {
  if (!tile || tile.length < 2) return tile
  const [suit, rank] = [tile[0], tile.slice(1)]
  if (suit === 'z') return HONOR_NAMES[tile] ?? tile
  return `${rank}${SUIT_NAMES[suit] ?? suit}`
}

/** 麻将牌 — 万红 / 筒蓝 / 条绿 / 字黑。 */
function TileView({ tile, selected, onClick }: { tile: string; selected?: boolean; onClick?: () => void }) {
  const suit = tile[0]
  const cls = `mahjong-tile ${suit === 'm' ? 'suit-m' : suit === 'p' ? 'suit-p' : suit === 's' ? 'suit-s' : 'suit-z'}${selected ? ' selected' : ''}`
  return (
    <div className={cls} onClick={onClick} role="button">
      <span className="tile-rank">{tile.slice(1)}</span>
      <span className="tile-suit">{SUIT_NAMES[suit] ?? suit}</span>
    </div>
  )
}

function MeldView({ meld }: { meld: MahjongMeld }) {
  return (
    <span className="mahjong-meld">
      <span className="meld-tag">{MELD_NAMES[meld.type] ?? meld.type}</span>
      {meld.tiles.map((t, i) => (
        <TileView key={i} tile={t} />
      ))}
    </span>
  )
}

function DiscardRiver({ tiles, compact }: { tiles: string[]; compact?: boolean }) {
  return (
    <div className={`discard-river${compact ? ' compact' : ''}`}>
      {tiles.map((t, i) => (
        <TileView key={i} tile={t} />
      ))}
    </div>
  )
}

interface SeatProps {
  pid: string
  label: string
  snapshot: MahjongSnapshot
  interactive: boolean
  onAction?: (action: Record<string, unknown>) => void
  isHuman: boolean
  isTurn: boolean
}

/** 一家座位的展示：手牌（自己可见）/ 副露 / 河。 */
function Seat({ pid, label, snapshot, interactive, onAction, isHuman, isTurn }: SeatProps) {
  const { my_hand, ai_hand, melds, discards, over, ai_pid } = snapshot
  const hand = isHuman ? my_hand : pid === ai_pid && over ? ai_hand : []
  const hiddenCount = isHuman ? 0 : snapshot.hand_counts[pid] ?? 0
  const [selected, setSelected] = useState<string | null>(null)

  const canAct = interactive && isHuman && isTurn

  function discard(tile: string) {
    onAction?.({ type: 'discard', tile })
    setSelected(null)
  }

  return (
    <div className={`mahjong-seat${isTurn && !snapshot.over ? ' active' : ''}${snapshot.done.includes(pid) ? ' done' : ''}`}>
      <div className="zone-label">
        {label}
        {isTurn && !snapshot.over && <span className="turn-tag">行动中</span>}
        {snapshot.done.includes(pid) && <span className="folded-tag">已胡</span>}
      </div>

      <div className="mahjong-melds">{melds[pid]?.map((m, i) => <MeldView key={i} meld={m} />)}</div>

      {isHuman ? (
        <div className="mahjong-hand">
          {hand.map((t, i) => (
            <TileView
              key={i}
              tile={t}
              selected={selected === t}
              onClick={canAct ? () => setSelected(selected === t ? null : t) : undefined}
            />
          ))}
        </div>
      ) : (
        <div className="mahjong-hand hidden-hand">
          {hiddenCount > 0
            ? Array.from({ length: Math.min(hiddenCount, 13) }, (_, i) => (
                <div key={i} className="mahjong-tile tile-back" />
              ))
            : null}
        </div>
      )}

      <DiscardRiver tiles={discards[pid] ?? []} compact={!isHuman} />

      {canAct && selected && (
        <button className="btn btn-primary discard-btn" onClick={() => discard(selected)}>
          打出 {tileLabel(selected)}
        </button>
      )}
    </div>
  )
}

/** 麻将桌面 — 二人（上下）与四人（环绕）布局 + claim 操作条。 */
export default function MahjongTable({ snapshot, interactive, onAction }: Props) {
  const { player_pid, ai_pid, over } = snapshot
  const seats = Object.keys(snapshot.discards)
  const isFour = seats.length === 4
  const myTurn = snapshot.turn === player_pid
  const canAct = interactive && myTurn
  const legal = snapshot.legal

  const legalByType = (t: string) => legal.filter((l) => l.type === t)
  const claimWin = legalByType('claim_win')
  const peng = legalByType('claim_peng')
  const gang = legalByType('claim_gang')
  const chis = legalByType('claim_chi')
  const winSelf = legalByType('win_self')
  const gangConcealed = legalByType('gang_concealed')
  const gangAdded = legalByType('gang_added')

  // 四人座次：自己在下，对家在上，左右按剩余座位
  const others = seats.filter((s) => s !== player_pid)
  const top = isFour ? others[1] : ai_pid
  const left = isFour ? others[2] : null
  const right = isFour ? others[0] : null

  const seatLabel = (pid: string) =>
    pid === player_pid ? '你' : pid === ai_pid ? 'AI' : `AI · ${SEAT_LABELS[pid] ?? pid}`

  return (
    <div className="mahjong-table-wrap">
      <div className={`mahjong-table${isFour ? ' four' : ''}`}>
        {left && (
          <div className="side-seat left">
            <Seat pid={left} label={seatLabel(left)} snapshot={snapshot} interactive={false} isHuman={false} isTurn={snapshot.turn === left} />
          </div>
        )}
        <div className="center-col">
          <Seat
            pid={top}
            label={seatLabel(top)}
            snapshot={snapshot}
            interactive={false}
            isHuman={false}
            isTurn={snapshot.turn === top}
          />
          <div className="mahjong-center">
            <div className="wall-info">🀫 牌墙 {snapshot.wall_remaining} 张</div>
            <div className="last-discard">
              {snapshot.last_discard ? (
                <>
                  刚打出 <TileView tile={snapshot.last_discard} />
                </>
              ) : (
                '等待出牌'
              )}
            </div>
            {over && (
              <div className="hand-names">
                <span>番/分: {snapshot.payoffs.map((p, i) => `${seats[i]}: ${p}`).join(' · ')}</span>
                {snapshot.winner && <span>胜者 {snapshot.winner}</span>}
              </div>
            )}
          </div>
          <Seat
            pid={player_pid}
            label="你"
            snapshot={snapshot}
            interactive={interactive}
            onAction={onAction}
            isHuman
            isTurn={myTurn}
          />
        </div>
        {right && (
          <div className="side-seat right">
            <Seat pid={right} label={seatLabel(right)} snapshot={snapshot} interactive={false} isHuman={false} isTurn={snapshot.turn === right} />
          </div>
        )}
      </div>

      {!over && canAct && snapshot.phase === 'claim' && (
        <div className="action-panel mahjong-claim-bar">
          {claimWin.map((l) => (
            <button key="win" className="btn btn-primary" onClick={() => onAction?.({ type: 'claim_win', tile: l.tile })}>
              荣和 🎉
            </button>
          ))}
          {gang.map((l) => (
            <button key="gang" className="btn" onClick={() => onAction?.({ type: 'claim_gang', tile: l.tile })}>
              杠 {tileLabel(l.tile ?? '')}
            </button>
          ))}
          {peng.map((l) => (
            <button key="peng" className="btn" onClick={() => onAction?.({ type: 'claim_peng', tile: l.tile })}>
              碰 {tileLabel(l.tile ?? '')}
            </button>
          ))}
          {chis.map((l, i) => (
            <button key={i} className="btn" onClick={() => onAction?.({ type: 'claim_chi', tiles: l.tiles })}>
              吃 {l.tiles?.map(tileLabel).join('')}
            </button>
          ))}
          <button className="btn" onClick={() => onAction?.({ type: 'claim_pass' })}>
            过
          </button>
        </div>
      )}

      {!over && canAct && snapshot.phase === 'action' && (
        <div className="action-panel">
          {winSelf.map(() => (
            <button key="win" className="btn btn-primary" onClick={() => onAction?.({ type: 'win_self' })}>
              自摸 🎉
            </button>
          ))}
          {gangConcealed.map((l) => (
            <button key="gc" className="btn" onClick={() => onAction?.({ type: 'gang_concealed', tile: l.tile })}>
              暗杠 {tileLabel(l.tile ?? '')}
            </button>
          ))}
          {gangAdded.map((l) => (
            <button key="ga" className="btn" onClick={() => onAction?.({ type: 'gang_added', tile: l.tile })}>
              加杠 {tileLabel(l.tile ?? '')}
            </button>
          ))}
          <span className="hint">点击手牌后「打出」</span>
        </div>
      )}
    </div>
  )
}
