import { useState } from 'react'
import type { SocialSnapshot } from '../../types'

interface Props {
  snapshot: SocialSnapshot
  interactive: boolean
  onMove?: (action: unknown) => void
}

const PHASE_LABELS: Record<string, string> = {
  deal_0: '发牌',
  describe: '描述·发言',
  vote: '投票',
  resolve: '结算',
  night_wolf: '夜晚·狼人行动',
  night_witch: '夜晚·女巫行动',
  night_seer: '夜晚·预言家行动',
  night_hunter: '夜晚·猎人行动',
  night_end: '夜晚结算',
  day_speech: '白天·发言',
  day_vote: '白天·投票',
  vote_resolve: '放逐结算',
  vote_hunter: '猎人开枪',
  game_over: '游戏结束',
}

const ROLE_LABELS: Record<string, string> = {
  wolf: '狼人',
  villager: '村民',
  seer: '预言家',
  witch: '女巫',
  hunter: '猎人',
  guard: '守卫',
  civilian: '平民',
  undercover: '卧底',
  blank: '白板',
}

const ACTION_LABELS: Record<string, string> = {
  speak: '发言',
  vote: '投票',
  kill: '击杀',
  check: '查验',
  heal: '救援',
  poison: '下毒',
  shoot: '开枪',
  shoot_lynched: '开枪',
  guard: '守护',
  pass: '过',
}

function Discourse({ snapshot }: { snapshot: SocialSnapshot }) {
  const mine = snapshot.player_pid
  return (
    <div className="social-discourse">
      {snapshot.discourse.length === 0 && <div className="social-empty">暂无公开发言</div>}
      {snapshot.discourse.map((entry, i) => (
        <div key={i} className={`social-message ${entry.speaker === mine ? 'own' : ''}`}>
          <span className="social-speaker">{entry.speaker === mine ? '你' : entry.speaker}</span>
          <span className="social-text">{entry.text || '（未发言）'}</span>
        </div>
      ))}
    </div>
  )
}

function SpeakBox({
  interactive,
  onSpeak,
}: {
  interactive: boolean
  onSpeak: (text: string) => void
}) {
  const [text, setText] = useState('')
  return (
    <div className="social-speak-row">
      <textarea
        className="social-speak-input"
        placeholder="说点什么…"
        value={text}
        maxLength={200}
        disabled={!interactive}
        onChange={(e) => setText(e.target.value)}
      />
      <button
        className="btn btn-primary"
        disabled={!interactive || !text.trim()}
        onClick={() => {
          onSpeak(text)
          setText('')
        }}
      >
        {ACTION_LABELS.speak}（发送）
      </button>
    </div>
  )
}

function TargetRow({
  type,
  targets,
  interactive,
  onTarget,
}: {
  type: string
  targets: string[]
  interactive: boolean
  onTarget: (type: string, target: string) => void
}) {
  const [target, setTarget] = useState(targets[0] ?? '')
  return (
    <div className="social-target-row">
      <span className="social-action-label">{ACTION_LABELS[type] ?? type}</span>
      <select
        value={target}
        disabled={!interactive || targets.length === 0}
        onChange={(e) => setTarget(e.target.value)}
      >
        {targets.map((t) => (
          <option key={t} value={t}>
            {ACTION_LABELS[t] ?? t}
          </option>
        ))}
      </select>
      <button className="btn btn-primary" disabled={!interactive} onClick={() => onTarget(type, target)}>
        确认
      </button>
    </div>
  )
}

/** 社交聊天桌 — 阶段 / 身份 / 存活 / 公开发言 / 合法动作（发言 + 目标动作）。 */
export default function SocialChatTable({ snapshot, interactive, onMove }: Props) {
  const grouped: Record<string, string[]> = {}
  for (const action of snapshot.legal) {
    if (action.type === 'speak') continue
    if (!grouped[action.type]) grouped[action.type] = []
    grouped[action.type].push(action.target ?? '')
  }
  const canSpeak = snapshot.legal.some((a) => a.type === 'speak')
  const myTurn = interactive && snapshot.turn === snapshot.player_pid

  return (
    <div className="social-table">
      <div className="social-header">
        <span className={`social-phase-badge ${snapshot.over ? 'over' : ''}`}>
          {snapshot.over ? '已结束' : PHASE_LABELS[snapshot.phase ?? ''] ?? snapshot.phase ?? '对局中'}
        </span>
        {snapshot.over ? (
          <span className="social-winner">
            胜方：{ROLE_LABELS[snapshot.winner ?? ''] ?? snapshot.winner ?? '未知'}
            {snapshot.winners.length > 0 && <>（{snapshot.winners.join('、')}）</>}
          </span>
        ) : (
          <>
            <span className="social-my-role">
              你的身份：{ROLE_LABELS[snapshot.my_role ?? ''] ?? snapshot.my_role ?? '未知'}
            </span>
            {snapshot.my_word && (
              <span className="social-my-role">你的词：{snapshot.my_word}</span>
            )}
            <span className="social-ai-mode">
              AI 模式：{snapshot.ai_mode === 'ollama' ? '本地大模型' : '随机策略'}
            </span>
          </>
        )}
      </div>

      {!snapshot.over && (
        <div className="social-players">
          {snapshot.alive.map((pid) => (
            <span key={pid} className={`social-player-chip ${pid === snapshot.turn ? 'turn' : ''}`}>
              {pid}
              {pid === snapshot.player_pid ? '（你）' : ''}
            </span>
          ))}
          <span className="social-alive-count">存活 {snapshot.alive.length}</span>
        </div>
      )}

      <Discourse snapshot={snapshot} />

      {!snapshot.over && myTurn && snapshot.legal.length > 0 && (
        <div className="social-actions">
          {canSpeak && <SpeakBox interactive={interactive} onSpeak={(text) => onMove?.({ type: 'speak', text })} />}
          {Object.entries(grouped).map(([type, targets]) => (
            <TargetRow
              key={`${type}:${targets.join(',')}`}
              type={type}
              targets={targets}
              interactive={interactive}
              onTarget={(t, target) => onMove?.({ type: t, target })}
            />
          ))}
        </div>
      )}
      {!snapshot.over && snapshot.turn !== snapshot.player_pid && (
        <div className="social-waiting">其他玩家行动中…</div>
      )}
    </div>
  )
}