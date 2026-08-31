import { useState } from 'react'
import type { SocialSnapshot } from '../../types'
import { humanWon as resolveHumanWon } from '../../matchResult'

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
  good: '好人',
  civilian: '平民',
  undercover: '卧底',
  blank: '白板',
}

const ACTION_LABELS: Record<string, string> = {
  speak: '发言',
  vote: '投票',
  self_destruct: '自爆',
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
      {snapshot.discourse.map((entry, i) => {
        if (entry.event === 'self_destruct') {
          const tgt = entry.target === mine ? '你' : entry.target
          return (
            <div key={i} className={`social-message ${entry.speaker === mine ? 'own' : ''}`}>
              <span className="social-speaker">{entry.speaker === mine ? '你' : entry.speaker}</span>
              <span className="social-text">自爆 {tgt}（猜：{entry.guess || '？'}）</span>
            </div>
          )
        }
        return (
          <div key={i} className={`social-message ${entry.speaker === mine ? 'own' : ''}`}>
            <span className="social-speaker">{entry.speaker === mine ? '你' : entry.speaker}</span>
            <span className="social-text">{entry.text || '（未发言）'}</span>
          </div>
        )
      })}
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

/** 自爆行：选目标 + 输入要猜的词（卧底猜对平民词→胜；猜错/平民→出局）。 */
function SelfDestructRow({
  targets,
  interactive,
  onSelfDestruct,
}: {
  targets: string[]
  interactive: boolean
  onSelfDestruct: (target: string, guess: string) => void
}) {
  const [target, setTarget] = useState(targets[0] ?? '')
  const [guess, setGuess] = useState('')
  return (
    <div className="social-target-row">
      <span className="social-action-label">{ACTION_LABELS.self_destruct}</span>
      <select
        value={target}
        disabled={!interactive || targets.length === 0}
        onChange={(e) => setTarget(e.target.value)}
      >
        {targets.map((t) => (
          <option key={t} value={t}>
            {t === targets.find(() => true) ? t : t}
          </option>
        ))}
      </select>
      <input
        className="social-speak-input"
        style={{ height: 'auto' }}
        placeholder="猜他的词…"
        value={guess}
        maxLength={50}
        disabled={!interactive}
        onChange={(e) => setGuess(e.target.value)}
      />
      <button
        className="btn btn-primary"
        disabled={!interactive || !guess.trim()}
        onClick={() => {
          onSelfDestruct(target, guess)
          setGuess('')
        }}
      >
        自爆
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
  const votes = snapshot.votes ?? []
  const myTurn = interactive && snapshot.turn === snapshot.player_pid
  // 终局胜负（阵营制）：winner 是阵营名，用公开身份表 final_roles 判玩家所属
  // 阵营（统一走 matchResult.humanWon——狼人杀 winner=good 时非狼身份全胜，
  // 不能只比身份名等于胜者；实测对局 e7deb84b 卧底获胜曾被显示成“AI 获胜”）。
  const finalRoles = snapshot.final_roles ?? []
  const myFinalRole = finalRoles.find((r) => r.pid === snapshot.player_pid)?.role ?? null
  const humanWon = resolveHumanWon(snapshot.winner, snapshot.player_pid, finalRoles)

  return (
    <div className="social-table">
      <div className="social-header">
        <span className={`social-phase-badge ${snapshot.over ? 'over' : ''}`}>
          {snapshot.over ? '已结束' : PHASE_LABELS[snapshot.phase ?? ''] ?? snapshot.phase ?? '对局中'}
        </span>
        {snapshot.over ? (
          <span className="social-winner">
            {humanWon ? '🎉 你赢了！' : snapshot.winner != null && myFinalRole != null ? '😢 你输了' : ''}
            {snapshot.winner != null && <>　胜方：{ROLE_LABELS[snapshot.winner ?? ''] ?? snapshot.winner}</>}
            {snapshot.winners.length > 0 && <>（{snapshot.winners.join('、')}）</>}
          </span>
        ) : (
          <>
            {typeof snapshot.round === 'number' && (
              <span className="social-ai-mode">第 {snapshot.round} 轮</span>
            )}
            {snapshot.my_role ? (
              <span className="social-my-role">
                你的身份：{ROLE_LABELS[snapshot.my_role] ?? snapshot.my_role}
              </span>
            ) : (
              <span className="social-my-role">你的身份：未知（靠发言推断）</span>
            )}
            {snapshot.my_word && (
              <span className="social-my-role">你的词：{snapshot.my_word}</span>
            )}
            <span className="social-ai-mode">
              AI 模式：{snapshot.ai_mode === 'ollama' ? '本地大模型' : '随机策略'}
            </span>
          </>
        )}
      </div>

      {snapshot.over && snapshot.final_roles && snapshot.final_roles.length > 0 && (
        <div className="social-reveal">
          <div className="social-reveal-title">身份揭晓</div>
          <div className="social-reveal-grid">
            {snapshot.final_roles.map((r) => (
              <div
                key={r.pid}
                className={`social-reveal-item${r.pid === snapshot.player_pid ? ' mine' : ''}`}
              >
                <span className="social-reveal-pid">
                  {r.pid === snapshot.player_pid ? '你' : r.pid}
                </span>
                <span className="social-reveal-role">
                  {ROLE_LABELS[r.role ?? ''] ?? r.role ?? '？'}
                </span>
                {r.word && <span className="social-reveal-word">词：{r.word}</span>}
              </div>
            ))}
          </div>
        </div>
      )}

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

      {/* 投票记录 + 出局信息（v5.2 公开 vote_log/deaths_arr/eliminated 投影）：
          让玩家能看到「谁投了谁」，而不是只见发言不见投票。 */}
      {votes.length > 0 && (
        <div className="social-votes">
          <div className="social-votes-title">投票记录</div>
          {votes.map((v, i) => (
            <div key={i} className="social-vote-row">
              <span className="social-voter">
                {v.voter === snapshot.player_pid ? '你' : v.voter}
              </span>
              <span className="social-vote-arrow">→</span>
              <span className="social-vote-target">
                {v.target === snapshot.player_pid ? '你' : v.target}
              </span>
              {typeof v.round === 'number' && (
                <span className="social-vote-round">第 {v.round} 轮</span>
              )}
            </div>
          ))}
        </div>
      )}
      {snapshot.eliminated && (
        <div className="social-death-note">
          {snapshot.eliminated === snapshot.player_pid ? '你' : snapshot.eliminated} 被投票出局
          {snapshot.deaths && snapshot.deaths.length > 0 && (
            <>（已出局：{snapshot.deaths.map((d) => (d === snapshot.player_pid ? '你' : d)).join('、')}）</>
          )}
        </div>
      )}

      {!snapshot.over && myTurn && snapshot.legal.length > 0 && (
        <div className="social-actions">
          {canSpeak && <SpeakBox interactive={interactive} onSpeak={(text) => onMove?.({ type: 'speak', text })} />}
          {Object.entries(grouped).map(([type, targets]) =>
            type === 'self_destruct' ? (
              <SelfDestructRow
                key={`${type}:${targets.join(',')}`}
                targets={targets}
                interactive={interactive}
                onSelfDestruct={(target, guess) => onMove?.({ type: 'self_destruct', target, guess })}
              />
            ) : (
              <TargetRow
                key={`${type}:${targets.join(',')}`}
                type={type}
                targets={targets}
                interactive={interactive}
                onTarget={(t, target) => onMove?.({ type: t, target })}
              />
            )
          )}
        </div>
      )}
      {!snapshot.over && snapshot.turn !== snapshot.player_pid && (
        <div className="social-waiting">
          {/* 流式进度帧里 turn = 当前说话/行动的 AI 座位 —— 逐条发言时这里会
              跟随每一帧实时切换，让「谁在说话」和发言本身一样动态可见。 */}
          {snapshot.turn != null ? (
            <>
              <span className="social-waiting-name">
                {snapshot.turn === snapshot.player_pid ? '你' : snapshot.turn}
              </span>
              <span className="social-typing-dots">
                <i />
                <i />
                <i />
              </span>
              正在发言…
            </>
          ) : (
            '其他玩家行动中…'
          )}
        </div>
      )}
    </div>
  )
}