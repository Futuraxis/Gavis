// CreateCard — 创建游戏面板（内嵌于聊天消息；自然语言规则 / 模板变体）。
// 直接调 /api/custom/games，成功回调 onCreated 让对话流接续。

import { useState } from 'react'
import { createCustomGame } from '../../api/client'
import type { GameInfo } from '../../types'

interface Props {
  onCreated: (game: GameInfo) => void
}

export default function CreateCard({ onCreated }: Props) {
  const [mode, setMode] = useState<'from_scratch' | 'variant'>('from_scratch')
  const [gameName, setGameName] = useState('')
  const [ruleText, setRuleText] = useState('')
  const [baseGameId, setBaseGameId] = useState('stochastic_gomoku')
  const [changeText, setChangeText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit() {
    setBusy(true)
    setError(null)
    try {
      const result = await createCustomGame(
        mode === 'from_scratch'
          ? { mode: 'from_scratch', game_name: gameName || undefined, rule_text: ruleText, use_llm: true }
          : { mode: 'variant', base_game_id: baseGameId, change_text: changeText, game_name: gameName || undefined, use_llm: true },
      )
      onCreated(result.game)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="chat-card">
      <div className="chat-card-title">创建新游戏</div>
      <div className="chat-form">
        <div className="chat-form-row">
          <label>方式：</label>
          <select value={mode} onChange={(e) => setMode(e.target.value as 'from_scratch' | 'variant')}>
            <option value="from_scratch">自然语言写规则</option>
            <option value="variant">模板变体</option>
          </select>
        </div>
        <div className="chat-form-row">
          <label>名字：</label>
          <input value={gameName} onChange={(e) => setGameName(e.target.value)} placeholder="如：三子棋传奇" />
        </div>
        {mode === 'from_scratch' ? (
          <div className="chat-form-row">
            <label>规则：</label>
            <textarea
              value={ruleText}
              onChange={(e) => setRuleText(e.target.value)}
              placeholder="用中文描述规则，如：3x3 棋盘，双方轮流落子，连成三子获胜"
              rows={3}
            />
          </div>
        ) : (
          <>
            <div className="chat-form-row">
              <label>基础游戏：</label>
              <input value={baseGameId} onChange={(e) => setBaseGameId(e.target.value)} placeholder="stochastic_gomoku" />
            </div>
            <div className="chat-form-row">
              <label>变化：</label>
              <input value={changeText} onChange={(e) => setChangeText(e.target.value)} placeholder="如：15x15 连五" />
            </div>
          </>
        )}
        {error && <div className="chat-form-error">{error}</div>}
        <button className="btn btn-primary" disabled={busy} onClick={submit}>
          {busy ? '翻译规则中…' : '创建'}
        </button>
      </div>
    </div>
  )
}