import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { createCustomGame, deleteCustomGame, listCustomGames } from '../api/client'
import type { CustomCreateResult, GameInfo } from '../types'

/** 变体翻译的 base 模板（layer1_translator TEMPLATE_FILES 对应规则 id）。 */
const BASE_TEMPLATES = [
  { id: 'moon_chess', label: '月亮棋' },
  { id: 'stochastic_gomoku', label: '随机五子棋' },
  { id: 'gomoku', label: '五子棋' },
  { id: 'texas_holdem', label: '德州扑克' },
  { id: 'mahjong', label: '麻将' },
  { id: 'werewolf', label: '狼人杀' },
]

const FAMILY_LABELS: Record<string, string> = { grid: '网格', poker: '扑克', mahjong: '麻将', social: '社交' }

type Mode = 'from_scratch' | 'variant'

export default function CreateGamePage() {
  const [mode, setMode] = useState<Mode>('from_scratch')
  const [baseGameId, setBaseGameId] = useState(BASE_TEMPLATES[0].id)
  const [ruleText, setRuleText] = useState('')
  const [changeText, setChangeText] = useState('')
  const [gameName, setGameName] = useState('')
  const [useLlm, setUseLlm] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<CustomCreateResult | null>(null)
  const navigate = useNavigate()

  // ── 我的自定义游戏（含变体）管理列表 ────────────────────────────
  const [customGames, setCustomGames] = useState<GameInfo[]>([])
  const [listError, setListError] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const loadCustomGames = useCallback(async () => {
    setListError(null)
    try {
      const data = await listCustomGames()
      setCustomGames(data.games)
    } catch (err) {
      setListError((err as Error).message)
    }
  }, [])

  useEffect(() => {
    void loadCustomGames()
  }, [loadCustomGames])

  async function removeCustom(game: GameInfo) {
    if (deletingId !== null) return
    if (!window.confirm(`确定删除「${game.display_name}」（id: ${game.game_id}）吗？\n删除后不可恢复。`)) return
    setDeletingId(game.game_id)
    setListError(null)
    try {
      await deleteCustomGame(game.game_id)
      setCustomGames((prev) => prev.filter((g) => g.game_id !== game.game_id))
    } catch (err) {
      setListError((err as Error).message)
    } finally {
      setDeletingId(null)
    }
  }

  const requiredText = mode === 'from_scratch' ? ruleText : changeText
  const canSubmit = requiredText.trim().length > 0 && !busy

  async function submit() {
    if (!canSubmit) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const res = await createCustomGame({
        mode,
        rule_text: mode === 'from_scratch' ? ruleText : undefined,
        base_game_id: mode === 'variant' ? baseGameId : undefined,
        change_text: mode === 'variant' ? changeText : undefined,
        game_name: gameName.trim() || undefined,
        source_lang: 'zh',
        use_llm: useLlm,
      })
      setResult(res)
      await loadCustomGames() // 新游戏/变体创建成功 → 刷新管理列表
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <h1 className="page-title">创建游戏</h1>
      <p className="page-sub">用一句话描述规则，或基于已有游戏生成变体；已创建的游戏可在下方管理（删除）</p>
      {error && <div className="error-banner">{error}</div>}

      <div className="panel create-card">
        <div className="create-mode-tabs">
          <button
            className={`create-mode-btn${mode === 'from_scratch' ? ' active' : ''}`}
            onClick={() => setMode('from_scratch')}
          >
            🖊️ 规则描述
          </button>
          <button
            className={`create-mode-btn${mode === 'variant' ? ' active' : ''}`}
            onClick={() => setMode('variant')}
          >
            🔀 基于模板变体
          </button>
        </div>

        {mode === 'from_scratch' ? (
          <div>
            <label className="create-label" htmlFor="cc-rule-text">
              规则描述（自然语言，例如「8×8 棋盘，四子连珠获胜，黑棋先手」）
            </label>
            <textarea
              id="cc-rule-text"
              className="create-input"
              rows={5}
              placeholder="用一句话描述你想要的规则…"
              value={ruleText}
              onChange={(e) => setRuleText(e.target.value)}
            />
          </div>
        ) : (
          <div>
            <div className="form-row">
              <label>基础游戏:</label>
              <select value={baseGameId} onChange={(e) => setBaseGameId(e.target.value)}>
                {BASE_TEMPLATES.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.label} ({t.id})
                  </option>
                ))}
              </select>
            </div>
            <label className="create-label" htmlFor="cc-change-text">
              变更描述（例如「棋盘改成 7×7，五子连珠获胜，每步落子后 30% 概率抹去一格」）
            </label>
            <textarea
              id="cc-change-text"
              className="create-input"
              rows={5}
              placeholder="描述要在基础规则上做的改动…"
              value={changeText}
              onChange={(e) => setChangeText(e.target.value)}
            />
          </div>
        )}

        <div className="form-row" style={{ marginTop: 14 }}>
          <label>游戏名称:</label>
          <input
            type="text"
            placeholder="可选，默认用规则生成的名字"
            value={gameName}
            onChange={(e) => setGameName(e.target.value)}
          />
        </div>

        <div className="form-row">
          <label>LLM 生成:</label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
            <span>使用 LLM 翻译规则（需本地模型可用，否则自动回落确定性翻译）</span>
          </label>
        </div>

        <button className="btn btn-primary" disabled={!canSubmit} onClick={submit}>
          {busy ? (
            <span>
              <span className="spinner" /> 生成中…
            </span>
          ) : (
            '创建游戏'
          )}
        </button>
        {!canSubmit && !busy && (
          <p className="hint" style={{ marginTop: 8 }}>
            {mode === 'from_scratch' ? '请先填写规则描述' : '请先填写变更描述'}
          </p>
        )}
      </div>

      {result && (
        <div className="panel create-result">
          <div className="success-banner" style={{ marginBottom: 0 }}>
            🎉 创建成功 — 游戏 id: <strong>{result.game_id}</strong>
          </div>
          <div className="create-kv">
            <span>
              族: <span className="badge accent">{FAMILY_LABELS[result.family] ?? result.family}</span>
            </span>
            <span>
              置信度: <span className="badge">{Math.round(result.confidence * 100)}%</span>
            </span>
            <span>
              名称: <span className="badge">{result.game.display_name}</span>
            </span>
          </div>
          {result.diff_summary && (
            <p style={{ color: 'var(--muted)', fontSize: 14 }}>
              变更摘要: {result.diff_summary}
            </p>
          )}
          {result.validation.warnings.length > 0 && (
            <div>
              <div className="create-label">⚠️ 校验警告</div>
              <ul className="validation-list">
                {result.validation.warnings.map((w, i) => (
                  <li key={i} className="validation-warning-item">
                    {w}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {result.validation.errors.length > 0 && (
            <div>
              <div className="create-label">❌ 校验错误</div>
              <ul className="validation-list">
                {result.validation.errors.map((e, i) => (
                  <li key={i} className="validation-error-item">
                    {e}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div className="create-actions">
            <button className="btn btn-primary" onClick={() => navigate('/')}>
              去大厅
            </button>
            <button
              className="btn"
              onClick={() => {
                setResult(null)
                setRuleText('')
                setChangeText('')
                setGameName('')
              }}
            >
              再建一个
            </button>
          </div>
        </div>
      )}

      {/* ── 我的自定义游戏（含变体）管理 ─────────────────────────── */}
      <div className="panel" style={{ marginTop: 24 }}>
        <h3 style={{ marginBottom: 4 }}>我的自定义游戏</h3>
        <p style={{ color: 'var(--muted)', fontSize: 13, marginBottom: 14 }}>
          这里列出平台上的全部自定义游戏与模板变体（来自 /api/custom/games），可在此删除。
        </p>
        {listError && <div className="error-banner">{listError}</div>}
        {customGames.length === 0 && !listError && (
          <p style={{ color: 'var(--muted)', fontSize: 14 }}>还没有自定义游戏 — 用上面的表单创建第一个吧。</p>
        )}
        {customGames.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {customGames.map((game) => (
              <div
                key={game.game_id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 12,
                  padding: '10px 14px',
                  border: '1px solid var(--border)',
                  borderRadius: 10,
                  background: 'var(--board)',
                }}
              >
                <span className="badge accent">🛠 {FAMILY_LABELS[game.family] ?? game.family}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 600 }}>{game.display_name}</div>
                  <div style={{ color: 'var(--muted)', fontSize: 13 }}>
                    {game.game_id}
                    {game.created_at ? ` · ${new Date(game.created_at).toLocaleString()}` : ''}
                  </div>
                </div>
                <button
                  className="btn btn-danger manage-delete-btn"
                  title="删除此自定义游戏/变体"
                  disabled={deletingId !== null}
                  onClick={() => void removeCustom(game)}
                >
                  {deletingId === game.game_id ? '删除中…' : '🗑 删除'}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}