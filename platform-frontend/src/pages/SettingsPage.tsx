import { useEffect, useState } from 'react'
import { clearProfile, getProfile, saveProfile } from '../api/client'
import { DEFAULT_PROFILE } from '../mock'
import {
  getStoredMuted,
  getStoredTheme,
  getStoredVoice,
  setStoredMuted,
  setStoredTheme,
  setStoredVoice,
} from '../settings'
import type { HintLevel, PersonaKey, Profile } from '../types'

const PERSONAS: { key: PersonaKey; name: string; intro: string; example: string }[] = [
  { key: 'gentle', name: '温柔陪伴', intro: '鼓励、耐心、在意你的感受。', example: '“没关系，这局运气差了点，再来一局？”' },
  { key: 'teacher', name: '认真教学', intro: '讲规则、讲原因、给建议。', example: '“这手先占角更好，后面展开空间大。”' },
  { key: 'banter', name: '轻松吐槽', intro: '幽默、活跃气氛、不伤人。', example: '“这手棋很有想法，就是有点费棋盘。”' },
  { key: 'cold', name: '高冷竞技', intro: '少说话、只报关键信息。', example: '“轮到你了。”' },
]

const HINT_LEVELS: { value: HintLevel; label: string }[] = [
  { value: 'off', label: '关闭' },
  { value: 'direction', label: '方向提示' },
  { value: 'specific', label: '具体建议' },
  { value: 'demo', label: '演示' },
]

export default function SettingsPage() {
  const [profile, setProfile] = useState<Profile>(DEFAULT_PROFILE)
  const [theme, setTheme] = useState(getStoredTheme())
  const [voice, setVoice] = useState(getStoredVoice())
  const [muted, setMuted] = useState(getStoredMuted())
  const [toast, setToast] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    getProfile()
      .then((p) => {
        setProfile(p)
        if (p.theme === 'dark' || p.theme === 'light') setTheme(p.theme)
      })
      .catch(() => setProfile(DEFAULT_PROFILE))
  }, [])

  function patch(next: Partial<Profile>) {
    setProfile((prev) => ({ ...prev, ...next }))
  }

  async function persist(next: Profile) {
    setSaving(true)
    setToast(null)
    try {
      await saveProfile(next)
      setToast('已保存 ✓')
    } catch {
      setToast('已保存（本地）✓ 后端 /profile 接线后将持久化')
    } finally {
      setSaving(false)
    }
  }

  function toggleTheme() {
    const next = theme === 'light' ? 'dark' : 'light'
    setTheme(next)
    setStoredTheme(next)
    patch({ theme: next })
  }

  function toggleVoice() {
    const next = !voice
    setVoice(next)
    setStoredVoice(next)
  }

  function toggleMuted() {
    const next = !muted
    setMuted(next)
    setStoredMuted(next)
  }

  async function clearData() {
    setToast(null)
    try {
      await clearProfile()
      setToast('个人数据已清除 ✓')
    } catch {
      setToast('个人数据已清除（本地）✓')
    }
    setProfile(DEFAULT_PROFILE)
    setStoredVoice(true)
    setStoredMuted(false)
    setVoice(true)
    setMuted(false)
  }

  const hintIndex = Math.max(0, HINT_LEVELS.findIndex((h) => h.value === profile.hint_level))

  return (
    <div>
      <h1 className="page-title">Agent 设置</h1>
      <p className="page-sub">性格、提示、语音与外观</p>
      {toast && <div className="success-banner">{toast}</div>}

      <section className="panel" style={{ marginBottom: 18 }}>
        <h3 style={{ marginBottom: 14 }}>性格</h3>
        <div className="persona-grid">
          {PERSONAS.map((p) => (
            <div
              key={p.key}
              className={`persona-card ${profile.default_persona === p.key ? 'selected' : ''}`}
              onClick={() => patch({ default_persona: p.key })}
            >
              <h4>{p.name}</h4>
              <p className="persona-intro">{p.intro}</p>
              <p className="persona-example">{p.example}</p>
              <button
                className="btn"
                onClick={(e) => {
                  e.stopPropagation()
                  setToast(`试听（占位）：${p.example}`)
                }}
              >
                试听
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="panel" style={{ marginBottom: 18 }}>
        <h3 style={{ marginBottom: 14 }}>提示级别</h3>
        <input
          type="range"
          min={0}
          max={3}
          step={1}
          value={hintIndex}
          onChange={(e) => patch({ hint_level: HINT_LEVELS[Number(e.target.value)].value })}
        />
        <div className="hint-levels">
          {HINT_LEVELS.map((h, i) => (
            <span key={h.value} className={i === hintIndex ? 'active' : ''}>
              {h.label}
            </span>
          ))}
        </div>
      </section>

      <section className="panel" style={{ marginBottom: 18 }}>
        <h3 style={{ marginBottom: 14 }}>开关</h3>
        <div className="form-row">
          <label>语音朗读:</label>
          <button className={`btn ${voice ? '' : 'btn-ghost'}`} onClick={toggleVoice}>
            {voice ? '开' : '关'}
          </button>
          <span style={{ color: 'var(--muted)', fontSize: 13 }}>（P2 占位）</span>
        </div>
        <div className="form-row">
          <label>对话:</label>
          <button className={`btn ${muted ? 'btn-ghost' : ''}`} onClick={toggleMuted}>
            {muted ? '已静音' : '开启'}
          </button>
        </div>
        <div className="form-row">
          <label>主题:</label>
          <button className="btn" onClick={toggleTheme}>
            {theme === 'light' ? '☀️ 浅色' : '🌙 深色'}
          </button>
        </div>
      </section>

      <section className="panel" style={{ marginBottom: 18 }}>
        <h3 style={{ marginBottom: 14 }}>数据管理</h3>
        <button className="btn btn-danger" onClick={clearData}>
          一键清除个人数据
        </button>
      </section>

      <div className="form-row">
        <button className="btn btn-primary" disabled={saving} onClick={() => persist(profile)}>
          {saving ? '保存中…' : '保存设置'}
        </button>
      </div>
    </div>
  )
}
