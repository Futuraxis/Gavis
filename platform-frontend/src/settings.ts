// 本地 UI 偏好 — 主题 / 语音 / 对话静音。独立于后端 profile，存 localStorage。
// 主题与「浅/深色切换」对齐 PRD 5.6；语音为 P2 占位；静音即 PRD 4.2.3 的一键静音。

export type ThemeMode = 'light' | 'dark'

const THEME_KEY = 'gavis.theme'
const VOICE_KEY = 'gavis.voice'
const MUTED_KEY = 'gavis.muted'

export function getStoredTheme(): ThemeMode {
  const v = localStorage.getItem(THEME_KEY)
  return v === 'dark' ? 'dark' : 'light'
}

export function applyStoredTheme(): void {
  document.documentElement.setAttribute('data-theme', getStoredTheme())
}

export function setStoredTheme(theme: ThemeMode): void {
  localStorage.setItem(THEME_KEY, theme)
  document.documentElement.setAttribute('data-theme', theme)
}

export function getStoredVoice(): boolean {
  return localStorage.getItem(VOICE_KEY) !== 'off'
}

export function setStoredVoice(on: boolean): void {
  localStorage.setItem(VOICE_KEY, on ? 'on' : 'off')
}

export function getStoredMuted(): boolean {
  return localStorage.getItem(MUTED_KEY) === 'on'
}

export function setStoredMuted(on: boolean): void {
  localStorage.setItem(MUTED_KEY, on ? 'on' : 'off')
}
