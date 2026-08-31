// 本地 UI 偏好 — 主题 / 语音 / 对话静音 / 调试模式。独立于后端 profile，存 localStorage。
// 主题与「浅/深色切换」对齐 PRD 5.6；语音为 P2 占位；静音即 PRD 4.2.3 的一键静音。
// 调试模式：仅在此开关打开时前端才展示思维链（reasoning）折叠块；
// 后端照常产出 / 透传 / 存档 reasoning，默认隐藏避免把模型思考过程暴露给玩家。

export type ThemeMode = 'light' | 'dark'

const THEME_KEY = 'gavis.theme'
const VOICE_KEY = 'gavis.voice'
const MUTED_KEY = 'gavis.muted'
const DEBUG_KEY = 'gavis.debug'

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

export function getStoredDebug(): boolean {
  return localStorage.getItem(DEBUG_KEY) === 'on'
}

export function setStoredDebug(on: boolean): void {
  localStorage.setItem(DEBUG_KEY, on ? 'on' : 'off')
}
