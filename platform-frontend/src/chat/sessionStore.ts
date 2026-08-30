// Chat-first 会话状态 — 持久化「对话即一切」模式的核心开关与当前对局。
// 平台界面仍保留（口令/按钮召唤），这里只存最少的跨页状态。

export type ViewMode = 'chat' | 'platform'

export interface ChatStore {
  viewMode: ViewMode
  activeGameId: string | null
  /** 对局界面已收起（「专心对话」模式）：对局不结束、后台继续，随时可再展开。 */
  boardCollapsed: boolean
  /** 当前对话存档 id（后端 data/conversations/<id>.json；null = 新对话，首条消息时懒建）。 */
  conversationId: string | null
}

const KEY = 'gavis.chat.v1'
const VIEW_EVENT = 'gavis:view-chat'
const PLATFORM_EVENT = 'gavis:view-platform'

const DEFAULT_STORE: ChatStore = {
  viewMode: 'chat',
  activeGameId: null,
  boardCollapsed: false,
  conversationId: null,
}

export function loadChatStore(): ChatStore {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return { ...DEFAULT_STORE }
    const parsed = JSON.parse(raw) as Partial<ChatStore>
    return {
      viewMode: parsed.viewMode === 'platform' ? 'platform' : 'chat',
      activeGameId: typeof parsed.activeGameId === 'string' ? parsed.activeGameId : null,
      boardCollapsed: parsed.boardCollapsed === true,
      conversationId: typeof parsed.conversationId === 'string' ? parsed.conversationId : null,
    }
  } catch {
    return { ...DEFAULT_STORE }
  }
}

export function saveChatStore(patch: Partial<ChatStore>): ChatStore {
  const next = { ...loadChatStore(), ...patch }
  try {
    localStorage.setItem(KEY, JSON.stringify(next))
  } catch {
    // localStorage 不可用（隐私模式等）时仅保留内存态。
  }
  return next
}

/** 任何时候切回对话模式（全局事件，App 订阅）。 */
export function goChat(): void {
  saveChatStore({ viewMode: 'chat' })
  window.dispatchEvent(new CustomEvent(VIEW_EVENT))
}

/** 打开完整平台界面（ChatPage 头部按钮 / 平台口令触发）。 */
export function openPlatform(): void {
  saveChatStore({ viewMode: 'platform' })
  window.dispatchEvent(new CustomEvent(PLATFORM_EVENT))
}

export { VIEW_EVENT, PLATFORM_EVENT }