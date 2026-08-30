// 会话状态持久化测试（chat/sessionStore.ts — viewMode + 当前对局）。
//
// node:test 无浏览器环境：这里给 globalThis 装上 localStorage 桩，
// 并把 window 指向 globalThis，验证默认值、读写持久化与全局切模式事件。

import assert from 'node:assert/strict'
import { test } from 'node:test'

// ── localStorage 桩（纯内存 Map）＋ window 桩（EventTarget，可监听事件） ──

function installStubs(): { target: EventTarget; restore: () => void } {
  const stored = new Map<string, string>()
  const storage: Storage = {
    get length() {
      return stored.size
    },
    clear: () => stored.clear(),
    getItem: (key: string) => (stored.has(key) ? stored.get(key)! : null),
    key: (index: number) => [...stored.keys()][index] ?? null,
    removeItem: (key: string) => {
      stored.delete(key)
    },
    setItem: (key: string, value: string) => {
      stored.set(key, value)
    },
  }
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: storage,
  })
  // Node 的 globalThis 没有 addEventListener —— 用独立 EventTarget 充当 window。
  const target = new EventTarget()
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: target,
  })
  return {
    target,
    restore: () => {
      stored.clear()
      delete (globalThis as Record<string, unknown>).localStorage
      delete (globalThis as Record<string, unknown>).window
    },
  }
}

import { VIEW_EVENT, PLATFORM_EVENT, goChat, loadChatStore, openPlatform, saveChatStore } from '../src/chat/sessionStore.ts'

test('默认：viewMode=chat、无活跃对局（对话即一切为主）', () => {
  const { restore } = installStubs()
  try {
    const s = loadChatStore()
    assert.equal(s.viewMode, 'chat')
    assert.equal(s.activeGameId, null)
    assert.equal(s.boardCollapsed, false)
  } finally {
    restore()
  }
})

test('saveChatStore：只补丁指定字段，其余保留', () => {
  const { restore } = installStubs()
  try {
    saveChatStore({ activeGameId: 'sess-9' })
    const s = loadChatStore()
    assert.equal(s.activeGameId, 'sess-9')
    assert.equal(s.viewMode, 'chat') // 未动 viewMode，仍为默认
  } finally {
    restore()
  }
})

test('viewMode 持久化：platform 写回后可恢复', () => {
  const { restore } = installStubs()
  try {
    saveChatStore({ viewMode: 'platform', activeGameId: 'sess-9' })
    const s = loadChatStore()
    assert.equal(s.viewMode, 'platform')
    assert.equal(s.activeGameId, 'sess-9')
  } finally {
    restore()
  }
})

test('boardCollapsed 持久化：收起界面写回后可恢复（专心对话的选择跨刷新保留）', () => {
  const { restore } = installStubs()
  try {
    saveChatStore({ boardCollapsed: true })
    assert.equal(loadChatStore().boardCollapsed, true)
    // 只补丁指定字段：其它字段不受影响
    saveChatStore({ activeGameId: 'sess-1' })
    const s = loadChatStore()
    assert.equal(s.boardCollapsed, true)
    assert.equal(s.activeGameId, 'sess-1')
  } finally {
    restore()
  }
})

test('boardCollapsed 容错：旧数据/非布尔值 → 视为未收起', () => {
  const { restore } = installStubs()
  try {
    const ls = (globalThis as { localStorage: Storage }).localStorage
    // 旧版本存储（无 boardCollapsed 字段）
    ls.setItem('gavis.chat.v1', JSON.stringify({ viewMode: 'chat', activeGameId: 'sess-9' }))
    assert.equal(loadChatStore().boardCollapsed, false)
    // 非法类型同样回落 false
    ls.setItem('gavis.chat.v1', JSON.stringify({ boardCollapsed: 'yes' }))
    assert.equal(loadChatStore().boardCollapsed, false)
  } finally {
    restore()
  }
})

test('conversationId 持久化：存档会话 id 写回后可恢复，旧数据/坏类型 → null', () => {
  const { restore } = installStubs()
  try {
    saveChatStore({ conversationId: 'conv-abc123' })
    assert.equal(loadChatStore().conversationId, 'conv-abc123')
    // 只补丁指定字段：其它字段不受影响
    assert.equal(loadChatStore().viewMode, 'chat')
    // 旧版本存储（无 conversationId 字段）→ null（新开对话）
    const ls = (globalThis as { localStorage: Storage }).localStorage
    ls.setItem('gavis.chat.v1', JSON.stringify({ viewMode: 'chat' }))
    assert.equal(loadChatStore().conversationId, null)
    // 非字符串（数字等坏数据）→ null
    ls.setItem('gavis.chat.v1', JSON.stringify({ conversationId: 42 }))
    assert.equal(loadChatStore().conversationId, null)
  } finally {
    restore()
  }
})

test('坏数据容错：非法 JSON / 非法 viewMode → 回落默认（局部存储损坏不炸页面）', () => {
  const { restore } = installStubs()
  try {
    ;(globalThis as { localStorage: Storage }).localStorage.setItem('gavis.chat.v1', '{oops')
    assert.equal(loadChatStore().viewMode, 'chat')
    ;(globalThis as { localStorage: Storage }).localStorage.setItem('gavis.chat.v1', JSON.stringify({ viewMode: 'sideways' }))
    assert.equal(loadChatStore().viewMode, 'chat') // 非 platform 一律视为 chat
  } finally {
    restore()
  }
})

test('goChat / openPlatform：派发全局事件 + 持久化 viewMode', () => {
  const { target, restore } = installStubs()
  try {
    const seen: string[] = []
    target.addEventListener(VIEW_EVENT, () => seen.push('chat'))
    target.addEventListener(PLATFORM_EVENT, () => seen.push('platform'))
    openPlatform()
    assert.deepEqual(seen, ['platform'])
    assert.equal(loadChatStore().viewMode, 'platform')
    goChat()
    assert.deepEqual(seen, ['platform', 'chat'])
    assert.equal(loadChatStore().viewMode, 'chat')
  } finally {
    restore()
  }
})