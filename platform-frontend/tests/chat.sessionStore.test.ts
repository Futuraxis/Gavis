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