// 对话镜像持久化测试（chat/conversationMirror.ts）。
//
// 回归锚：对话曾只活在组件内存里——刷新即清零。镜像层保证后端
// ConversationStore 不可用时（旧服务端/断连），localStorage 里仍有最近
// 一段可恢复的对话流。node:test 无浏览器环境：装 localStorage 桩。

import assert from 'node:assert/strict'
import { test } from 'node:test'

function installStorage(): { restore: () => void } {
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
  return {
    restore: () => {
      stored.clear()
      delete (globalThis as Record<string, unknown>).localStorage
    },
  }
}

import { readConversationMirror, writeConversationMirror } from '../src/chat/conversationMirror.ts'

function msg(id: string, role: 'agent' | 'player', text: string) {
  return { id, role, text, ts: 1700000000000 }
}

test('写入后按会话 id 读回（正常的恢复路径）', () => {
  const { restore } = installStorage()
  try {
    const messages = [msg('u1', 'player', '玩月亮棋'), msg('a1', 'agent', '好，来一局！')]
    writeConversationMirror('conv-7', messages)
    const read = readConversationMirror('conv-7')
    assert.deepEqual(read, messages)
  } finally {
    restore()
  }
})

test('会话不匹配（串档防护）：换了会话不读旧镜像', () => {
  const { restore } = installStorage()
  try {
    writeConversationMirror('conv-7', [msg('u1', 'player', 'hi')])
    // 切到另一段对话（或新开对话 conv_id=null）——旧镜像不属于它
    assert.equal(readConversationMirror('conv-8'), null)
    assert.equal(readConversationMirror(null), null)
  } finally {
    restore()
  }
})

test('坏数据容错：非法 JSON / 结构不对 → null（不炸恢复流程）', () => {
  const { restore } = installStorage()
  try {
    const ls = (globalThis as { localStorage: Storage }).localStorage
    ls.setItem('gavis.chat.mirror.v1', '{oops')
    assert.equal(readConversationMirror('conv-7'), null)
    ls.setItem('gavis.chat.mirror.v1', JSON.stringify({ conv_id: 'conv-7', messages: 'not-array' }))
    assert.equal(readConversationMirror('conv-7'), null)
    // 消息结构不合法的条目被过滤，剩下的仍可恢复
    ls.setItem(
      'gavis.chat.mirror.v1',
      JSON.stringify({ conv_id: 'conv-7', messages: [{ nope: 1 }, msg('u1', 'player', 'ok')] }),
    )
    assert.deepEqual(readConversationMirror('conv-7'), [msg('u1', 'player', 'ok')])
  } finally {
    restore()
  }
})

test('空镜像（无消息）→ null（避免恢复出空对话）', () => {
  const { restore } = installStorage()
  try {
    writeConversationMirror('conv-7', [])
    assert.equal(readConversationMirror('conv-7'), null)
  } finally {
    restore()
  }
})

test('写入截断到最近 200 条（镜像不无限膨胀）', () => {
  const { restore } = installStorage()
  try {
    const many = Array.from({ length: 260 }, (_, i) => msg(`m${i}`, 'player', `消息${i}`))
    writeConversationMirror('conv-7', many)
    const read = readConversationMirror('conv-7')
    assert.equal(read!.length, 200)
    assert.equal(read![0].id, 'm60') // 最早的 60 条被裁掉
    assert.equal(read![199].id, 'm259')
  } finally {
    restore()
  }
})

test('localStorage 不可用时静默失败（写不炸、读返回 null）', () => {
  const { restore } = installStorage()
  restore() // 立即卸掉桩 → localStorage 未定义（隐私模式同款场景）
  try {
    writeConversationMirror('conv-7', [msg('u1', 'player', 'hi')]) // 不应抛
    assert.equal(readConversationMirror('conv-7'), null)
  } finally {
    // nothing to restore
  }
})
