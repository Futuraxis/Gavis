// 快照聊天增量消费测试（chat/snapshotChat.ts — 教练/陪伴消息落地对话流）。
//
// node:test 无浏览器环境：直接验证纯函数行为——chat 增量 → ChatMessage
// 映射、空/缺失 chat 的兜底、teaching 标记透传。

import assert from 'node:assert/strict'
import { test } from 'node:test'

import { readSnapshotChat, snapshotChatToMessages } from '../src/chat/snapshotChat.ts'
import type { SnapshotChatEntry } from '../src/types.ts'

function fakeSnapshot(chat?: SnapshotChatEntry[], teaching = false): Record<string, unknown> {
  return { game_id: 'abc', player_pid: 'p0', over: false, chat, teaching }
}

test('readSnapshotChat 提取非空 chat 增量', () => {
  const snap = fakeSnapshot([
    { scenario: 'teach_turn', text: '轮到你了，先看看手牌。', mood: 'thinking', step: 2 },
    { scenario: 'teach_move', text: '', mood: 'neutral', step: 3 }, // 空文本应被过滤
  ])
  const entries = readSnapshotChat(snap as never)
  assert.equal(entries.length, 1)
  assert.equal(entries[0].scenario, 'teach_turn')
})

test('readSnapshotChat 缺失 / 非数组 chat 返回空', () => {
  assert.deepEqual(readSnapshotChat(fakeSnapshot() as never), [])
  assert.deepEqual(readSnapshotChat(fakeSnapshot(null) as never), [])
  assert.deepEqual(readSnapshotChat({} as never), [])
})

test('snapshotChatToMessages 全部映射为 agent 消息', () => {
  const snap = fakeSnapshot([
    { scenario: 'teach_greet', text: '教学局开始。', mood: 'neutral', step: 0 },
    { scenario: 'teach_move', text: '这手可以更好。', mood: 'thinking', step: 4 },
  ])
  const messages = snapshotChatToMessages(snap as never)
  assert.equal(messages.length, 2)
  for (const m of messages) {
    assert.equal(m.role, 'agent')
    assert.ok(m.id)
    assert.ok(m.text.length > 0)
  }
  assert.equal(messages[0].mood, 'neutral')
  assert.equal(messages[1].mood, 'thinking')
})

test('snapshotChatToMessages 透传 speaker 标签（对手/群聊多气泡区分）', () => {
  const snap = fakeSnapshot([
    { scenario: 'opp_react', text: '我这边落定了。', mood: 'neutral', step: 1, speaker: '轻松吐槽' },
    { scenario: 'opp_read', text: '你在试探。', mood: 'thinking', step: 1, speaker: '轻松吐槽' },
  ])
  const messages = snapshotChatToMessages(snap as never)
  assert.equal(messages.length, 2)
  assert.equal(messages[0].speaker, '轻松吐槽')
  assert.equal(messages[1].speaker, '轻松吐槽')
})

test('snapshotChatToMessages 无 speaker 时缺省（旧后端兼容）', () => {
  const snap = fakeSnapshot([{ scenario: 'greet', text: '欢迎。', mood: 'neutral', step: 0 }])
  const messages = snapshotChatToMessages(snap as never)
  assert.equal(messages.length, 1)
  assert.equal(messages[0].speaker, undefined)
})
