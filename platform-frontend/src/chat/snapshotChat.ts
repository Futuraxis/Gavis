// snapshotChat — 把后端 session.snapshot() 里待投递的 chat 增量（陪伴
// Agent / 教练的消息）转换成前端 ChatMessage 数组。
//
// 后端 pending_chat 在每条快照里附 ``chat: [{scenario, text, mood, step}]``
// （PlayManager.move / start / state 都会 drain 一次）。前端此前并未消费
// 这个通道——本模块把教练（教学对局）与陪伴 Agent 的实时消息落到对话
// 流里，让玩家真正看得到教练的读牌/讲评。

import type { ChatMessage, Mood, Snapshot, SnapshotChatEntry } from '../types'

function uid(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

/** 安全取出快照里的 chat 增量（Snapshot 是 union，chat 是 session 级注入键）。 */
export function readSnapshotChat(snap: Snapshot): SnapshotChatEntry[] {
  const chat = (snap as unknown as { chat?: SnapshotChatEntry[] }).chat
  return Array.isArray(chat) ? chat.filter((e) => e && typeof e.text === 'string' && e.text.trim() !== '') : []
}

/** 把快照里待投递的 chat 增量转成前端 ChatMessage（全部为 agent 消息）。 */
export function snapshotChatToMessages(snap: Snapshot): ChatMessage[] {
  return readSnapshotChat(snap).map((entry) => ({
    id: uid(),
    role: 'agent',
    text: entry.text,
    mood: (entry.mood as Mood) ?? 'neutral',
    ts: Date.now(),
  }))
}
