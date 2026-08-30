// conversationMirror — 对话的 localStorage 镜像兜底。
//
// 后端 ConversationStore（data/conversations/）是对话存档的唯一事实来源；
// 这个镜像是「后端不可用/旧服务端」时的降级层：每次消息变化同步一份
// 截断的镜像，恢复时仅当后端取档失败（网络断/服务未重启）才读它——
// 保证「一个刷新对话就没了」在最坏情况下也不会发生。
//
// 镜像只存当前活跃会话（conv_id 配对校验，防止切换会话后的串档）。

import type { ChatMessage } from '../types'

const MIRROR_KEY = 'gavis.chat.mirror.v1'
const MIRROR_MAX_MESSAGES = 200

export interface ConversationMirror {
  conv_id: string | null
  messages: ChatMessage[]
}

function isChatMessage(m: unknown): m is ChatMessage {
  if (typeof m !== 'object' || m === null) return false
  const msg = m as Record<string, unknown>
  return (
    typeof msg.id === 'string' &&
    (msg.role === 'agent' || msg.role === 'player') &&
    typeof msg.text === 'string' &&
    typeof msg.ts === 'number'
  )
}

/** 写镜像（fail-soft：隐私模式等 localStorage 不可用时静默跳过）。 */
export function writeConversationMirror(convId: string | null, messages: ChatMessage[]): void {
  try {
    const payload: ConversationMirror = { conv_id: convId, messages: messages.slice(-MIRROR_MAX_MESSAGES) }
    localStorage.setItem(MIRROR_KEY, JSON.stringify(payload))
  } catch {
    // 镜像只是兜底——写不进就算了，后端存档不受影响。
  }
}

/**
 * 读镜像。仅当镜像就是指定会话（convId 匹配，或会话尚未建档 convId === null）
 * 且有消息时返回；其余（无镜像/损坏/串档）一律 null。
 */
export function readConversationMirror(convId: string | null): ChatMessage[] | null {
  try {
    const raw = localStorage.getItem(MIRROR_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<ConversationMirror>
    const convMatch =
      (typeof parsed.conv_id === 'string' || parsed.conv_id === null) && parsed.conv_id === convId
    if (!convMatch || !Array.isArray(parsed.messages)) return null
    const messages = parsed.messages.filter(isChatMessage)
    return messages.length > 0 ? messages : null
  } catch {
    return null
  }
}
