// sse — 纯 SSE 帧解析（跨 chunk 自动缓冲），零依赖，供 chatTurnStream 消费。
//
// 后端 /api/chat 流式模式的事件契约（见 layer4_interface/frontend/platform/chat.py
// 的 chat_turn_stream）：event: reasoning / text / intent / error / done。
// 本模块只做「字节流 → 事件帧」机械解析，不解业务语义 —— 业务在 client.ts。

export interface SseEvent {
  event: string
  data: string
}

/** 把一帧（不含结尾空行）解析成事件；无 data 行返回 null。 */
export function parseEvent(frame: string): SseEvent | null {
  const lines = frame.split('\n')
  let event = 'message'
  const dataLines: string[] = []
  for (const line of lines) {
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trimStart())
    }
    // 注释行 / 未知字段行天然忽略（SSE 规范）。
  }
  if (dataLines.length === 0) return null
  return { event, data: dataLines.join('\n') }
}

/**
 * 增量解析器：喂任意分块文本，产出完整事件；跨块残留自动缓冲。
 * 帧分隔为空行（\n\n）；流结束调用 finish() 吐出缓冲尾帧（若有）。
 */
export class SseParser {
  private buffer = ''

  push(chunk: string): SseEvent[] {
    if (!chunk) return []
    this.buffer += chunk
    const events: SseEvent[] = []
    let idx: number
    while ((idx = this.buffer.indexOf('\n\n')) !== -1) {
      const frame = this.buffer.slice(0, idx)
      this.buffer = this.buffer.slice(idx + 2)
      const ev = parseEvent(frame)
      if (ev) events.push(ev)
    }
    return events
  }

  finish(): SseEvent[] {
    if (!this.buffer.trim()) return []
    const ev = parseEvent(this.buffer)
    this.buffer = ''
    return ev ? [ev] : []
  }
}