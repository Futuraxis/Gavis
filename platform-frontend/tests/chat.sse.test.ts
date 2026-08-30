// SSE 帧解析器契约测试（src/chat/sse.ts）。
//
// 后端 /api/chat 流式模式按 chat_turn_stream 事件契约发帧：
//   event: reasoning / text / intent / error / done
// 这里锁定「字节流 → 事件帧」的机械解析：跨块缓冲、注释行容错、
// 无 data 行丢弃、finish() 收尾。业务语义由 chat.client-api.test.ts 覆盖。

import assert from 'node:assert/strict'
import { test } from 'node:test'

import { SseParser, parseEvent } from '../src/chat/sse.ts'

test('parseEvent：event + data 行解析', () => {
  const ev = parseEvent('event: text\ndata: {"delta":"你好"}')
  assert.ok(ev)
  assert.equal(ev.event, 'text')
  assert.equal(ev.data, '{"delta":"你好"}')
})

test('parseEvent：只有 data 行时事件名为默认 message', () => {
  const ev = parseEvent('data: {"ok":true}')
  assert.ok(ev)
  assert.equal(ev.event, 'message')
  assert.equal(ev.data, '{"ok":true}')
})

test('parseEvent：多 data 行以换行拼接', () => {
  const ev = parseEvent('data: 第一行\ndata: 第二行')
  assert.ok(ev)
  assert.equal(ev.data, '第一行\n第二行')
})

test('parseEvent：注释行 / 未知字段行忽略，无 data 行返回 null', () => {
  assert.equal(parseEvent(': keep-alive\n: 注释'), null)
  assert.equal(parseEvent('event: text'), null)
  assert.equal(parseEvent(''), null)
})

test('SseParser：完整帧一次 push 即出事件', () => {
  const parser = new SseParser()
  const events = parser.push('event: text\ndata: {"delta":"a"}\n\nevent: done\ndata: {}\n\n')
  assert.equal(events.length, 2)
  assert.equal(events[0].event, 'text')
  assert.equal(events[1].event, 'done')
})

test('SseParser：跨块分片（帧内断行）自动缓冲拼接', () => {
  const parser = new SseParser()
  const frame = 'event: text\ndata: {"delta":"流式文本"}\n\n'
  const cut = Math.floor(frame.length / 2)
  const first = parser.push(frame.slice(0, cut))
  const second = parser.push(frame.slice(cut))
  assert.equal(first.length, 0) // 半帧不吐
  assert.equal(second.length, 1)
  assert.equal(second[0].event, 'text')
  assert.equal(second[0].data, '{"delta":"流式文本"}')
})

test('SseParser：块尾留残缺行，finish 时吐出（完整帧语义）', () => {
  const parser = new SseParser()
  assert.equal(parser.push('event: done\ndata: {}').length, 0) // 缺帧尾空行
  const events = parser.finish()
  assert.equal(events.length, 1)
  assert.equal(events[0].event, 'done')
})

test('SseParser：空块 / 无残留 finish 不产出事件', () => {
  const parser = new SseParser()
  assert.equal(parser.push('').length, 0)
  assert.equal(parser.finish().length, 0)
})