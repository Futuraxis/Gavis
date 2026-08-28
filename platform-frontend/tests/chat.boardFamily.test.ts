// InlineBoard 渲染族解析测试（chat/boardFamily.ts）。
//
// 回归锚：family 未知（快照无 family 且 game 缺失 / family 为 null）时必须
// 返回 ''，绝不能回落 'grid' —— 非 grid 快照没有 board，误路由到
// GenericGridBoard 会在 board.length 上崩掉整个对话页。历史上
// mahjong_sichuan / changsha / taiwan 不在后端 _BUILTIN_FAMILY，快照与
// GameInfo 的 family 均为 null，正是那次崩溃的触发路径。
// 对应后端对齐约束：
// tests/test_layer4_interface/test_platform_session.py::
// TestGameSpecRegistry::test_builtin_family_covers_every_registry_game。

import assert from 'node:assert/strict'
import { test } from 'node:test'

import { resolveBoardFamily } from '../src/chat/boardFamily.ts'
import type { GameInfo, Snapshot } from '../src/types'

const socialSnap = { family: 'social' } as unknown as Snapshot
const plainSnap = {} as unknown as Snapshot

const gameWithFamily = { game_id: 'texas_holdem', family: 'poker' } as GameInfo
const gameNullFamily = { game_id: 'mahjong_sichuan', family: null } as unknown as GameInfo
const gameMissingFamily = { game_id: 'mahjong_changsha' } as GameInfo

test('快照显式 family 优先于 GameInfo.family', () => {
  assert.equal(resolveBoardFamily(socialSnap, gameWithFamily), 'social')
})

test('快照无 family 时取 GameInfo.family', () => {
  assert.equal(resolveBoardFamily(plainSnap, gameWithFamily), 'poker')
})

test('回归锚：game 为 null 且快照无 family → 返回空串，绝不默认 grid', () => {
  assert.equal(resolveBoardFamily(plainSnap, null), '')
})

test('回归锚：game.family 为 null 且快照无 family → 返回空串（曾崩掉的麻将变体路径）', () => {
  assert.equal(resolveBoardFamily(plainSnap, gameNullFamily), '')
})

test('回归锚：game 缺 family 字段且快照无 family → 返回空串', () => {
  assert.equal(resolveBoardFamily(plainSnap, gameMissingFamily), '')
})