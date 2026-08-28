// boardFamily — 会话渲染族解析（InlineBoard 复用；纯逻辑无 JSX，便于 Node 测试）。
//
// 防回归锚：family 未知时返回空串，**绝不默认 grid** —— 非 grid 快照没有
// board 字段，若把 poker/mahjong 快照误路由到 GenericGridBoard，会在
// board.length 处抛 TypeError 崩掉整个对话页（历史上 mahjong_sichuan /
// changsha / taiwan 不在后端 _BUILTIN_FAMILY，GameInfo.family 为 null，
// 正是那次线上崩溃的根因）。查不到 family 的调用方应按 game.kind 兜底
// （对齐 BattlePage / InlineBoard 的分发逻辑）。

import type { GameInfo, Snapshot } from '../types'

/** 快照显式 family > GameInfo.family > 未知（''）。 */
export function resolveBoardFamily(snapshot: Snapshot, game: GameInfo | null): string {
  return snapshot.family ?? game?.family ?? ''
}