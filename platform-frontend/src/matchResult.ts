// matchResult — 「相对玩家谁赢了」的唯一前端判据。
//
// 社交推理游戏（谁是卧底 / 狼人杀）的 snapshot.winner 是**阵营名**（undercover /
// civilian / blank / wolf / good），不是玩家 pid；终局身份表 snapshot.final_roles
// 在 over 时公开揭晓（含已出局者）。所有「你赢没赢」的渲染必须走这里：
//   - pid 胜者（网格/扑克/UNO/麻将单人胡）→ winner === player_pid；
//   - 阵营胜者 → 我的终局身份按阵营侧比对；
//   - 狼人杀特例：winner=good 时非狼身份（villager/seer/witch/hunter/guard）全赢；
//   - 无胜者（平局 / 血战多胡无单人胜者）→ null，调用方显示平局。
// 不这样做，卧底获胜的一局会被误标「AI 赢了」（实测对局 e7deb84b）。

import type { SocialSnapshot } from './types'

/** 胜者阵营的中文标签（无匹配 → null，回退 pid 座位称呼）。 */
export function factionLabel(winner: string): string | null {
  if (winner === 'undercover') return '卧底'
  if (winner === 'civilian') return '平民'
  if (winner === 'blank') return '白板'
  if (winner === 'wolf') return '狼人'
  if (winner === 'good') return '好人'
  return null
}

/** role 是否属于 winner 阵营（社交终局身份表用；狼人杀好人侧 = 非狼身份全属 good）。 */
export function roleBelongs(role: string | null, winner: string): boolean {
  if (role == null) return false
  if (role === winner) return true
  if (winner === 'good') return role !== 'wolf'
  if (winner === 'wolf') return role === 'wolf'
  return false
}

/**
 * 玩家视角胜负：优先后端已解析的 won；缺省时 pid 回退 + 社交终局身份表阵营比对。
 * 返回 null = 无法判定胜负（显示为平局）。
 */
export function humanWon(
  winner: string | null,
  playerPid: string,
  finalRoles?: SocialSnapshot['final_roles'],
  won?: boolean | null,
): boolean | null {
  if (typeof won === 'boolean') return won
  if (winner == null) return null
  if (winner === playerPid) return true
  const myRole = finalRoles?.find((r) => r.pid === playerPid)?.role ?? null
  if (myRole != null && roleBelongs(myRole, winner)) return true
  return false
}