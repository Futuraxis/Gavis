// 档案字段兜底 — 旧档案 / 新契约可能缺 recent 等字段。
// 单一可信实现：全站读取统一走这里，避免各页面各自防御、产生回归差异。
// （2026-08 前端事故：getProfile 曾经不解包 {ok, profile} 信封，导致
//   profile.recent 为 undefined、HomePage 整页崩溃；此函数是第二道防线。）

import type { Profile } from './types'

/** 安全读取近期战绩表：缺 recent 时返回空表（不会抛 TypeError）。 */
export function recentOf(profile: Partial<Profile> | null | undefined): Profile['recent'] {
  return profile?.recent ?? {}
}