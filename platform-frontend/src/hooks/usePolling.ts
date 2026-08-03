import { useEffect } from 'react'

/**
 * 定时轮询: active 时立即执行一次, 之后每 intervalMs 毫秒执行一次。
 * 用于评测任务进度与对局状态刷新。
 */
export function usePolling(
  fn: () => void,
  intervalMs: number,
  active: boolean,
  deps: unknown[] = [],
): void {
  useEffect(() => {
    if (!active) return
    fn()
    const timer = setInterval(fn, intervalMs)
    return () => clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, intervalMs, ...deps])
}
