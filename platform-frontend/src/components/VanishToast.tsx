import { useEffect, useRef, useState } from 'react'
import type { Snapshot } from '../types'

/** 五子棋棋子被随机抹去时的提示条 — 出现约 1.8s 后自动消失。 */
export default function VanishToast({ snapshot }: { snapshot: Snapshot }) {
  const [visible, setVisible] = useState(false)
  const [key, setKey] = useState(0)
  const prevVanish = useRef<number | null>(null)
  const vanished = 'last_vanish' in snapshot ? snapshot.last_vanish : null

  useEffect(() => {
    if (vanished != null && vanished !== prevVanish.current) {
      prevVanish.current = vanished
      setKey((k) => k + 1)
      setVisible(true)
      const timer = setTimeout(() => setVisible(false), 1800)
      return () => clearTimeout(timer)
    }
  }, [vanished])

  if (!visible) return null
  return (
    <span className="vanish-toast" key={key}>
      💨 棋子被随机抹去了！
    </span>
  )
}
