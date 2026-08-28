// Chat-first 运行时 — 一句话 → 意图 → 平台动作（对话即一切的核心编排）。
//
// 主路径：POST /api/chat（后端 LLM function calling + schema 校验 + 正则兜底）
// 返回 {intent, text, mood, params}；本 Hook 把意图翻译成平台动作：
// play→开局配置卡、resume/move/restart→对局快照、history/review→战绩/复盘卡、
// create→创建面板、benchmark/learning→进度卡、platform/settings→切回平台界面。
//
// 快速落子红线：棋盘/牌面点击直接走 /match/move，**不经 LLM**——LLM 只处理
// 文本表达的动作（碰/跟注/打五条等），点击永远是即时快路径。

import { useCallback, useEffect, useRef, useState } from 'react'
import { apiGet, apiPost, chatTurn, matchHint } from '../api/client'
import type {
  ChatMessage,
  ChatTurnResult,
  GameInfo,
  MatchMeta,
  ReviewReport,
  Snapshot,
} from '../types'
import type { BattleConfig } from '../components/BattleSetup'
import { classifyLocal } from './intents'
import { loadChatStore, openPlatform, saveChatStore } from './sessionStore'

export interface StatsData {
  matches: MatchMeta[]
  wins: number
  plays: number
}

export interface BenchmarkJob {
  job_id: string
  game_id: string
  solver_a: string
  solver_b: string
  iterations: number
  budget: number | null
  status: string
  progress: number
}

export interface LearningItem {
  game_id: string
  enabled: boolean
  matches: number
  decisions: number
  model?: { version: number } | null
}

export interface ChatRuntime {
  messages: ChatMessage[]
  busy: boolean
  error: string | null
  games: GameInfo[]
  activeSession: Snapshot | null
  activeGameInfo: GameInfo | null
  send: (text: string) => Promise<void>
  moveAction: (action: unknown) => Promise<void>
  startSession: (gameId: string, config: BattleConfig) => Promise<void>
  notifyCreated: (game: GameInfo) => void
  clearSession: () => void
}

function uid(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

export function useChatRuntime(): ChatRuntime {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [games, setGames] = useState<GameInfo[]>([])
  const [activeSession, setActiveSession] = useState<Snapshot | null>(null)
  const [, setActiveGameId] = useState<string | null>(() => loadChatStore().activeGameId)
  const busyRef = useRef(false)
  const gamesRef = useRef<GameInfo[]>([])
  // 镜像 messages 的 ref（useEffect 同步），send 里取「已渲染完的消息」当对话历史；
  // 刚 push 的当前句不在其中，它本来就是 /api/chat 的 text 参数，不会重复。
  const messagesRef = useRef<ChatMessage[]>([])

  useEffect(() => {
    gamesRef.current = games
  }, [games])

  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  // 初始数据：游戏目录 + 恢复上次对局。
  useEffect(() => {
    apiGet<{ games: GameInfo[] }>('/games')
      .then((d) => {
        setGames(d.games)
        gamesRef.current = d.games
      })
      .catch((err: Error) => setError(err.message))
    const stored = loadChatStore().activeGameId
    if (stored) {
      apiPost<{ session: Snapshot }>('/match/state', { game_id: stored })
        .then((d) => {
          setActiveSession(d.session)
          if (d.session.over) setActiveGameId(null)
        })
        .catch(() => setActiveGameId(null))
    }
  }, [])

  const pushPlayer = useCallback((text: string) => {
    setMessages((prev) => [...prev, { id: uid(), role: 'player', text, ts: Date.now() }])
  }, [])

  const pushAgent = useCallback(
    (text: string, mood?: ChatMessage['mood'], intent?: ChatMessage['intent'], params?: Record<string, unknown>) => {
      setMessages((prev) => [...prev, { id: uid(), role: 'agent', text, mood, ts: Date.now(), intent, params }])
    },
    [],
  )

  const setBusyState = useCallback((v: boolean) => {
    busyRef.current = v
    setBusy(v)
  }, [])

  const refreshSession = useCallback(async (gameId: string): Promise<void> => {
    const d = await apiPost<{ session: Snapshot }>('/match/state', { game_id: gameId })
    setActiveSession(d.session)
    if (d.session.over) {
      setActiveGameId(null)
      saveChatStore({ activeGameId: null })
    } else {
      setActiveGameId(gameId)
      saveChatStore({ activeGameId: gameId })
    }
  }, [])

  const startSession = useCallback(
    async (gameId: string, config: BattleConfig): Promise<void> => {
      setBusyState(true)
      setError(null)
      try {
        const data = await apiPost<{ session: Snapshot }>('/match/start', {
          game_id: gameId,
          player_pid: config.playerPid,
          difficulty: config.difficulty,
          player_count: config.playerCount,
          persona: config.persona,
          hint_level: config.hintLevel,
          pacing: config.pacing,
          adaptive: config.adaptive,
        })
        setActiveSession(data.session)
        setActiveGameId(gameId)
        saveChatStore({ activeGameId: gameId })
        const name = gamesRef.current.find((g) => g.game_id === gameId)?.display_name ?? gameId
        pushAgent(`对局已开始：${name} 🎮 轮到你了就下，也可以随时问我“这步怎么走”。`, 'happy')
      } catch (err) {
        setError((err as Error).message)
        pushAgent(`开局失败：${(err as Error).message}`, 'sorry')
      } finally {
        setBusyState(false)
      }
    },
    [pushAgent, setBusyState],
  )

  const moveAction = useCallback(
    async (action: unknown): Promise<void> => {
      if (busyRef.current || !activeSession?.game_id) return
      setBusyState(true)
      setError(null)
      try {
        const data = await apiPost<{ session: Snapshot }>('/match/move', {
          game_id: activeSession.game_id,
          action,
        })
        setActiveSession(data.session)
        if (!data.session.over) setActiveGameId(data.session.game_id)
      } catch (err) {
        setError((err as Error).message)
        pushAgent(`这一步没走成：${(err as Error).message}`, 'sorry')
      } finally {
        setBusyState(false)
      }
    },
    [activeSession, pushAgent, setBusyState],
  )

  // ── 意图 → 平台动作 ───────────────────────────────────────────

  const fetchStats = useCallback(async (): Promise<StatsData> => {
    const d = await apiGet<{ matches: MatchMeta[] }>('/history?limit=10')
    const matches = d.matches ?? []
    const wins = matches.filter((m) => m.winner === m.player_pid).length
    return { matches, wins, plays: matches.length }
  }, [])

  const dispatch = useCallback(
    async (result: ChatTurnResult): Promise<void> => {
      const { intent, text, mood } = result
      const params = result.params ?? {}
      switch (intent) {
        case 'play':
          pushAgent(text, mood, 'play', { game_id: params.game_id })
          break
        case 'resume': {
          const gameId = String(params.game_id ?? '')
          if (!gameId) {
            pushAgent('当前没有进行中的对局。', 'neutral')
            break
          }
          try {
            await refreshSession(gameId)
            pushAgent(text, mood, 'resume', { game_id: gameId })
          } catch (err) {
            pushAgent(`恢复对局失败：${(err as Error).message}`, 'sorry')
          }
          break
        }
        case 'move':
          pushAgent(text, mood, 'move', {})
          // 直接快路径落子（外层 send 已置 busy，moveAction 自身的守卫会误拦，
          // 因此文本意图落子在这里直连 /match/move）。
          if (activeSession && !activeSession.over) {
            try {
              const data = await apiPost<{ session: Snapshot }>('/match/move', {
                game_id: activeSession.game_id,
                action: params.action,
              })
              setActiveSession(data.session)
              if (!data.session.over) setActiveGameId(data.session.game_id)
            } catch (err) {
              pushAgent(`这一步没走成：${(err as Error).message}`, 'sorry')
            }
          }
          break
        case 'hint': {
          if (!activeSession) {
            pushAgent('现在没有对局，先来一局吧。', 'neutral')
            break
          }
          try {
            const hint = await matchHint(activeSession.game_id, 'direction')
            pushAgent(hint.text, hint.mood, 'hint', {})
          } catch {
            pushAgent('提示暂时不可用。', 'neutral')
          }
          break
        }
        case 'restart': {
          const gameId = String(params.game_id ?? '')
          if (!gameId) {
            pushAgent('现在没有可重开的对局。', 'neutral')
            break
          }
          await startSession(gameId, {
            playerPid: activeSession?.player_pid ?? 'random',
            difficulty: activeSession?.difficulty ?? 'easy',
            playerCount: games.find((g) => g.game_id === gameId)?.player_counts[0] ?? 2,
            persona: 'gentle',
            hintLevel: 'off',
            pacing: 'standard',
            adaptive: true,
          })
          break
        }
        case 'history': {
          try {
            const data = await fetchStats()
            pushAgent(text, mood, 'history', { matches: data.matches, wins: data.wins, plays: data.plays })
          } catch (err) {
            pushAgent(`取战绩失败：${(err as Error).message}`, 'sorry')
          }
          break
        }
        case 'review': {
          try {
            const matches = (await fetchStats()).matches
            const latest = matches[0]
            if (!latest) {
              pushAgent('还没有可复盘的对局。', 'neutral')
              break
            }
            const d = await apiGet<{ report: ReviewReport }>('/review/' + latest.match_id)
            pushAgent(text, mood, 'review', { report: d.report, match_id: latest.match_id })
          } catch (err) {
            pushAgent(`复盘失败：${(err as Error).message}`, 'sorry')
          }
          break
        }
        case 'create':
          pushAgent(text, mood, 'create', {})
          break
        case 'settings':
          openPlatform()
          window.location.hash = '#/settings'
          break
        case 'platform':
          openPlatform()
          break
        case 'benchmark': {
          try {
            const d = await apiGet<{ jobs: BenchmarkJob[] }>('/benchmark')
            pushAgent(text, mood, 'benchmark', { jobs: d.jobs ?? [] })
          } catch (err) {
            pushAgent(`评测中心不可用：${(err as Error).message}`, 'sorry')
          }
          break
        }
        case 'learning': {
          try {
            const d = await apiGet<{ learning: LearningItem[] }>('/learning/status')
            pushAgent(text, mood, 'learning', { learning: d.learning ?? [] })
          } catch (err) {
            pushAgent(`在线学习不可用：${(err as Error).message}`, 'sorry')
          }
          break
        }
        case 'help':
          pushAgent(text, mood)
          break
        case 'chat':
          pushAgent(text, mood, 'chat', {})
          break
        case 'clarify':
          pushAgent(text, mood, 'clarify', { chips: params.chips ?? [] })
          break
      }
    },
    [activeSession, fetchStats, pushAgent, refreshSession, startSession],
  )

  const notifyCreated = useCallback(
    (game: GameInfo) => {
      pushAgent(`《${game.display_name}》创建成功！想马上来一局？直接说“玩${game.display_name}”。`, 'happy', 'play', {
        game_id: game.game_id,
      })
    },
    [pushAgent],
  )

  const send = useCallback(
    async (text: string): Promise<void> => {
      const trimmed = text.trim()
      if (!trimmed || busyRef.current) return
      pushPlayer(trimmed)
      setBusyState(true)
      setError(null)
      try {
        // 对话历史：最近 24 条 user/assistant 消息（player→user, agent→assistant），
        // 只带文本；当前输入作为 text 单独传。让 LLM 能接住“那德州扑克呢”这类回指。
        const history = messagesRef.current
          .slice(-24)
          .filter((m) => m.text.trim().length > 0)
          .map((m) => ({
            role: m.role === 'player' ? ('user' as const) : ('assistant' as const),
            content: m.text,
          }))
        const result = await chatTurn(trimmed, activeSession?.game_id, history)
        await dispatch(result)
      } catch {
        // 后端 /api/chat 不可用（旧服务端/断连）→ 本地正则兜底。
        const local = classifyLocal(trimmed, {
          games: gamesRef.current.map((g) => ({ game_id: g.game_id, display_name: g.display_name })),
          activeGameId: activeSession?.game_id ?? null,
          activeDisplay: activeSession ? '' : null,
        })
        await dispatch(local)
      } finally {
        setBusyState(false)
      }
    },
    [activeSession, dispatch, pushPlayer, setBusyState],
  )

  const clearSession = useCallback(() => {
    setActiveSession(null)
    setActiveGameId(null)
    saveChatStore({ activeGameId: null })
  }, [])

  const activeGameInfo = activeSession
    ? (games.find((g) => g.game_id === activeSession.game_id) ?? null)
    : null

  return { messages, busy, error, games, activeSession, activeGameInfo, send, moveAction, startSession, notifyCreated, clearSession }
}