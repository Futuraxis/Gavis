// Chat-first 运行时 — 一句话 → 意图 → 平台动作（对话即一切的核心编排）。
//
// 主路径：POST /api/chat（后端 LLM function calling + schema 校验 + 正则兜底）
// 返回 {intent, text, mood, params}；本 Hook 把意图翻译成平台动作：
// play→开局配置卡、resume/move/restart→对局快照、history/review→战绩/复盘卡、
// create→创建面板、benchmark/learning→进度卡、platform/settings→切回平台界面。
//
// 快速落子红线：棋盘/牌面点击直接走 /match/move，**不经 LLM**——LLM 只处理
// 文本表达的动作（碰/跟注/打五条等），点击永远是即时快路径。
//
// 对话管理与存档：消息流按「可存档消息」（除开场白外全部）增量同步到后端
// ConversationStore（data/conversations/，懒建档、每回合一批）；刷新/重开时
// 先取后端存档恢复，后端不可用再回落 localStorage 镜像——聊天记录不再只
// 活在组件内存里。

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  apiGet,
  apiPost,
  appendConversationMessages,
  chatTurnStream,
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  matchHint,
  updateConversation,
} from '../api/client'
import type {
  ChatMessage,
  ChatTurnResult,
  ConversationMeta,
  GameInfo,
  MatchMeta,
  ReviewReport,
  Snapshot,
} from '../types'
import type { BattleConfig } from '../components/BattleSetup'
import { snapshotChatToMessages } from './snapshotChat'
import { classifyLocal } from './intents'
import { readConversationMirror, writeConversationMirror } from './conversationMirror'
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
  /** 对话存档：当前会话 id（null = 新对话，首条消息懒建档）。 */
  conversationId: string | null
  /** 对话存档列表（供面板展示；后端不可用时为空）。 */
  conversations: ConversationMeta[]
  startNewConversation: () => void
  switchConversation: (convId: string) => Promise<void>
  renameConversation: (convId: string, title: string) => Promise<boolean>
  setConversationArchived: (convId: string, archived: boolean) => Promise<boolean>
  removeConversation: (convId: string) => Promise<boolean>
}

function uid(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

/** 开场白 — 作为第一条消息常驻对话流，不随首次输入消失。 */
const WELCOME_TEXT =
  '你好，我是 Gavis。对局、战绩、创建游戏、评测——一句话就行。\n\n' +
  '试试：\n' +
  '· “玩月亮棋” · “来一局德州扑克” · “继续上一局”\n' +
  '· “看战绩” · “创建游戏” · “打开平台界面”'

function welcomeMessage(): ChatMessage {
  return { id: 'welcome', role: 'agent', text: WELCOME_TEXT, mood: 'happy', ts: Date.now() }
}

/** 可入档消息：除常驻开场白外的全部（开场白每次新会话都会重建，入档无意义）；
 * 流式进行中的 pending 草稿不入档（定稿后才同步）。 */
function persistable(m: ChatMessage): boolean {
  return m.id !== 'welcome' && !m.pending
}

/** 增量同步节流（ms）：一次回合的 player+agent+教练消息合并成一批入档。 */
const SYNC_DEBOUNCE_MS = 400

export function useChatRuntime(): ChatRuntime {
  const [messages, setMessages] = useState<ChatMessage[]>(() => [welcomeMessage()])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [games, setGames] = useState<GameInfo[]>([])
  const [activeSession, setActiveSession] = useState<Snapshot | null>(null)
  const [, setActiveGameId] = useState<string | null>(() => loadChatStore().activeGameId)
  const [conversationId, setConversationId] = useState<string | null>(() => loadChatStore().conversationId)
  const [conversations, setConversations] = useState<ConversationMeta[]>([])
  const busyRef = useRef(false)
  // 流式草稿目标：send 期间 pushAgent 应原地定稿该消息（而非追加新消息）。
  // dispatch 的各个分支无需改动 —— 定稿规则收敛在 pushAgent 一处。
  const draftTargetRef = useRef<string | null>(null)
  const gamesRef = useRef<GameInfo[]>([])
  // 镜像 messages 的 ref（useEffect 同步），send 里取「已渲染完的消息」当对话历史；
  // 刚 push 的当前句不在其中，它本来就是 /api/chat 的 text 参数，不会重复。
  const messagesRef = useRef<ChatMessage[]>([])
  // 对话存档增量同步状态：
  //   token — 会话切换纪元（在途请求完成后据此丢弃过期回写）
  //   convId / synced — 当前会话 id 与已入档消息数（按 persistable 过滤后的计数）
  //   inflight / dirty — 单飞闸门：同一时刻至多一个同步请求，期间的新消息置脏重排
  const syncRef = useRef({
    token: 0,
    convId: loadChatStore().conversationId,
    synced: 0,
    inflight: false,
    dirty: false,
  })

  useEffect(() => {
    gamesRef.current = games
  }, [games])

  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  // 初始数据：游戏目录 + 恢复上次对局 + 恢复对话存档。
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
          _drainSnapshot(d.session)
          if (d.session.over) setActiveGameId(null)
        })
        .catch(() => setActiveGameId(null))
    }
    // 对话存档：后端为唯一事实来源；失败（旧服务端/断连）回落 localStorage 镜像。
    void refreshConversations()
    const storedConvId = loadChatStore().conversationId
    if (storedConvId) {
      getConversation(storedConvId)
        .then((conv) => {
          adoptConversation(conv.conv_id, conv.messages)
        })
        .catch(() => {
          const mirrored = readConversationMirror(storedConvId)
          if (mirrored) adoptConversation(storedConvId, mirrored)
        })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── 对话存档：恢复 / 增量同步 / 镜像 ─────────────────────────

  /** 采用一段消息流作为当前会话（恢复/切换/新开共用）：重置同步纪元与计数。 */
  const adoptConversation = useCallback((convId: string | null, msgs: ChatMessage[]) => {
    const st = syncRef.current
    st.token += 1 // 使在途同步请求的回写失效（其结果属于旧会话）
    st.convId = convId
    st.synced = msgs.filter(persistable).length
    st.inflight = false
    st.dirty = false
    setConversationId(convId)
    saveChatStore({ conversationId: convId })
    setMessages(msgs.length > 0 ? msgs : [welcomeMessage()])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const refreshConversations = useCallback(async (): Promise<void> => {
    try {
      const d = await listConversations()
      setConversations(d.conversations ?? [])
    } catch {
      // 旧服务端/断连：面板列表拿不到就不展示，聊天与镜像兜底不受影响。
    }
  }, [])

  /** 把「未入档尾部」同步到后端（懒建档 → 增量 append），失败静默保持本地态。 */
  const flushPending = useCallback(async (): Promise<void> => {
    const st = syncRef.current
    if (st.inflight) {
      st.dirty = true
      return
    }
    const persistableMsgs = messagesRef.current.filter(persistable)
    if (st.synced >= persistableMsgs.length) return
    const tail = persistableMsgs.slice(st.synced)
    const token = st.token
    st.inflight = true
    try {
      if (st.convId) {
        const meta = await appendConversationMessages(st.convId, tail)
        if (st.token !== token) return // 会话已切换：本批已落在旧会话名下，计数不动
        st.synced += tail.length
        setConversations((prev) => [meta, ...prev.filter((c) => c.conv_id !== meta.conv_id)])
      } else {
        const conv = await createConversation({ messages: tail })
        if (st.token !== token) return
        st.convId = conv.conv_id
        st.synced += tail.length
        setConversationId(conv.conv_id)
        saveChatStore({ conversationId: conv.conv_id })
        const { messages: _strip, ...meta } = conv
        setConversations((prev) => [meta, ...prev.filter((c) => c.conv_id !== meta.conv_id)])
      }
    } catch {
      // 后端不可用：synced 未推进，下一次消息变化自动重试；镜像仍每回合在写。
    } finally {
      st.inflight = false
      if (st.dirty) {
        st.dirty = false
        void flushPending()
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 消息变化 →（防抖）批量入档；同帧多条（player+agent+教练讲评）合成一批。
  useEffect(() => {
    const timer = setTimeout(() => void flushPending(), SYNC_DEBOUNCE_MS)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages])

  // 消息变化 → localStorage 镜像（即时，不防抖：断电/关页的最坏情况也有底）。
  useEffect(() => {
    writeConversationMirror(syncRef.current.convId, messages)
  }, [messages])

  const pushPlayer = useCallback((text: string) => {
    setMessages((prev) => [...prev, { id: uid(), role: 'player', text, ts: Date.now() }])
  }, [])

  const pushAgent = useCallback(
    (text: string, mood?: ChatMessage['mood'], intent?: ChatMessage['intent'], params?: Record<string, unknown>) => {
      const draftId = draftTargetRef.current
      if (draftId) {
        // 流式定稿：正文优先保留已流出的增量；否则采用调用方文案（兜底/错误提示）。
        setMessages((prev) =>
          prev.map((m) => {
            if (m.id !== draftId) return m
            const finalText = m.text && m.text.trim() ? m.text : text
            return { ...m, text: finalText, mood, intent, params, pending: false }
          }),
        )
        return
      }
      setMessages((prev) => [...prev, { id: uid(), role: 'agent', text, mood, ts: Date.now(), intent, params }])
    },
    [],
  )

  const setBusyState = useCallback((v: boolean) => {
    busyRef.current = v
    setBusy(v)
  }, [])

  /** 把后端快照里待投递的陪伴/教练消息（chat 增量）落进对话流。 */
  const _drainSnapshot = useCallback(
    (snap: Snapshot) => {
      const msgs = snapshotChatToMessages(snap)
      if (msgs.length > 0) {
        setMessages((prev) => [...prev, ...msgs])
      }
    },
    [],
  )

  const refreshSession = useCallback(async (gameId: string): Promise<void> => {
    const d = await apiPost<{ session: Snapshot }>('/match/state', { game_id: gameId })
    setActiveSession(d.session)
    _drainSnapshot(d.session)
    if (d.session.over) {
      setActiveGameId(null)
      saveChatStore({ activeGameId: null })
    } else {
      setActiveGameId(gameId)
      saveChatStore({ activeGameId: gameId })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
          theme: config.theme,
          player_count: config.playerCount,
          persona: config.persona,
          hint_level: config.hintLevel,
          pacing: config.pacing,
          adaptive: config.adaptive,
          teaching: config.teaching,
        })
        setActiveSession(data.session)
        _drainSnapshot(data.session)
        setActiveGameId(gameId)
        saveChatStore({ activeGameId: gameId })
        const name = gamesRef.current.find((g) => g.game_id === gameId)?.display_name ?? gameId
        const teach = data.session.teaching ? '教学局：教练看得到你的牌，边打边讲。' : ''
        pushAgent(`对局已开始：${name} 🎮 ${teach}轮到你了就下，也可以随时问我“这步怎么走”。`, 'happy')
      } catch (err) {
        setError((err as Error).message)
        pushAgent(`开局失败：${(err as Error).message}`, 'sorry')
      } finally {
        setBusyState(false)
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        _drainSnapshot(data.session) // 教练讲评（teach_move）与导读（teach_turn）
        if (!data.session.over) setActiveGameId(data.session.game_id)
      } catch (err) {
        setError((err as Error).message)
        pushAgent(`这一步没走成：${(err as Error).message}`, 'sorry')
      } finally {
        setBusyState(false)
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
              _drainSnapshot(data.session)
              if (!data.session.over) setActiveGameId(data.session.game_id)
            } catch (err) {
              pushAgent(`这一步没走成：${(err as Error).message}`, 'sorry')
            }
          }
          break
        case 'hint': {
          // 后端已算好机械提示（ask_hint 信息工具，params.hint 携带）时直接展示，
          // 不再二次请求 /match/hint；仅正则兜底路径（无 params.hint）才回源取数。
          const backendHint = params.hint as { hint?: string } | undefined
          if (backendHint) {
            pushAgent(text || backendHint.hint || '这一步的思路是…', mood, 'hint', {})
            break
          }
          if (!activeSession) {
            pushAgent('现在没有对局，先来一局吧。', 'neutral')
            break
          }
          try {
            const level = String(params.level ?? 'direction') as Parameters<typeof matchHint>[1]
            const hint = await matchHint(activeSession.game_id, level)
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
            teaching: activeSession?.teaching ?? false,
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
          // 后端已带 report（get_match_review 信息工具，含 LLM 讲解文本）时直接用；
          // 正则兜底路径才自己取最近一局 + /review/<id>。
          const backendReport = params.report as ReviewReport | undefined
          const backendMatchId = params.match_id ? String(params.match_id) : ''
          if (backendReport && backendMatchId) {
            pushAgent(text, mood, 'review', { report: backendReport, match_id: backendMatchId })
            break
          }
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
          // 知识回答（“X 是什么”）：后端/本地兜底都在 params 里带
          // game_id + chips（如“玩月亮棋”）—— 透传给消息渲染快捷动作
          // （复用 clarify 的 Chips 组件），不再丢弃。
          pushAgent(text, mood, 'chat', {
            game_id: params.game_id,
            chips: Array.isArray(params.chips) ? params.chips : [],
          })
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
      // 流式草稿：先落一条 pending agent 消息，增量原地更新（正文 + 思维链），
      // 最终由 dispatch 内的 pushAgent 命中 draftTargetRef 原地定稿 ——
      // 不产生第二条消息，也不丢已流出的亲笔文本。
      const draftId = uid()
      draftTargetRef.current = draftId
      setMessages((prev) => [
        ...prev,
        { id: draftId, role: 'agent', text: '', pending: true, mood: 'neutral', ts: Date.now() },
      ])
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
        const result = await chatTurnStream(trimmed, activeSession?.game_id, history, {
          onText: (delta) =>
            setMessages((prev) => prev.map((m) => (m.id === draftId ? { ...m, text: m.text + delta } : m))),
          onReasoning: (delta) =>
            setMessages((prev) =>
              prev.map((m) => (m.id === draftId ? { ...m, reasoning: (m.reasoning ?? '') + delta } : m)),
            ),
        })
        await dispatch(result)
      } catch (err) {
        // 流中断/后端不可用：草稿已流出内容则保留定稿为 chat 消息；
        // 完全没收到任何内容则移除草稿、走本地正则兜底（旧行为不变）。
        const draft = messagesRef.current.find((m) => m.id === draftId)
        const hasStreamed = !!draft && (draft.text.trim().length > 0 || (draft.reasoning ?? '').trim().length > 0)
        if (hasStreamed) {
          pushAgent(`（回复中断：${(err as Error).message}）`, 'sorry')
        } else {
          draftTargetRef.current = null
          setMessages((prev) => prev.filter((m) => m.id !== draftId))
          // description / aliases 供 WHAT_IS 知识回答与短名匹配（与后端对齐）。
          const local = classifyLocal(trimmed, {
            games: gamesRef.current.map((g) => ({
              game_id: g.game_id,
              display_name: g.display_name,
              description: g.description,
              aliases: g.aliases,
            })),
            activeGameId: activeSession?.game_id ?? null,
            activeDisplay: activeSession ? '' : null,
          })
          await dispatch(local)
        }
      } finally {
        draftTargetRef.current = null
        setBusyState(false)
      }
    },
    [activeSession, dispatch, pushPlayer, pushAgent, setBusyState],
  )

  const clearSession = useCallback(() => {
    setActiveSession(null)
    setActiveGameId(null)
    saveChatStore({ activeGameId: null })
  }, [])

  // ── 对话存档管理操作（面板调用） ─────────────────────────────

  const startNewConversation = useCallback((): void => {
    // 新开对话不影响进行中的对局（activeGameId 保留——对局与对话独立）。
    adoptConversation(null, [welcomeMessage()])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const switchConversation = useCallback(
    async (convId: string): Promise<void> => {
      if (convId === syncRef.current.convId) return
      try {
        const conv = await getConversation(convId)
        adoptConversation(conv.conv_id, conv.messages)
      } catch (err) {
        setError(`读取对话失败：${(err as Error).message}`)
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    },
    [],
  )

  const renameConversation = useCallback(
    async (convId: string, title: string): Promise<boolean> => {
      try {
        const meta = await updateConversation(convId, { title })
        setConversations((prev) => prev.map((c) => (c.conv_id === convId ? meta : c)))
        return true
      } catch (err) {
        setError(`重命名失败：${(err as Error).message}`)
        return false
      }
    },
    [],
  )

  const setConversationArchived = useCallback(
    async (convId: string, archived: boolean): Promise<boolean> => {
      try {
        const meta = await updateConversation(convId, { archived })
        setConversations((prev) => prev.map((c) => (c.conv_id === convId ? meta : c)))
        return true
      } catch (err) {
        setError(archived ? `归档失败：${(err as Error).message}` : `取消归档失败：${(err as Error).message}`)
        return false
      }
    },
    [],
  )

  const removeConversation = useCallback(
    async (convId: string): Promise<boolean> => {
      try {
        await deleteConversation(convId)
        setConversations((prev) => prev.filter((c) => c.conv_id !== convId))
        if (syncRef.current.convId === convId) {
          // 删的是当前会话 → 回到全新对话（开场白重置，不再引用已删档案）。
          adoptConversation(null, [welcomeMessage()])
        }
        return true
      } catch (err) {
        setError(`删除失败：${(err as Error).message}`)
        return false
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    },
    [],
  )

  const activeGameInfo = activeSession
    ? (games.find((g) => g.game_id === activeSession.game_id) ?? null)
    : null

  return {
    messages,
    busy,
    error,
    games,
    activeSession,
    activeGameInfo,
    send,
    moveAction,
    startSession,
    notifyCreated,
    clearSession,
    conversationId,
    conversations,
    startNewConversation,
    switchConversation,
    renameConversation,
    setConversationArchived,
    removeConversation,
  }
}