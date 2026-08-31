import { useState } from 'react'
import type { GameInfo, HintLevel, Pacing, PersonaKey } from '../types'

export interface BattleConfig {
  playerPid: string
  difficulty: string
  theme?: string
  playerCount: number
  persona: PersonaKey
  hintLevel: HintLevel
  pacing: Pacing
  adaptive: boolean
  teaching: boolean
}

interface Props {
  game: GameInfo
  busy: boolean
  error: string | null
  onStart: (config: BattleConfig) => void
}

// 座位称呼统一用 game.seat_names（后端按族下发）；本表仅留"随机"选项文案。
const SEAT_LABELS: Record<string, string> = {
  random: '随机 🎲',
}

const DIFFICULTY_LABELS: Record<string, string> = {
  easy: '简单 😊',
  normal: '普通 🤔',
  hard: '困难 😈',
}

const PERSONA_OPTIONS: { value: PersonaKey; label: string }[] = [
  { value: 'gentle', label: '温柔陪伴 🌸' },
  { value: 'teacher', label: '认真教学 📖' },
  { value: 'banter', label: '轻松吐槽 😄' },
  { value: 'cold', label: '高冷竞技 🗿' },
]

const HINT_OPTIONS: { value: HintLevel; label: string }[] = [
  { value: 'off', label: '关闭' },
  { value: 'direction', label: '方向提示' },
  { value: 'specific', label: '具体建议' },
  { value: 'demo', label: '演示' },
]

const PACING_OPTIONS: { value: Pacing; label: string }[] = [
  { value: 'fast', label: '快棋 ⚡' },
  { value: 'standard', label: '标准 🕐' },
  { value: 'slow', label: '慢棋 🐢' },
]

// 谁是卧底主题（词对类别）；难度档决定词对相似度，主题决定词对类别。
const THEME_LABELS: Record<string, string> = {
  fruit: '水果 🍎',
  food: '美食 🍔',
  animal: '动物 🐾',
  object: '物品 📦',
  place: '地点 🏛️',
  plant: '植物 🌿',
}

const RULES_SUMMARY: Record<string, string> = {
  moon_chess: '3×3 棋盘，双方轮流落子，三子连珠即胜；棋盘放满后最旧的棋子会被挤出。',
  stochastic_gomoku: '9×9 棋盘，五子连珠即胜；每次落子后，棋子有 50% 概率被随机抹去。',
  texas_holdem: '双人德州扑克：每人两张底牌，依次翻牌/转牌/河牌，比五张最大牌型。',
  mahjong_guangdong: '四人广东鸡胡：吃碰杠、自摸荣和、清一色等番种。',
  mahjong_hongzhong: '红中万能牌：红中可代任意牌凑搭子，其余规则同鸡胡。',
  mahjong_blood: '血流成河：胡牌后不退场继续摸打（不能重复胡），可多次胡牌累计番分，直到三家胡牌或牌墙摸空。',
  mahjong_sichuan: '四人四川麻将（血战到底）：108 张无字牌，缺一门才能胡，禁吃，胡牌后胡家退场，直到三家胡牌或牌墙摸空。',
  mahjong_changsha: '四人长沙麻将：258将为将的小胡 + 大胡（碰碰胡/清一色等）乱将豁免。',
  mahjong_taiwan: '四人台湾麻将（16 张）：5 副露 + 将成胡，呖咕呖咕（八对半）可胡。',
  undercover: '谁是卧底：平民同词、卧底近义词、白板无词，轮流一句话描述后投票，票最多者出局（平票无人出局）；默认 8 人（1 卧底 + 1 白板 + 6 平民），可 4-12 人。',
}

export default function BattleSetup({ game, busy, error, onStart }: Props) {
  const [playerPid, setPlayerPid] = useState('random')
  const [difficulty, setDifficulty] = useState('normal')
  const [playerCount, setPlayerCount] = useState(game.player_counts[0] ?? 2)
  const [theme, setTheme] = useState(game.variant_themes?.[0] ?? 'fruit')
  const [persona, setPersona] = useState<PersonaKey>('gentle')
  const [hintLevel, setHintLevel] = useState<HintLevel>('off')
  const [pacing, setPacing] = useState<Pacing>('standard')
  const [adaptive, setAdaptive] = useState(true)
  const [teaching, setTeaching] = useState(false)
  const [showRules, setShowRules] = useState(false)

  // 座位按人数取前 N 个（麻将默认 4 人 → 显 p0-p3）——避免选到人数外的座位造成死局。
  const seatOptions = game.seat_options.slice(0, playerCount)

  function changePlayerCount(n: number) {
    setPlayerCount(n)
    const valid = game.seat_options.slice(0, n)
    if (playerPid !== 'random' && !valid.includes(playerPid)) setPlayerPid('random')
  }

  function toggleTeaching(next: boolean) {
    setTeaching(next)
    // 教学对局默认搭配「认真教学」人格（仍可手动改选其它性格）。
    if (next && persona === 'gentle') setPersona('teacher')
  }

  return (
    <div className="panel" style={{ maxWidth: 560 }}>
      <h1 className="page-title">{game.display_name}</h1>
      <p className="page-sub">{game.description}</p>
      {error && <div className="error-banner">{error}</div>}

      <div className="form-row">
        <button className="btn" onClick={() => setShowRules((v) => !v)}>
          {showRules ? '收起规则速览' : '📜 规则速览'}
        </button>
      </div>
      {showRules && (
        <div className="rules-panel">
          {RULES_SUMMARY[game.game_id] ?? `${game.display_name} 的规则摘要（后端尚未回填，此处为占位文本）。`}
        </div>
      )}

      {game.player_counts.length > 1 && (
        <div className="form-row">
          <label>人数:</label>
          <select value={playerCount} onChange={(e) => changePlayerCount(Number(e.target.value))}>
            {game.player_counts.map((n) => (
              <option key={n} value={n}>
                {n} 人
              </option>
            ))}
          </select>
        </div>
      )}
      {game.variant_themes && game.variant_themes.length > 1 && (
        <div className="form-row">
          <label>主题:</label>
          <select value={theme} onChange={(e) => setTheme(e.target.value)}>
            {game.variant_themes.map((t) => (
              <option key={t} value={t}>
                {THEME_LABELS[t] ?? t}
              </option>
            ))}
          </select>
        </div>
      )}
      <div className="form-row">
        <label>{game.seat_label}:</label>
        <select value={playerPid} onChange={(e) => setPlayerPid(e.target.value)}>
          <option value="random">{SEAT_LABELS.random}</option>
          {seatOptions.map((pid) => (
            <option key={pid} value={pid}>
              {game.seat_names[pid] ?? pid}
            </option>
          ))}
        </select>
      </div>
      <div className="form-row">
        <label>难度:</label>
        <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
          {game.difficulties.map((d) => (
            <option key={d} value={d}>
              {DIFFICULTY_LABELS[d] ?? d}
            </option>
          ))}
        </select>
      </div>
      {game.kind === 'mahjong' && !game.custom && (
        <div className="form-row" style={{ marginTop: -6 }}>
          <span style={{ color: 'var(--muted)', fontSize: 13 }}>
            ℹ️ 麻将 AI 当前为固定强度的启发式策略，三档难度暂无实际差异
          </span>
        </div>
      )}
      <div className="form-row">
        <label>性格:</label>
        <select value={persona} onChange={(e) => setPersona(e.target.value as PersonaKey)}>
          {PERSONA_OPTIONS.map((p) => (
            <option key={p.value} value={p.value}>
              {p.label}
            </option>
          ))}
        </select>
      </div>
      <div className="form-row">
        <label>提示:</label>
        <select value={hintLevel} onChange={(e) => setHintLevel(e.target.value as HintLevel)}>
          {HINT_OPTIONS.map((h) => (
            <option key={h.value} value={h.value}>
              {h.label}
            </option>
          ))}
        </select>
      </div>
      <div className="form-row">
        <label>节奏:</label>
        <select value={pacing} onChange={(e) => setPacing(e.target.value as Pacing)}>
          {PACING_OPTIONS.map((p) => (
            <option key={p.value} value={p.value}>
              {p.label}
            </option>
          ))}
        </select>
      </div>
      <div className="form-row">
        <label>自适应难度:</label>
        <label
          style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}
          title="AI 强度按你近 10 局胜率自动升降：连胜变难、连败变易。下方难度档位只作初始锚点。"
        >
          <input type="checkbox" checked={adaptive} onChange={(e) => setAdaptive(e.target.checked)} />
          <span>{adaptive ? '开启 ⚙' : '关闭'}</span>
        </label>
      </div>
      {adaptive && (
        <div className="form-row" style={{ marginTop: -6 }}>
          <span style={{ color: 'var(--muted)', fontSize: 13 }}>
            AI 强度将按你近 10 局胜率自动调整（当前难度档位只作初始锚点；麻将族暂为固定启发式）
          </span>
        </div>
      )}
      <div className="form-row">
        <label>教学对局:</label>
        <label
          style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}
          title="教练 Agent 能看到你的牌并推理：每步走完会对照参考动作讲评，轮到你时会读牌导读。它看不到 AI/对手的牌。"
        >
          <input type="checkbox" checked={teaching} onChange={(e) => toggleTeaching(e.target.checked)} />
          <span>{teaching ? '开启 📖' : '关闭'}</span>
          <span style={{ color: 'var(--muted)', fontSize: 13 }}>
            教练看你的牌带你打、走完点评
          </span>
        </label>
      </div>
      <button
        className="btn btn-primary"
        disabled={busy}
        onClick={() => onStart({ playerPid, difficulty, theme, playerCount, persona, hintLevel, pacing, adaptive, teaching })}
      >
        {busy ? '加载中…' : '开始对局'}
      </button>
    </div>
  )
}
