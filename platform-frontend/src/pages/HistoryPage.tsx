// 对局记录页（平台模式 · 专业用户元数据视图）。
//
// 「对话即一切」之后，棋盘收进了对话里；这一页承担对话侧刻意不做的**详细
// 元数据**：每局的变体/人数/种子/时长/提示/教学/性格/AI 强度，可筛选、可分页、
// 可展开查看、可导出 CSV。
//
// 口径与实现要点：
// - 标签 / 玩家视角胜负 / 格式化 / 查询串 / CSV 全部来自 `src/history.ts`
//   单一来源（与复盘页共用），页面内不再有内联字典与胜负三元表达式；
// - 筛选在**服务端**做（`/api/history` 的 game_id/result/q/since + offset），
//   所以「加载更多」与「切换筛选」语义清晰：前者追加，后者重置；
// - 旧记录（缺 won / seed / teaching / finished_at）按「缺就不显示」处理，
//   绝不渲染 undefined、也不显示原始 pid。

import { Fragment, useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { listHistory } from '../api/client'
import {
  buildHistoryQuery,
  detailRows,
  difficultyLabel,
  durationLabel,
  formatWhen,
  gameLabel,
  matchResult,
  mergePage,
  movesLabel,
  relativeWhen,
  seatLabel,
  summarize,
  toCsv,
  variantLabel,
  winRateLabel,
  type ResultKind,
} from '../history'
import type { HistoryQueryParams } from '../api/client'
import type { MatchMeta } from '../types'

const PAGE_SIZE = 20
/** 表格列数：时间 / 游戏 / 变体 / 座位 / 难度 / AI 强度 / 步数 / 时长 / 结果。 */
const COLUMN_COUNT = 9

const SINCE_OPTIONS: { value: string; label: string; days: number }[] = [
  { value: '', label: '全部时间', days: 0 },
  { value: '7', label: '近 7 天', days: 7 },
  { value: '30', label: '近 30 天', days: 30 },
  { value: '90', label: '近 90 天', days: 90 },
]

const RESULT_OPTIONS: { value: '' | ResultKind; label: string }[] = [
  { value: '', label: '全部结果' },
  { value: 'win', label: '只看胜局' },
  { value: 'lose', label: '只看负局' },
  { value: 'draw', label: '只看平局' },
]

/** 日期范围取值 → ISO 起始日期（服务端按字符串前缀比较，含当天）。 */
function sinceOf(days: number): string {
  if (!days) return ''
  return new Date(Date.now() - days * 86_400_000).toISOString().slice(0, 10)
}

/** CSV 文件名（本地日期，避免中文文件名在部分浏览器被吞）。 */
function csvFileName(): string {
  return `gavis-对局记录-${new Date().toISOString().slice(0, 10)}.csv`
}

export default function HistoryPage() {
  const [matches, setMatches] = useState<MatchMeta[]>([])
  const [total, setTotal] = useState(0)
  // 已从服务端取回的条数（分页游标）；与 matches.length 分开，因为
  // mergePage 会按 match_id 去重——用去重后的长度当 offset 会漏掉后面的记录。
  const [loaded, setLoaded] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [gameFilter, setGameFilter] = useState('')
  const [resultFilter, setResultFilter] = useState<'' | ResultKind>('')
  // 关键字是唯一「边输边变」的筛选：草稿态（输入框）与已提交态（真正进查询）
  // 分开，避免每敲一个字符就打一次 /api/history。
  const [keyword, setKeyword] = useState('')
  const [keywordDraft, setKeywordDraft] = useState('')
  const [sinceDays, setSinceDays] = useState('')
  // 展开详情的对局（一次只展开一行，避免长列表被撑得看不清）。
  const [expanded, setExpanded] = useState<string | null>(null)
  const navigate = useNavigate()
  // 请求身份：筛选变化后旧请求的响应必须丢弃（慢响应不能覆盖新筛选结果）。
  const requestRef = useRef('')

  const load = useCallback(
    async (offset: number, append: boolean) => {
      const params: HistoryQueryParams = {
        gameId: gameFilter || undefined,
        result: resultFilter || undefined,
        q: keyword.trim() || undefined,
        since: sinceOf(Number(sinceDays)) || undefined,
        limit: PAGE_SIZE,
        offset,
      }
      const key = buildHistoryQuery(params)
      requestRef.current = key
      if (append) setLoadingMore(true)
      else {
        setLoading(true)
        setError(null)
      }
      try {
        const data = await listHistory(params)
        if (requestRef.current !== key) return // 筛选已变，丢弃过期响应
        setMatches((prev) => (append ? mergePage(prev, data.matches) : data.matches))
        setLoaded((prev) => (append ? prev + data.matches.length : data.matches.length))
        setTotal(data.total)
        setHasMore(data.has_more)
        if (!append) setExpanded(null)
      } catch (err) {
        if (requestRef.current !== key) return
        setError((err as Error).message)
      } finally {
        if (requestRef.current === key) {
          setLoading(false)
          setLoadingMore(false)
        }
      }
    },
    [gameFilter, resultFilter, keyword, sinceDays],
  )

  useEffect(() => {
    void load(0, false)
  }, [load])

  function resetFilters() {
    setGameFilter('')
    setResultFilter('')
    setKeyword('')
    setKeywordDraft('')
    setSinceDays('')
  }

  const summary = summarize(matches)
  const filtersActive = Boolean(gameFilter || resultFilter || keyword.trim() || sinceDays)

  function exportCsv() {
    const csv = toCsv(matches)
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }))
    const a = document.createElement('a')
    a.href = url
    a.download = csvFileName()
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div>
      <h1 className="page-title">对局记录</h1>
      <p className="page-sub">
        共 {total} 局 · {summary.wins} 胜 · 胜率 {winRateLabel(total > 0 ? summary.wins / total : 0)} —
        点任意一局展开完整元数据，或进复盘页逐步回看。
      </p>

      <div className="history-filters panel">
        <label>
          游戏
          <select value={gameFilter} onChange={(e) => setGameFilter(e.target.value)}>
            <option value="">全部游戏</option>
            {summary.byGame.map((g) => (
              <option key={g.game_id} value={g.game_id}>
                {g.label}（{g.plays}）
              </option>
            ))}
          </select>
        </label>
        <label>
          结果
          <select value={resultFilter} onChange={(e) => setResultFilter(e.target.value as '' | ResultKind)}>
            {RESULT_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          时间
          <select
            value={sinceDays}
            onChange={(e) => setSinceDays(e.target.value)}
            title="按开始时间过滤（服务端筛选）"
          >
            {SINCE_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          关键字
          <input
            value={keywordDraft}
            placeholder="游戏名 / 座位"
            onChange={(e) => setKeywordDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                setKeyword(keywordDraft)
              }
            }}
          />
        </label>
        <button
          className="btn"
          onClick={() => setKeyword(keywordDraft)}
          disabled={keywordDraft === keyword}
          title="按游戏名 / 座位称呼过滤（也可在下拉框直接切换）"
        >
          查询
        </button>
        <button className="btn" onClick={resetFilters} disabled={!filtersActive}>
          重置
        </button>
        <button className="btn" onClick={exportCsv} disabled={matches.length === 0} title="导出当前筛选结果">
          ⬇ 导出 CSV
        </button>
      </div>

      {error && <div className="error-banner">取对局记录失败：{error}</div>}

      {loading && <div className="panel" style={{ color: 'var(--muted)' }}>加载中…</div>}

      {!loading && !error && (
        matches.length === 0 ? (
          <div className="panel" style={{ color: 'var(--muted)', textAlign: 'center' }}>
            {total === 0 && !filtersActive
              ? '还没有对局记录 — 去对战中心玩一局吧 ⚔️'
              : '当前筛选没有匹配的对局，换个条件试试。'}
          </div>
        ) : (
          <div className="panel">
            <table className="data">
            <thead>
              <tr>
                <th>时间</th>
                <th>游戏</th>
                <th>变体</th>
                <th>你的座位</th>
                <th>难度</th>
                <th>AI 强度</th>
                <th>步数</th>
                <th>时长</th>
                <th>结果</th>
              </tr>
            </thead>
            <tbody>
              {matches.map((m) => {
                const result = matchResult(m)
                const open = expanded === m.match_id
                return (
                  <Fragment key={m.match_id}>
                    <tr className="clickable" onClick={() => setExpanded(open ? null : m.match_id)} title="点击展开完整元数据">
                      <td title={formatWhen(m.started_at)}>{relativeWhen(m.started_at) || formatWhen(m.started_at)}</td>
                      <td>
                        {gameLabel(m.game_id)}
                        {m.custom ? <span className="history-tag">自定义</span> : null}
                      </td>
                      <td>{variantLabel(m) || '—'}</td>
                      <td>{seatLabel(m, m.player_pid)}</td>
                      <td>{difficultyLabel(m)}</td>
                      <td>{m.ai_strength ?? '—'}</td>
                      <td>{movesLabel(m)}</td>
                      <td>{durationLabel(m) || '—'}</td>
                      <td>
                        <span className={`badge ${result.badge}`}>{result.label}</span>
                      </td>
                    </tr>
                    {open ? (
                      <tr className="history-detail-row">
                        <td colSpan={COLUMN_COUNT}>
                          <div className="history-detail">
                            <dl>
                              {detailRows(m).map((row) => (
                                <div key={row.label} className="history-detail-item">
                                  <dt>{row.label}</dt>
                                  <dd>{row.value}</dd>
                                </div>
                              ))}
                            </dl>
                            <div className="history-detail-actions">
                              <button className="btn btn-primary" onClick={() => navigate(`/review/${m.match_id}`)}>
                                ▶ 复盘这一局
                              </button>
                            </div>
                          </div>
                        </td>
                      </tr>
                    ) : null}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
          <div className="history-footer">
            <span style={{ color: 'var(--muted)' }}>
              已显示 {matches.length} / 共 {total} 局
            </span>
            {hasMore && (
              <button className="btn" disabled={loadingMore} onClick={() => void load(loaded, true)}>
                {loadingMore ? '加载中…' : '加载更多'}
              </button>
            )}
          </div>
          </div>
        )
      )}
    </div>
  )
}
