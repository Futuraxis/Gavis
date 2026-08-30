import { useEffect, useState } from 'react'
import {
  ApiError,
  getLlmConfig,
  saveLlmConfig,
  testLlmConnection,
  type LlmConfigInfo,
} from '../api/client'

const SOURCE_LABELS: Record<string, string> = {
  platform: '平台配置',
  env: '环境变量',
  default: '内置默认',
}

const APPLIES_TO = [
  '平台聊天 / Agent 对话（/api/chat、/api/agent/say）',
  '自定义游戏规则翻译（/api/rules/translate、创建游戏）',
  '社交类 AI 求解器（狼人杀 / 谁是卧底的 AI 座位）',
]

export default function LlmConfigPage() {
  const [info, setInfo] = useState<LlmConfigInfo | null>(null)
  const [baseUrl, setBaseUrl] = useState('')
  const [model, setModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [apiKeyTouched, setApiKeyTouched] = useState(false)
  const [busy, setBusy] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getLlmConfig()
      .then((d) => {
        setInfo(d.config)
        setBaseUrl(d.config.base_url)
        setModel(d.config.model)
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)))
  }, [])

  function showApiError(e: unknown) {
    setError(e instanceof ApiError ? e.message : String(e))
  }

  async function testConnection() {
    setBusy(true)
    setToast(null)
    setError(null)
    try {
      const patch: { base_url?: string; api_key?: string } = { base_url: baseUrl }
      if (apiKeyTouched) patch.api_key = apiKey
      const r = await testLlmConnection(patch)
      setToast(
        r.reachable
          ? `✅ 连接成功：${r.base_url}/v1/models`
          : `❌ 连接失败：${r.error || '端点不可达'}`,
      )
    } catch (e) {
      showApiError(e)
    } finally {
      setBusy(false)
    }
  }

  async function save() {
    setBusy(true)
    setToast(null)
    setError(null)
    try {
      const patch: { base_url?: string; model?: string; api_key?: string } = {
        base_url: baseUrl,
        model,
      }
      if (apiKeyTouched) patch.api_key = apiKey
      const d = await saveLlmConfig(patch)
      setInfo(d.config)
      setApiKey('')
      setApiKeyTouched(false)
      setToast('已保存 ✓ 聊天 / 翻译 / 社交 AI 已切换到新配置')
    } catch (e) {
      showApiError(e)
    } finally {
      setBusy(false)
    }
  }

  async function restoreDefaults() {
    setBusy(true)
    setToast(null)
    setError(null)
    try {
      const d = await saveLlmConfig({ base_url: '', model: '', api_key: '' })
      setInfo(d.config)
      setBaseUrl('')
      setModel('')
      setApiKey('')
      setApiKeyTouched(false)
      setToast('已恢复默认 ✓ 回退到环境变量 / 内置默认')
    } catch (e) {
      showApiError(e)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <h1 className="page-title">LLM 配置</h1>
      <p className="page-sub">API 端点与模型 — 无需改代码，保存即全局生效</p>
      {toast && <div className="success-banner">{toast}</div>}
      {error && <div className="error-banner">{error}</div>}

      <section className="panel" style={{ marginBottom: 18 }}>
        <h3 style={{ marginBottom: 14 }}>端点与模型</h3>
        <div className="form-row">
          <label>API 端点</label>
          <input
            className="create-input"
            style={{ maxWidth: 420 }}
            placeholder="例如 http://127.0.0.1:11434 或 https://api.deepseek.com"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
          />
        </div>
        <div className="form-row">
          <label>模型</label>
          <input
            className="create-input"
            style={{ maxWidth: 420 }}
            placeholder="例如 qwen3:8b / deepseek-chat"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          />
        </div>
        <div className="form-row">
          <label>API 密钥</label>
          <input
            className="create-input"
            style={{ maxWidth: 420 }}
            type="password"
            placeholder={info?.has_api_key ? '已设置（留空 = 保持不变）' : '本地端点可留空'}
            value={apiKey}
            onChange={(e) => {
              setApiKey(e.target.value)
              setApiKeyTouched(true)
            }}
          />
          {info?.has_api_key && (
            <span className="badge accent">密钥已设置</span>
          )}
        </div>
        <p style={{ color: 'var(--muted)', fontSize: 13 }}>
          请求路径为 <code>{`{端点}/v1/chat/completions`}</code>（OpenAI 兼容协议，支持本地
          Ollama / vLLM / DeepSeek / Qwen 等）。
        </p>
      </section>

      <section className="panel" style={{ marginBottom: 18 }}>
        <h3 style={{ marginBottom: 14 }}>当前生效</h3>
        <div className="form-row">
          <label>端点</label>
          <code>{info?.effective_base_url ?? '—'}</code>
        </div>
        <div className="form-row">
          <label>模型</label>
          <code>{info?.effective_model ?? '—'}</code>
        </div>
        <div className="form-row">
          <label>来源</label>
          <span className="badge accent">{SOURCE_LABELS[info?.source ?? 'default']}</span>
          {info?.available === true && <span className="badge win">端点可达</span>}
          {info?.available === false && <span className="badge lose">端点不可达</span>}
        </div>
        <p style={{ color: 'var(--muted)', fontSize: 13, marginTop: 6 }}>
          优先级：平台配置 &gt; 环境变量（LLM_BASE_URL / LLM_MODEL / LLM_API_KEY）&gt;
          内置默认（127.0.0.1:11434 / qwen3:8b）。环境变量在启动平台服务前设置即可。
        </p>
      </section>

      <section className="panel" style={{ marginBottom: 18 }}>
        <h3 style={{ marginBottom: 14 }}>生效范围</h3>
        <ul style={{ color: 'var(--muted)', fontSize: 14, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 6 }}>
          {APPLIES_TO.map((item) => (
            <li key={item}>• {item}</li>
          ))}
        </ul>
        <p style={{ color: 'var(--muted)', fontSize: 13, marginTop: 8 }}>
          配置持久化在 <code>data/llm_config.json</code>（密钥明文存本地，不回显）。
        </p>
      </section>

      <div className="form-row">
        <button className="btn btn-primary" disabled={busy} onClick={save}>
          {busy ? '保存中…' : '保存配置'}
        </button>
        <button className="btn" disabled={busy} onClick={testConnection}>
          测试连接
        </button>
        <button className="btn btn-danger" disabled={busy} onClick={restoreDefaults}>
          恢复默认
        </button>
      </div>
    </div>
  )
}