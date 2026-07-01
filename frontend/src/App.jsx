import { useEffect, useState } from 'react'
import MarkdownMessage from './MarkdownMessage'
import WarningBlock from './WarningBlock'
import SearchParamsPanel from './SearchParamsPanel'
import GlossaryPanel, { glossaryFromMessages } from './GlossaryPanel'
import TokenUsageBar from './TokenUsageBar'

function formatDuration(ms) {
  if (ms == null || ms < 0) return null
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function ResponseTiming({ timing, clientMs }) {
  const server = formatDuration(timing?.total_ms)
  const retrieval = timing?.retrieval_ms
  const llm = timing?.llm_ms
  const client = formatDuration(clientMs)

  if (!server && !client) return null

  const parts = []
  if (server) {
    let detail = `응답 ${server}`
    if (retrieval != null && llm != null && llm > 0) {
      detail += ` (검색 ${formatDuration(retrieval)}, LLM ${formatDuration(llm)})`
    } else if (retrieval != null) {
      detail += ` (검색 ${formatDuration(retrieval)})`
    }
    parts.push(detail)
  }
  if (client && clientMs != null && Math.abs(clientMs - (timing?.total_ms ?? 0)) > 50) {
    parts.push(`전체 ${client}`)
  }

  return <div className="msg-timing">{parts.join(' · ')}</div>
}

function MessageBody({ role, text, citations, warnings, streaming }) {
  const label = role === 'user' ? 'You' : 'Assistant'
  const isProblematic = warnings?.length > 0

  return (
    <div className={`msg-body ${isProblematic ? 'msg-problematic' : ''}`}>
      <b>{label}:</b>
      {role === 'assistant' ? (
        <div className="msg-text md-content">
          <WarningBlock warnings={warnings} />
          {text ? (
            <MarkdownMessage text={text} citations={citations} />
          ) : streaming ? (
            <span className="stream-placeholder">Searching and drafting…</span>
          ) : null}
          {streaming && text ? <span className="stream-cursor" aria-hidden>▍</span> : null}
        </div>
      ) : (
        <div className="msg-text">{text}</div>
      )}
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [params, setParams] = useState(null)
  const [defaults, setDefaults] = useState(null)
  const [glossary, setGlossary] = useState([])
  const [lastRetrieval, setLastRetrieval] = useState(null)
  const [lastUsage, setLastUsage] = useState({ input_tokens: 0, output_tokens: 0, total_tokens: 0 })
  const [lastSaved, setLastSaved] = useState({ total: 0, actions: [] })
  const [sessionTokens, setSessionTokens] = useState(0)
  const [sessionSaved, setSessionSaved] = useState(0)

  useEffect(() => {
    setGlossary(glossaryFromMessages(messages))
  }, [messages])

  useEffect(() => {
    fetch('/api/config')
      .then((r) => r.json())
      .then((data) => {
        setDefaults(data.defaults)
        setParams(data.defaults)
      })
      .catch(() => {
        const fallback = {
          top_k: 5,
          max_distance: 0.5,
          fallback_max_distance: 0.65,
          max_context_chars: 4500,
          max_tokens: 768,
        }
        setDefaults(fallback)
        setParams(fallback)
      })
  }, [])

  function updateParam(key, value) {
    setParams((p) => ({ ...p, [key]: value }))
  }

  function resetParams() {
    setParams(defaults)
  }

  async function send(e) {
    e.preventDefault()
    const question = input.trim()
    if (!question || loading || !params) return

    const history = messages.map(({ role, text }) => ({ role, text }))
    setMessages((m) => [...m, { role: 'user', text: question }])
    setInput('')
    setError('')
    setLoading(true)

    const assistantIdx = messages.length + 1
    setMessages((m) => [
      ...m,
      {
        role: 'assistant',
        text: '',
        citations: [],
        warnings: [],
        streaming: true,
      },
    ])

    const t0 = performance.now()

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question, history, params, stream: true }),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.error || 'Request failed')
      }

      const reader = res.body?.getReader()
      if (!reader) {
        throw new Error('Streaming not supported')
      }

      const decoder = new TextDecoder()
      let buffer = ''
      let doneData = null

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.trim()) continue
          const event = JSON.parse(line)

          if (event.type === 'retrieval') {
            const r = event.data
            setLastRetrieval({
              search_query: r.search_query,
              used_fallback: r.used_fallback,
              retrieval: r.retrieval,
            })
          } else if (event.type === 'delta' && event.text) {
            setMessages((m) => {
              const next = [...m]
              const msg = next[assistantIdx]
              if (!msg || msg.role !== 'assistant') return m
              next[assistantIdx] = { ...msg, text: msg.text + event.text }
              return next
            })
          } else if (event.type === 'done') {
            doneData = event.data
          } else if (event.type === 'error') {
            throw new Error(event.error || 'Stream failed')
          }
        }
      }

      if (!doneData) {
        throw new Error('Incomplete stream response')
      }

      const clientMs = Math.round(performance.now() - t0)
      const usage = doneData.usage || { input_tokens: 0, output_tokens: 0, total_tokens: 0 }
      const saved = doneData.tokens_saved || { total: 0, actions: [] }
      setLastUsage(usage)
      setLastSaved(saved)
      setSessionTokens((t) => t + (usage.total_tokens || 0))
      setSessionSaved((t) => t + (saved.total || 0))

      setLastRetrieval({
        search_query: doneData.search_query,
        used_fallback: doneData.used_fallback,
        retrieval: doneData.retrieval,
      })

      setMessages((m) => {
        const next = [...m]
        next[assistantIdx] = {
          role: 'assistant',
          text: doneData.reply,
          citations: doneData.citations || [],
          glossary: doneData.glossary || [],
          warnings: doneData.warnings || [],
          retrieved: doneData.retrieved,
          timing: doneData.timing,
          clientMs,
          streaming: false,
        }
        return next
      })
    } catch (err) {
      setMessages((m) => m.filter((_, i) => i !== assistantIdx))
      setError(err.message || 'Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  if (!params) {
    return <div className="app-loading">Loading settings…</div>
  }

  return (
    <div className="layout">
      <aside className="sidebar sidebar-left">
        <GlossaryPanel entries={glossary} />
      </aside>

      <main className="main">
        <header className="header">
          <h1>RAG Chat</h1>
          <p className="subtitle">
            Answers grounded in your indexed documents with citations and adjustable retrieval settings.
          </p>
        </header>

        <div className="messages">
          {messages.length === 0 && (
            <p className="empty-hint">
              Ask a question about your indexed documents.
            </p>
          )}
          {messages.map((m, i) => (
            <div
              key={i}
              className={`msg msg-${m.role}${m.warnings?.length ? ' msg-has-warning' : ''}`}
            >
              <MessageBody
                role={m.role}
                text={m.text}
                citations={m.citations}
                warnings={m.warnings}
                streaming={m.streaming}
              />
              {m.role === 'assistant' && (
                <ResponseTiming timing={m.timing} clientMs={m.clientMs} />
              )}
              {m.role === 'assistant' && m.citations?.length > 0 && (
                <div className="sources">
                  <div className="sources-label">Sources</div>
                  {m.citations.map((c) => (
                    <details key={c.n} className="source">
                      <summary>
                        [{c.n}] {c.source} · chunk {c.chunk_index}
                        {c.distance != null && (
                          <span className="cite-dist"> · d={c.distance}</span>
                        )}
                      </summary>
                      <p className="excerpt">{c.excerpt}</p>
                    </details>
                  ))}
                </div>
              )}
            </div>
          ))}
          {loading && !messages.some((m) => m.streaming) && (
            <div className="msg msg-assistant loading">Assistant is searching…</div>
          )}
        </div>

        {error && <p className="error">{error}</p>}

        <form onSubmit={send}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your documents…"
            autoFocus
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim()}>
            {loading ? '…' : 'Send'}
          </button>
        </form>
      </main>

      <aside className="sidebar sidebar-right">
        <div className="sidebar-scroll">
          <SearchParamsPanel
            params={params}
            defaults={defaults}
            onChange={updateParam}
            onReset={resetParams}
            lastUsed={lastRetrieval}
          />
        </div>
        <TokenUsageBar
          lastUsage={lastUsage}
          sessionTotal={sessionTokens}
          lastSaved={lastSaved}
          sessionSaved={sessionSaved}
        />
      </aside>
    </div>
  )
}
