import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

/** Render assistant markdown; keep citation markers [n] as styled spans. */
export default function MarkdownMessage({ text, citations = [] }) {
  const citationSet = new Set((citations || []).map((c) => c.n))

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <p className="md-p">{children}</p>,
        ul: ({ children }) => <ul className="md-ul">{children}</ul>,
        ol: ({ children }) => <ol className="md-ol">{children}</ol>,
        li: ({ children }) => <li className="md-li">{children}</li>,
        strong: ({ children }) => <strong className="md-strong">{children}</strong>,
        em: ({ children }) => <em>{children}</em>,
        h1: ({ children }) => <h3 className="md-h">{children}</h3>,
        h2: ({ children }) => <h3 className="md-h">{children}</h3>,
        h3: ({ children }) => <h4 className="md-h">{children}</h4>,
        code: ({ inline, children }) =>
          inline
            ? <code className="md-code-inline">{children}</code>
            : <pre className="md-pre"><code>{children}</code></pre>,
        a: ({ href, children }) => (
          <a href={href} target="_blank" rel="noreferrer">{children}</a>
        ),
        text: ({ value }) => {
          const parts = value.split(/(\[\d+\])/g)
          if (parts.length === 1) return value
          return (
            <>
              {parts.map((part, i) => {
                const m = part.match(/^\[(\d+)\]$/)
                if (m) {
                  const n = Number(m[1])
                  const valid = citationSet.has(n)
                  return (
                    <span
                      key={i}
                      className={valid ? 'cite-marker cite-valid' : 'cite-marker cite-invalid'}
                      title={valid ? `Source [${n}]` : 'Unverified citation'}
                    >
                      {part}
                    </span>
                  )
                }
                return part
              })}
            </>
          )
        },
      }}
    >
      {text}
    </ReactMarkdown>
  )
}
