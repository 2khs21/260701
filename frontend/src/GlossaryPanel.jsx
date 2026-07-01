import { Fragment } from 'react'

/** Split "ATC (Air Traffic Control) extra" into acronym + readable definition. */
export function normalizeGlossaryEntry({ term, definition, sources = [] }) {
  const match = term.trim().match(/^([A-Z][A-Z0-9]{1,9})\s*\(([^)]+)\)\s*(.*)$/s)
  if (!match) return { term, definition, sources }

  const [, abbr, expansion, trailing] = match
  const parts = [expansion.trim()]
  if (trailing.trim()) parts.push(trailing.trim())
  if (definition?.trim() && definition.trim().toLowerCase() !== expansion.trim().toLowerCase()) {
    parts.push(definition.trim())
  }
  return { term: abbr, definition: parts.join(' — '), sources }
}

/** Extract term/definition pairs from assistant markdown (mirrors backend logic). */
export function extractGlossaryFromText(text) {
  if (!text) return []

  const terms = []
  const seen = new Set()

  function add(term, definition) {
    term = term.trim().replace(/^\*+|\*+$/g, '')
    definition = definition.trim().replace(/^\*+|\*+$/g, '')
    if (term.length < 2 || definition.length < 3) return
    if (term.toLowerCase() === definition.toLowerCase()) return

    const normalized = normalizeGlossaryEntry({ term, definition, sources: [] })
    const key = normalized.term.toUpperCase()
    if (seen.has(key)) return
    seen.add(key)
    terms.push(normalized)
  }

  const patterns = [
    { re: /\*\*([^*]+)\*\*\s*\(([^)]+)\)/g, reverse: false },
    { re: /\*\*([^*]+)\*\*\s*[:\-—]\s*([^\n\[\]]+)/g, reverse: false },
    { re: /(?:^|\n)\s*[-*]\s*\*\*([^*]+)\*\*\s*[:\-—]\s*([^\n]+)/g, reverse: false },
    { re: /\b([A-Z][A-Z0-9]{1,9})\s*\(([^)]+)\)/g, reverse: false },
    { re: /\b([A-Za-z][\w\s-]{2,50})\s*\(([A-Z][A-Z0-9]{1,9})\)/g, reverse: true },
  ]

  for (const { re, reverse } of patterns) {
    re.lastIndex = 0
    let match
    while ((match = re.exec(text)) !== null) {
      if (reverse) add(match[2], match[1])
      else add(match[1], match[2])
    }
  }
  return terms
}

function GlossarySourceItem({ source }) {
  const meta = [
    source.part && `Part ${source.part}`,
    source.section && `§${source.section}`,
    source.chunk_index != null && `chunk ${source.chunk_index}`,
    source.citation && `[${source.citation}]`,
  ].filter(Boolean).join(' · ')

  return (
    <li className="glossary-source">
      <span className="glossary-source-file" title={source.source}>
        {source.source}
      </span>
      {meta && <span className="glossary-source-meta">{meta}</span>}
      {source.section_title && (
        <span className="glossary-source-title">{source.section_title}</span>
      )}
    </li>
  )
}

function GlossarySourceList({ sources }) {
  if (!sources?.length) return null
  return (
    <ul className="glossary-sources">
      {sources.map((source, i) => (
        <GlossarySourceItem
          key={`${source.source}-${source.chunk_index}-${i}`}
          source={source}
        />
      ))}
    </ul>
  )
}

export default function GlossaryPanel({ entries }) {
  return (
    <section className="sidebar-panel glossary-panel">
      <h2>Glossary</h2>
      <p className="panel-desc">Terms and definitions introduced in answers, with PDF source when available.</p>
      {entries.length === 0 ? (
        <p className="glossary-empty">No terms yet. Defined abbreviations like <strong>TERM</strong> (definition) will appear here.</p>
      ) : (
        <dl className="glossary-list">
          {entries.map((entry) => {
            const { term, definition, sources } = normalizeGlossaryEntry(entry)
            return (
              <Fragment key={term.toUpperCase()}>
                <dt className={term.length <= 12 ? 'glossary-term-short' : undefined}>{term}</dt>
                <dd>
                  <p className="glossary-def">{definition}</p>
                  <GlossarySourceList sources={sources} />
                </dd>
              </Fragment>
            )
          })}
        </dl>
      )}
    </section>
  )
}

export function mergeGlossary(existing, incoming) {
  const map = new Map(existing.map((e) => [e.term.toUpperCase(), e]))
  for (const item of incoming || []) {
    const normalized = normalizeGlossaryEntry(item)
    const key = normalized.term.toUpperCase()
    const prev = map.get(key)
    if (!prev) {
      map.set(key, normalized)
      continue
    }
    const prevSources = prev.sources || []
    const nextSources = normalized.sources || []
    if (nextSources.length > prevSources.length) {
      map.set(key, { ...prev, sources: nextSources })
    }
  }
  return Array.from(map.values())
}

export function glossaryFromMessages(messages) {
  let glossary = []
  for (const m of messages) {
    if (m.role !== 'assistant') continue
    if (m.glossary?.length) {
      glossary = mergeGlossary(glossary, m.glossary)
    } else if (m.text) {
      glossary = mergeGlossary(glossary, extractGlossaryFromText(m.text))
    }
  }
  return glossary
}
