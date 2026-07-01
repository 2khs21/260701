const PARAM_META = {
  top_k: {
    label: 'Top-K',
    hint: 'Maximum number of chunks to retrieve',
    min: 1,
    max: 20,
    step: 1,
    type: 'int',
  },
  max_distance: {
    label: 'Max distance',
    hint: 'Exclude chunks farther than this (lower = stricter)',
    min: 0.1,
    max: 1.0,
    step: 0.01,
    type: 'float',
  },
  fallback_max_distance: {
    label: 'Fallback distance',
    hint: 'Relaxed threshold if the first search returns nothing',
    min: 0.1,
    max: 1.0,
    step: 0.01,
    type: 'float',
  },
  max_context_chars: {
    label: 'Context chars',
    hint: 'Maximum characters sent to the model as context',
    min: 500,
    max: 12000,
    step: 100,
    type: 'int',
  },
  max_tokens: {
    label: 'Max tokens',
    hint: 'Maximum tokens for the generated answer',
    min: 128,
    max: 4096,
    step: 64,
    type: 'int',
  },
}

export default function SearchParamsPanel({ params, defaults, onChange, onReset, lastUsed }) {
  return (
    <section className="sidebar-panel">
      <div className="panel-header">
        <h2>Search parameters</h2>
        <button type="button" className="btn-reset" onClick={onReset}>Reset</button>
      </div>
      <p className="panel-desc">
        Values that affect retrieval and generation. Changes apply from the next question.
      </p>

      {Object.entries(PARAM_META).map(([key, meta]) => (
        <label key={key} className="param-row">
          <span className="param-label">
            {meta.label}
            <span className="param-hint">{meta.hint}</span>
          </span>
          <div className="param-control">
            <input
              type="range"
              min={meta.min}
              max={meta.max}
              step={meta.step}
              value={params[key]}
              onChange={(e) => onChange(key, meta.type === 'int' ? parseInt(e.target.value, 10) : parseFloat(e.target.value))}
            />
            <input
              type="number"
              min={meta.min}
              max={meta.max}
              step={meta.step}
              value={params[key]}
              onChange={(e) => onChange(key, meta.type === 'int' ? parseInt(e.target.value, 10) : parseFloat(e.target.value))}
              className="param-number"
            />
          </div>
          <span className="param-default">Default: {defaults[key]}</span>
        </label>
      ))}

      {lastUsed && (
        <details className="last-retrieval">
          <summary>Last retrieval</summary>
          {lastUsed.search_query && (
            <p className="search-query"><b>Expanded query:</b> {lastUsed.search_query}</p>
          )}
          {lastUsed.used_fallback && (
            <p className="fallback-flag">Fallback distance was used</p>
          )}
          {lastUsed.retrieval?.length > 0 ? (
            <ul>
              {lastUsed.retrieval.map((r, i) => (
                <li key={i}>
                  <span className="dist">{r.distance}</span>
                  {' '}{r.source} · chunk {r.chunk_index}
                </li>
              ))}
            </ul>
          ) : (
            <p className="no-hits">No retrieval hits</p>
          )}
        </details>
      )}
    </section>
  )
}

export { PARAM_META }
