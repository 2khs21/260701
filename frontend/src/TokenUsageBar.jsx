export default function TokenUsageBar({ lastUsage, sessionTotal, lastSaved, sessionSaved }) {
  const last = lastUsage?.total_tokens ?? 0
  const lastIn = lastUsage?.input_tokens ?? 0
  const lastOut = lastUsage?.output_tokens ?? 0
  const saved = lastSaved?.total ?? 0
  const savedActions = lastSaved?.actions ?? []

  return (
    <div className="token-bar" aria-live="polite">
      <div className="token-bar-row">
        <span className="token-label">This question</span>
        <span className="token-value">{last.toLocaleString()} tokens</span>
        {last > 0 && (
          <span className="token-detail">
            ({lastIn.toLocaleString()} in · {lastOut.toLocaleString()} out)
          </span>
        )}
      </div>
      {saved > 0 && (
        <div className="token-bar-row token-saved-row">
          <span className="token-label">Saved</span>
          <span className="token-value token-saved">~{saved.toLocaleString()} tokens</span>
          {savedActions.length > 0 && (
            <ul className="token-saved-list">
              {savedActions.map((a) => (
                <li key={a.code + a.label}>{a.label} (~{a.tokens_saved.toLocaleString()})</li>
              ))}
            </ul>
          )}
        </div>
      )}
      <div className="token-bar-row">
        <span className="token-label">Session total</span>
        <span className="token-value">{sessionTotal.toLocaleString()} tokens</span>
      </div>
      {sessionSaved > 0 && (
        <div className="token-bar-row">
          <span className="token-label">Session saved</span>
          <span className="token-value token-saved">~{sessionSaved.toLocaleString()} tokens</span>
        </div>
      )}
    </div>
  )
}
