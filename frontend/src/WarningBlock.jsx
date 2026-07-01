export default function WarningBlock({ warnings }) {
  if (!warnings?.length) return null

  return (
    <div className="warnings">
      {warnings.map((w) => (
        <div key={w.code} className={`warning-block warning-${w.severity}`}>
          <div className="warning-title">{w.title}</div>
          <p className="warning-message">{w.message}</p>
          <span className="warning-code">{w.code}</span>
        </div>
      ))}
    </div>
  )
}
