export type TopBarStatus = 'ready' | 'working' | 'live' | 'degraded' | 'offline'

const statusCopy: Record<TopBarStatus, { label: string; state: string }> = {
  ready: { label: 'Research workspace', state: 'Ready' },
  working: { label: 'Research workflow', state: 'Working' },
  live: { label: 'Gemini research', state: 'Live' },
  degraded: { label: 'Research service', state: 'Degraded' },
  offline: { label: 'Research service', state: 'Offline' },
}

export function TopBar({ status = 'ready' }: { status?: TopBarStatus }) {
  const copy = statusCopy[status]

  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">CC</span>
        <div>
          <strong>Commercial Court Research</strong>
          <span>Indian judgments · claim-level citation verification</span>
        </div>
      </div>
      <div className="topbar__status" aria-label="Research service status">
        <span className="corpus-pill">100 judgments</span>
        <span className={`preview-status preview-status--${status}`} role="status">
          <i aria-hidden="true" /> {copy.label} · <strong>{copy.state}</strong>
        </span>
      </div>
    </header>
  )
}
