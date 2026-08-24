import type { VerificationStatus } from '../types'

const statusMeta: Record<VerificationStatus, { icon: string; label: string }> = {
  SUPPORTED: { icon: 'S', label: 'SUPPORTED' },
  PARTIAL: { icon: 'P', label: 'PARTIAL' },
  UNSUPPORTED: { icon: '!', label: 'UNSUPPORTED' },
}

export function StatusBadge({ status }: { status: VerificationStatus }) {
  const meta = statusMeta[status]

  return (
    <span className={`status-badge status-badge--${status.toLowerCase()}`}>
      <span className="status-badge__icon" aria-hidden="true">{meta.icon}</span>
      {meta.label}
    </span>
  )
}
