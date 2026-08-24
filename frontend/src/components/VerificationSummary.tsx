import type { VerificationSummary as VerificationSummaryValue } from '../types'

function claimCountLabel(value: number) {
  return `${value} ${value === 1 ? 'claim' : 'claims'}`
}

export function VerificationSummary({
  claimCount,
  summary,
}: {
  claimCount: number
  summary: VerificationSummaryValue
}) {
  return (
    <div className="verification-summary" aria-label="Citation verification summary">
      <strong>{claimCountLabel(claimCount)}</strong>
      <span aria-hidden="true">/</span>
      <span className="summary-count summary-count--supported">
        {summary.supported} supported
      </span>
      <span aria-hidden="true">/</span>
      <span className="summary-count summary-count--partial">
        {summary.partial} partial
      </span>
      <span aria-hidden="true">/</span>
      <span className="summary-count summary-count--unsupported">
        {summary.unsupported} unsupported
      </span>
    </div>
  )
}
