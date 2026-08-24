import type { EvidenceItem } from '../types'

interface EvidenceCardProps {
  evidence: EvidenceItem
  selected: boolean
  onSelect: (evidenceId: string) => void
}

export function EvidenceCard({ evidence, selected, onSelect }: EvidenceCardProps) {
  const year = evidence.judgment_date?.slice(0, 4) ?? 'Date unavailable'
  const page = evidence.page_number ?? '—'

  return (
    <button
      className={`evidence-row ${selected ? 'evidence-row--active' : ''}`}
      type="button"
      aria-pressed={selected}
      aria-controls="evidence-panel"
      aria-label={`Show evidence ${evidence.evidence_id}: ${evidence.case_name}, page ${page}, paragraph ${evidence.paragraph_number}`}
      onClick={() => onSelect(evidence.evidence_id)}
    >
      <span className="evidence-row__id">{evidence.evidence_id}</span>
      <span className="evidence-row__body">
        <strong>{evidence.case_name}</strong>
        <span>
          {evidence.court ?? 'Court unavailable'} · {year} · Page {page}, Para{' '}
          {evidence.paragraph_number}
        </span>
        <small>{evidence.text}</small>
      </span>
      <span className="evidence-row__arrow" aria-hidden="true">→</span>
    </button>
  )
}
