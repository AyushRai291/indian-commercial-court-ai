import type { EvidenceItem } from '../types'

interface EvidenceCardProps {
  evidence: EvidenceItem
  selected: boolean
  highlighted: boolean
  onSelect: (evidenceId: string) => void
}

export function EvidenceCard({ evidence, selected, highlighted, onSelect }: EvidenceCardProps) {
  const year = evidence.judgment_date?.slice(0, 4) ?? 'Date unavailable'
  const page = evidence.page_number ?? '—'

  return (
    <button
      className={`evidence-row ${selected ? 'evidence-row--active' : ''} ${highlighted ? 'evidence-row--linked' : ''}`}
      type="button"
      aria-pressed={selected}
      aria-controls="evidence-panel"
      aria-label={`Show evidence ${evidence.evidence_id}: ${evidence.case_name}, page ${page}, paragraph ${evidence.paragraph_number}${highlighted ? '; cited by selected claim' : ''}`}
      onClick={() => onSelect(evidence.evidence_id)}
    >
      <span className="evidence-row__id">{evidence.evidence_id}</span>
      <span className="evidence-row__body">
        <strong title={evidence.case_name}>{evidence.case_name}</strong>
        <span>
          {evidence.court ?? 'Court unavailable'} · {year} · Page {page}, Para{' '}
          {evidence.paragraph_number}
        </span>
        <small>{evidence.text}</small>
      </span>
      {highlighted ? <span className="evidence-row__link-mark">Linked</span> : null}
      <span className="evidence-row__arrow" aria-hidden="true">→</span>
    </button>
  )
}
