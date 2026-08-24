import type { EvidenceItem } from '../types'

interface CitationTokenProps {
  evidence: EvidenceItem
  selected?: boolean
  linked?: boolean
  onSelect: (evidenceId: string) => void
}

export function CitationToken({
  evidence,
  selected = false,
  linked = false,
  onSelect,
}: CitationTokenProps) {
  return (
    <button
      className={`citation-token ${linked ? 'citation-token--linked' : ''} ${selected ? 'citation-token--selected' : ''}`}
      type="button"
      aria-controls="evidence-panel"
      aria-pressed={selected}
      aria-label={`Show evidence ${evidence.evidence_id}: ${evidence.case_name}`}
      title={`${evidence.evidence_id} · ${evidence.case_name}`}
      onClick={() => onSelect(evidence.evidence_id)}
    >
      [{evidence.evidence_id}]
    </button>
  )
}
