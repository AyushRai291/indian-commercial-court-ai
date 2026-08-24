import type { EvidenceItem } from '../types'

interface CitationTokenProps {
  evidence: EvidenceItem
  onSelect: (evidenceId: string) => void
}

export function CitationToken({ evidence, onSelect }: CitationTokenProps) {
  return (
    <button
      className="citation-token"
      type="button"
      aria-controls="evidence-panel"
      aria-label={`Show evidence ${evidence.evidence_id}: ${evidence.case_name}`}
      onClick={() => onSelect(evidence.evidence_id)}
    >
      [{evidence.evidence_id}]
    </button>
  )
}
